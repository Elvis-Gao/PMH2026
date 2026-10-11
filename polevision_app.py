"""PoleVision: pole, transformer and crossarm heights from a photo.

  python polevision_app.py --live                  watch the Arduino sensors (Ctrl+C to stop)
  python polevision_app.py --record                watch AND save every reading to sensor_log.csv
  python polevision_app.py --clock-check IMG.jpg --shown 02:41:15
                                                   find how far the phone clock is off
  python polevision_app.py photos --log sensor_log.csv --clock-offset 0
                                                   match each phone photo to the reading at its timestamp
  python polevision_app.py photo.jpg               one photo (uses Arduino if plugged in)
  python polevision_app.py photo.jpg --distance 15 typed distance if there's no radar reading
  python polevision_app.py photos                  every photo in a folder (no live Arduino)
  python polevision_app.py photo.jpg --no-arduino  photo only

Sensor data per photo, in this order:
  1. photo.json saved next to the photo (from an earlier run, a log match, or the Pi)
  2. --log: the logged reading matching the photo's timestamp, then saved as photo.json
  3. live Arduino reading (waits until the rig is STILL), then saved as photo.json
  4. --distance / --pitch you type
  5. photo only: level shot at --cam-height (default 1.5 m)
"""
import argparse, csv, glob, json, math, os, time
from datetime import datetime, timedelta
import cv2, numpy as np
from PIL import Image, ImageOps

ELECTRIC = ("transformer", "cross", "insulator")
THIRD_PARTY = ("comm", "telecom", "fiber", "fibre", "splice", "terminal")
CLEARANCE_M = 1.02          # supply-to-comm separation to flag (about 40 in); verify vs NESC


# ======================= Arduino =======================
def find_port():
    from serial.tools import list_ports
    for p in list_ports.comports():
        t = f"{p.description} {p.manufacturer or ''}".lower()
        if p.vid == 0x2341 or "arduino" in t or "nano" in t:
            return p.device
    return None


class Arduino:
    def __init__(self, port):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=2)
        self.port = port
        time.sleep(2)
        self.ser.reset_input_buffer()

    def read(self, timeout=3):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.ser.readline().decode("ascii", "ignore").strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "err" in d:
                print("NANO ERROR:", d["err"])
            if "pitch" in d:
                return d
        return None

    def wait_still(self, timeout=10):
        self.ser.reset_input_buffer()
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            d = self.read()
            if not d:
                continue
            last = d
            dist = f"{d['dist_m']:.2f} m" if d["dist_m"] > 0 else "no radar"
            print(f"   pitch {d['pitch']:+6.1f} deg  roll {d['roll']:+5.1f} deg  "
                  f"{'STILL' if d['still'] else 'hold still...'}  {dist}")
            if d["still"]:
                return d
        return last

    def close(self):
        self.ser.close()


def open_arduino(port=None):
    try:
        port = port or find_port()
        if not port:
            return None, "Arduino not found"
        return Arduino(port), None
    except Exception as e:
        return None, f"can't open Arduino ({e}); close Serial Monitor"


# ======================= photo + model =======================
def load_photo(path):
    pil = Image.open(path)
    f35 = None
    try:
        ex = pil.getexif()
        f35 = ex.get_ifd(0x8769).get(41989) or ex.get(41989)
    except Exception:
        pass
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return pil, (float(f35) if f35 else None)


def iou(a, b):
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0


def detect(models, pil, conf, imgsz):
    dets = []
    for m in models:
        r = m(pil, conf=conf, imgsz=imgsz, verbose=False)[0]
        polys = r.masks.xy if getattr(r, "masks", None) is not None else None   # segmentation model
        for k, (b, s, c) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())):
            d = {"cls": r.names[int(c)].lower(), "conf": float(s),
                 "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
            if polys is not None and k < len(polys) and len(polys[k]) >= 3:
                poly = np.asarray(polys[k], dtype=float)
                d["poly"] = poly.tolist()
                # outline gives the object's true top/bottom -> tighter edges for the height math
                d["x1"], d["y1"] = float(poly[:, 0].min()), float(poly[:, 1].min())
                d["x2"], d["y2"] = float(poly[:, 0].max()), float(poly[:, 1].max())
            dets.append(d)
    dets.sort(key=lambda d: -d["conf"])
    kept = []
    for d in dets:
        if all(not (d["cls"] == k["cls"] and iou(d, k) > 0.5) for k in kept):
            kept.append(d)
    return kept


def role(cls):
    if any(k in cls for k in THIRD_PARTY) or "cable box" in cls:
        return "third-party"
    if any(k in cls for k in ELECTRIC):
        return "electric"
    if "pole" in cls:
        return "pole"
    if any(k in cls for k in ("wire", "line", "cable", "conductor")):
        return "wire"
    return "other"


# ======================= timestamps =======================
def photo_time(path):
    """When the photo was taken (EXIF DateTimeOriginal + sub-seconds), or None."""
    try:
        ex = Image.open(path).getexif()
        sub = ex.get_ifd(0x8769)
        t = sub.get(36867) or ex.get(306)
        if not t:
            return None
        dt = datetime.strptime(str(t).strip(), "%Y:%m:%d %H:%M:%S")
        ss = sub.get(37521)
        if ss:
            digits = "".join(ch for ch in str(ss) if ch.isdigit())
            if digits:
                dt += timedelta(seconds=float("0." + digits))
        return dt
    except Exception:
        return None


def load_log(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((datetime.fromisoformat(r["time"]), r))
            except (KeyError, ValueError):
                continue
    return rows


def match_log(rows, t_photo, max_gap=1.5):
    """Reading closest to the photo time; prefer a STILL one within max_gap seconds."""
    near = [(abs((t - t_photo).total_seconds()), t, r) for t, r in rows]
    near = [x for x in near if x[0] <= max_gap]
    if not near:
        return None, None
    still = [x for x in near if str(x[2].get("still")).lower() == "true"]
    gap, t, r = min(still or near, key=lambda x: x[0])
    return r, gap


# ======================= sensors per photo =======================
def sensor_for_photo(photo, args, ard, batch, log_rows=None):
    """Returns dict: pitch, pitch_src, roll, still, dist, dist_src, f35, time."""
    s = {"pitch": None, "pitch_src": None, "roll": None, "still": None,
         "dist": None, "dist_src": None, "f35": None, "time": None}
    side = os.path.splitext(photo)[0] + ".json"

    if log_rows is not None and not os.path.exists(side):
        tp = photo_time(photo)
        if tp is None:
            print(f"   {os.path.basename(photo)}: no EXIF time (send the ORIGINAL photo, not a screenshot/chat copy)")
        else:
            t_adj = tp - timedelta(seconds=args.clock_offset)
            r, gap = match_log(log_rows, t_adj)
            if r is None:
                print(f"   {os.path.basename(photo)}: no sensor reading within 1.5 s of {t_adj:%H:%M:%S} "
                      f"(was --record running? is --clock-offset right?)")
            else:
                with open(side, "w") as fh:
                    json.dump({"time": r["time"], "photo_time": t_adj.isoformat(timespec="milliseconds"),
                               "gap_s": round(gap, 2), "pitch": float(r["pitch"]), "roll": float(r["roll"]),
                               "still": str(r["still"]).lower() == "true", "dist_m": float(r["dist_m"])},
                              fh, indent=2)
                print(f"   {os.path.basename(photo)}: matched reading at {r['time']} ({gap:.2f} s from photo)")

    if os.path.exists(side):
        with open(side) as fh:
            j = json.load(fh)
        if j.get("pitch") is not None:
            s.update(pitch=float(j["pitch"]), pitch_src="saved reading", roll=j.get("roll"),
                     still=j.get("still"), time=j.get("time"), f35=j.get("f35"))
            if float(j.get("dist_m", -1)) > 0:
                s.update(dist=float(j["dist_m"]), dist_src="saved radar")
    elif ard and not batch:
        print(f"Arduino on {ard.port}: aim at the pole and hold the rig still...")
        d = ard.wait_still()
        if d:
            s.update(pitch=float(d["pitch"]), pitch_src="Arduino", roll=float(d["roll"]),
                     still=bool(d["still"]), time=datetime.now().isoformat(timespec="seconds"))
            if d["dist_m"] > 0:
                s.update(dist=float(d["dist_m"]), dist_src="radar")
            with open(side, "w") as fh:
                json.dump({"time": s["time"], "pitch": s["pitch"], "roll": s["roll"], "still": s["still"],
                           "dist_m": s["dist"] or -1}, fh, indent=2)

    if s["pitch"] is None and args.pitch is not None:
        s.update(pitch=args.pitch, pitch_src="typed")
    if s["dist"] is None and args.distance is not None:
        s.update(dist=args.distance, dist_src="typed")
    if s["pitch"] is None:
        s.update(pitch=0.0, pitch_src="assumed level")
    return s


# ======================= geometry =======================
def make_solver(W, H, f35, cam_h):
    fy = f35 * max(W, H) / 36.0
    cy = H / 2.0

    def tan_e(row, pitch):
        return math.tan(math.radians(pitch) + math.atan((cy - row) / fy))

    def solve(p):
        """p: pitch, base, row, D (or None), hcam -> (height, distance) or (None, None)"""
        tb = tan_e(p["base"], p["pitch"])
        D = p["D"]
        if D is None:
            if tb > -0.01:
                return None, None
            D = -p["hcam"] / tb
        return D * (tan_e(p["row"], p["pitch"]) - tb), D

    def height(row, base, pitch, D, sig):
        p = {"pitch": pitch, "base": base, "row": row, "D": D, "hcam": cam_h}
        h, dist = solve(p)
        if h is None:
            return None, None, None
        var = 0.0
        for k, sd in sig.items():
            if sd and p.get(k) is not None:
                q = dict(p)
                q[k] = p[k] + sd
                h2, _ = solve(q)
                if h2 is not None:
                    var += (h2 - h) ** 2
        return h, math.sqrt(var), dist

    return height


# ======================= one photo =======================
def analyze(photo, models, args, ard, batch=False, log_rows=None):
    s = sensor_for_photo(photo, args, ard, batch, log_rows)
    pil, exif_f35 = load_photo(photo)
    W, H = pil.size
    f35 = float(s["f35"] or exif_f35 or 26.0)
    height = make_solver(W, H, f35, args.cam_height)

    dets = detect(models, pil, args.conf, args.imgsz)
    for d in dets:
        d["role"] = role(d["cls"])
    poles = [d for d in dets if d["role"] == "pole"]

    sig = {"pitch": {"Arduino": 0.5, "saved reading": 0.5, "typed": 1.0}.get(s["pitch_src"], 2.0),
           "base": 3.0, "row": 3.0,
           "D": {"radar": 0.15, "saved radar": 0.15, "typed": 0.3}.get(s["dist_src"]),
           "hcam": 0.10 if s["dist"] is None else None}

    flags = []
    out = {"photo": photo, "sensors": s, "flags": flags, "items": [], "focal_35mm": f35}
    gray = cv2.cvtColor(np.array(pil.resize((640, max(1, int(640 * H / W))))), cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 60:
        flags.append("photo looks blurry")
    if s["still"] is False:
        flags.append("rig was moving during the reading")
    if s["roll"] is not None and abs(float(s["roll"])) > 3:
        flags.append(f"camera rolled {float(s['roll']):.1f} deg (keep it under 3)")

    if not poles:
        flags.append("no pole detected")
        return out, pil, dets, None
    pole = max(poles, key=lambda d: d["conf"] * (d["y2"] - d["y1"]))
    base, top = pole["y2"], pole["y1"]
    if base >= H - 3:
        flags.append("pole base touches image bottom -> heights unreliable")
    if top <= 3:
        flags.append("pole top cut off -> real pole is taller")

    ph, ps, D = height(top, base, s["pitch"], s["dist"], sig)
    if ph is None:
        flags.append("pole base is not below eye level: stand farther back, hold level, or give --distance")
        return out, pil, dets, pole
    out["pole"] = {"height_m": round(ph, 2), "sigma_m": round(ps, 2), "height_ft": round(ph * 3.2808, 1),
                   "distance_m": round(D, 2)}

    pw = max(pole["x2"] - pole["x1"], 8)
    near = [d for d in dets if d is not pole and d["role"] != "pole"
            and d["x2"] >= pole["x1"] - 3 * pw and d["x1"] <= pole["x2"] + 3 * pw]
    for d in near:
        if d["role"] == "wire":
            continue
        hc, sc, _ = height((d["y1"] + d["y2"]) / 2, base, s["pitch"], s["dist"], sig)
        hb, _, _ = height(d["y2"], base, s["pitch"], s["dist"], sig)
        ht, _, _ = height(d["y1"], base, s["pitch"], s["dist"], sig)
        if hc is None:
            continue
        out["items"].append({"cls": d["cls"], "role": d["role"], "conf": round(d["conf"], 2),
                             "height_m": round(hc, 2), "sigma_m": round(sc, 2),
                             "bottom_m": round(hb, 2), "top_m": round(ht, 2), "box": d})
    out["items"].sort(key=lambda it: -it["height_m"])
    out["wires_seen"] = sum(1 for d in near if d["role"] == "wire")

    elec = [it for it in out["items"] if it["role"] == "electric"]
    if elec:
        supply_bottom = min(it["bottom_m"] for it in elec)
        for it in out["items"]:
            if it["role"] == "third-party":
                gap = supply_bottom - it["top_m"]
                it["clearance_m"] = round(gap, 2)
                if gap < CLEARANCE_M:
                    flags.append(f"CLEARANCE: {it['cls']} at {it['height_m']} m is only "
                                 f"{gap:.2f} m below electric equipment")

    rel = ps / ph if ph else 1
    bad = any(("unreliable" in f or "moving" in f or "blurry" in f) for f in flags)
    out["confidence"] = "low" if bad or rel > 0.10 else ("high" if rel <= 0.05 else "medium")
    return out, pil, dets, pole


# ======================= output =======================
def report(out):
    s = out["sensors"]
    print("\n" + "=" * 66)
    print(f"PHOTO      {out['photo']}")
    src = [f"tilt: {s['pitch_src']} ({s['pitch']:+.1f} deg)"]
    src.append(f"distance: {s['dist_src']} ({s['dist']:.2f} m)" if s["dist"] else "distance: camera-height estimate")
    if s.get("time"):
        src.append(f"reading at {s['time']}")
    print("DATA       " + " | ".join(src))
    print("=" * 66)
    if "pole" in out:
        p = out["pole"]
        print(f"TOTAL POLE HEIGHT   {p['height_m']:.2f} m +/- {p['sigma_m']:.2f}   ({p['height_ft']} ft)")
        print(f"  distance {p['distance_m']} m   confidence {out['confidence']}")
        for title, key in (("TRANSFORMERS", "transformer"), ("CROSSARMS", "cross")):
            found = [it for it in out["items"] if key in it["cls"]]
            print(f"\n{title}: {'none detected' if not found else ''}")
            for i, it in enumerate(sorted(found, key=lambda x: -x["height_m"]), 1):
                print(f"  #{i}  {it['height_m']:.2f} m +/- {it['sigma_m']:.2f} ({it['height_m'] * 3.2808:.1f} ft)"
                      f"   bottom {it['bottom_m']:.2f}   top {it['top_m']:.2f}")
        others = [it for it in out["items"] if "transformer" not in it["cls"] and "cross" not in it["cls"]]
        if others:
            print("\nOTHER EQUIPMENT:")
            for it in others:
                extra = f"   clearance {it['clearance_m']} m" if "clearance_m" in it else ""
                print(f"  {it['cls']:<14} {it['height_m']:.2f} m +/- {it['sigma_m']:.2f}   ({it['role']}){extra}")
    else:
        print("No height could be computed.")
    for f in out["flags"]:
        print("FLAG:", f)


COLORS = {"pole": (255, 255, 255), "electric": (40, 40, 230), "third-party": (255, 160, 0),
          "wire": (160, 160, 160), "other": (0, 200, 255)}


def draw_label(img, text, x, y, color, sc):
    """Text on a solid colored tag so it's readable on any background."""
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc, max(1, int(2 * sc))
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    H, W = img.shape[:2]
    x = int(min(max(0, x), W - tw - 8))
    y = int(max(tht + 8, y))
    cv2.rectangle(img, (x, y - tht - 8), (x + tw + 8, y + base - 2), color, -1)
    txt_col = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(img, text, (x + 4, y - 4), font, fs, txt_col, th, cv2.LINE_AA)


def save(out, pil, dets, pole, photo):
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    sc = max(W, H) / 1500
    heights = {id(it["box"]): it for it in out.get("items", [])}

    # 1) translucent highlight over every detected object (its outline if the model gives one)
    overlay = img.copy()
    for d in dets:
        col = COLORS.get(d["role"], (0, 255, 0))
        if "poly" in d:
            cv2.fillPoly(overlay, [np.asarray(d["poly"], dtype=np.int32)], col)
        else:
            cv2.rectangle(overlay, (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"])), col, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # 2) outline + label (class, confidence, height if measured) on every object
    for d in dets:
        col = COLORS.get(d["role"], (0, 255, 0))
        if "poly" in d:
            cv2.polylines(img, [np.asarray(d["poly"], dtype=np.int32)], True, col, max(2, int(3 * sc)), cv2.LINE_AA)
        else:
            cv2.rectangle(img, (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"])), col, max(2, int(3 * sc)))
        label = f"{d['cls']} {d['conf']:.2f}"
        if d is pole and "pole" in out:
            label = f"POLE {out['pole']['height_m']:.1f} m  ({d['conf']:.2f})"
        elif id(d) in heights:
            label = f"{d['cls']} {heights[id(d)]['height_m']:.1f} m  ({d['conf']:.2f})"
        draw_label(img, label, d["x1"], d["y1"] - 4, col, sc)

    # 3) headline + legend
    if "pole" in out:
        draw_label(img, f"POLE HEIGHT {out['pole']['height_m']:.2f} +/- {out['pole']['sigma_m']:.2f} m"
                        f"  ({out['pole']['height_ft']} ft)", 10, int(45 * sc), (255, 255, 255), sc * 1.3)
    else:
        draw_label(img, "NO POLE HEIGHT: " + (out["flags"][0] if out["flags"] else ""), 10, int(45 * sc),
                   (40, 40, 230), sc * 1.1)
    ly = int(95 * sc)
    for name, key in (("pole", "pole"), ("electric", "electric"), ("third-party", "third-party"), ("wire", "wire")):
        draw_label(img, name, 10, ly, COLORS[key], sc * 0.9)
        ly += int(38 * sc)

    stem = os.path.splitext(photo)[0]
    cv2.imwrite(stem + "_heights.jpg", img)
    clean = dict(out)
    clean["items"] = [{k: v for k, v in it.items() if k != "box"} for it in out.get("items", [])]
    with open(stem + "_report.json", "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"\nSaved: {stem}_heights.jpg  and  {stem}_report.json")


def list_photos(folder):
    found = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        stem, ext = os.path.splitext(p)
        if ext.lower() in (".jpg", ".jpeg", ".png") and not stem.endswith("_heights"):
            found.append(p)
    return found


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="a photo, or a folder of photos")
    ap.add_argument("--live", action="store_true", help="just show live Arduino readings")
    ap.add_argument("--model", nargs="+", default=["best.pt"])
    ap.add_argument("--imgsz", type=int, default=960, help="match the size you trained at")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--distance", type=float, help="meters to the pole (backup if no radar)")
    ap.add_argument("--pitch", type=float, help="camera tilt in degrees, up = + (backup if no Arduino)")
    ap.add_argument("--cam-height", type=float, default=1.5)
    ap.add_argument("--port")
    ap.add_argument("--no-arduino", action="store_true")
    ap.add_argument("--record", action="store_true", help="show live readings and save them to --log-file")
    ap.add_argument("--log-file", default="sensor_log.csv", help="where --record saves readings")
    ap.add_argument("--log", help="match photos to readings in this log (from --record)")
    ap.add_argument("--clock-offset", type=float, default=0.0,
                    help="seconds the PHONE clock is ahead of the laptop (from --clock-check)")
    ap.add_argument("--clock-check", help="photo of the laptop screen taken during --record")
    ap.add_argument("--shown", help="the HH:MM:SS time visible on the screen in that photo")
    a = ap.parse_args()

    if a.clock_check:
        if not a.shown:
            raise SystemExit("Add --shown HH:MM:SS (the time visible on the laptop screen in the photo)")
        tp = photo_time(a.clock_check)
        if tp is None:
            raise SystemExit("That photo has no EXIF time. Copy the ORIGINAL photo (cable/AirDrop/Google Photos download).")
        h, m, sec = map(int, a.shown.split(":"))
        shown = tp.replace(hour=h, minute=m, second=sec, microsecond=0)
        off = (tp - shown).total_seconds()
        if off > 43200: off -= 86400
        if off < -43200: off += 86400
        print(f"Photo says {tp:%H:%M:%S.%f}"[:-3] + f", laptop screen showed {a.shown}")
        print(f"Phone clock is {off:+.1f} s vs the laptop.")
        print(f"Use:  --clock-offset {off:.1f}")
        print("(The screen only shows whole seconds, so this is good to about +/-1 s. Hold the rig still")
        print(" for 2-3 seconds around every shot and the match will land on a STILL reading.)")
        return

    if a.live or a.record:
        ard, why = open_arduino(a.port)
        if not ard:
            raise SystemExit(why)
        fh = None
        if a.record:
            new = not os.path.exists(a.log_file)
            fh = open(a.log_file, "a", newline="")
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "pitch", "roll", "still", "dist_m", "pitch_range"])
            print(f"Arduino on {ard.port}. RECORDING to {a.log_file}. Ctrl+C to stop.")
        else:
            print(f"Arduino on {ard.port}. Ctrl+C to stop.")
        try:
            while True:
                d = ard.read()
                if not d:
                    continue
                now = datetime.now()
                if fh:
                    w.writerow([now.isoformat(timespec="milliseconds"), d["pitch"], d["roll"],
                                "true" if d["still"] else "false", d["dist_m"], d.get("pitch_range", "")])
                    fh.flush()
                dist = f"{d['dist_m']:.2f} m" if d["dist_m"] > 0 else "no radar"
                print(f"{now:%H:%M:%S}  pitch {d['pitch']:+6.1f} deg  roll {d['roll']:+5.1f} deg  "
                      f"{'STILL ' if d['still'] else 'moving'}  distance {dist}")
        except KeyboardInterrupt:
            pass
        if fh:
            fh.close()
        ard.close()
        return

    if not a.target:
        ap.error("give a photo or folder, or use --live / --record")
    if not os.path.exists(a.target):
        raise SystemExit(f"Not found: {a.target}")
    for m in a.model:
        if not os.path.exists(m):
            raise SystemExit(f"Model file not found: {m}  (put best.pt in this folder)")

    from ultralytics import YOLO
    models = [YOLO(m) for m in a.model]

    batch = os.path.isdir(a.target)
    photos = list_photos(a.target) if batch else [a.target]
    if not photos:
        raise SystemExit(f"No .jpg/.png photos in {a.target}")

    log_rows = None
    if a.log:
        if not os.path.exists(a.log):
            raise SystemExit(f"Log not found: {a.log}")
        log_rows = load_log(a.log)
        print(f"Loaded {len(log_rows)} sensor readings from {a.log} (clock offset {a.clock_offset:+.1f} s)")

    ard = None
    if not a.no_arduino and not batch and log_rows is None:
        ard, why = open_arduino(a.port)
        if not ard:
            print(f"({why}: using saved readings / typed values / photo only)")

    for photo in photos:
        out, pil, dets, pole = analyze(photo, models, a, ard, batch, log_rows)
        report(out)
        save(out, pil, dets, pole, photo)
    if ard:
        ard.close()


if __name__ == "__main__":
    main()
