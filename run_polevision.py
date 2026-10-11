"""PoleVision - easy mode. Answers a few questions, then shows everything it found.

Run:   python run_polevision.py        (or double-click RUN_ME.bat on Windows)

For every photo you get:
  - a table of EVERY object it marked: type, confidence %, height, owner, notes
  - photoName_heights.jpg   annotated picture
  - photoName_summary.json  all results with easy camelCase names
  - allResults.csv          one row per object, every photo (open in Excel)
"""
import argparse, csv, json, os, sys
from polevision_app import (analyze, save, open_arduino, list_photos)

LOW_CONF = 0.30      # below this we tag the detection as lowConfidence


def ask(question, default=None, cast=str):
    """Ask a question; Enter keeps the default."""
    while True:
        hint = f" [Enter = {default}]" if default is not None else ""
        raw = input(f"{question}{hint}: ").strip().strip('"').strip("'")
        if raw == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            print("   Please type a number (like 12.5) or just press Enter.")


def yes(question, default=False):
    raw = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if raw == "" else raw.startswith("y")


def owner_name(role):
    return {"pole": "Pole", "electric": "Electric utility", "third-party": "Third-party (telecom/cable)",
            "wire": "Wire", "other": "Unknown"}.get(role, "Unknown")


def build_rows(photo, out, dets, pole):
    by_box = {id(it["box"]): it for it in out.get("items", [])}
    rows = []
    for d in sorted(dets, key=lambda d: d["y1"]):
        notes = []
        h = sig = None
        on_pole = d is pole or id(d) in by_box
        if d is pole and "pole" in out:
            h, sig = out["pole"]["height_m"], out["pole"]["sigma_m"]
        elif id(d) in by_box:
            it = by_box[id(d)]
            h, sig = it["height_m"], it["sigma_m"]
            if "clearance_m" in it:
                notes.append(f"clearanceToElectric {it['clearance_m']} m")
        elif d["role"] == "wire":
            notes.append("wiresAreNotMeasured")
        else:
            notes.append("notOnThisPole")
        if d["conf"] < LOW_CONF:
            notes.append("lowConfidence")
        rows.append({
            "photoName": os.path.basename(photo),
            "objectType": d["cls"],
            "confidencePercent": round(d["conf"] * 100, 1),
            "heightMeters": h,
            "heightFeet": round(h * 3.2808, 1) if h is not None else None,
            "plusMinusMeters": sig,
            "owner": owner_name(d["role"]),
            "isOnThisPole": on_pole,
            "notes": ", ".join(notes),
        })
    return rows


def print_table(rows):
    print(f"\n  {'#':<3}{'objectType':<14}{'confidence':>11}{'heightMeters':>14}{'+/-':>7}  {'owner':<28}notes")
    print("  " + "-" * 96)
    for i, r in enumerate(rows, 1):
        h = f"{r['heightMeters']:.2f}" if r["heightMeters"] is not None else "-"
        pm = f"{r['plusMinusMeters']:.2f}" if r["plusMinusMeters"] is not None else "-"
        print(f"  {i:<3}{r['objectType']:<14}{r['confidencePercent']:>10.1f}%{h:>14}{pm:>7}  {r['owner']:<28}{r['notes']}")
    if not rows:
        print("  (nothing detected - try a clearer photo)")


def main():
    if not os.path.exists("best.pt"):
        sys.exit("best.pt not found. Put your trained model in this folder (same place as this file).")
    print("=" * 60)
    print(" POLEVISION - pole + equipment heights from a photo")
    print("=" * 60)

    model_path = ask("Model file", "best.pt")
    target = ask("Photo or folder (you can drag it into this window)", "photos")
    if not os.path.exists(target):
        sys.exit(f"Not found: {target}")
    photos = list_photos(target) if os.path.isdir(target) else [target]
    if not photos:
        sys.exit(f"No .jpg/.png photos in {target}")
    conf = ask("Lowest confidence to show, as a percent", 10.0, float) / 100.0

    ard = None
    if yes("Is the Arduino plugged in and running?"):
        ard, why = open_arduino(None)
        print(f"   Arduino on {ard.port}" if ard else f"   {why} -> you'll type the values instead")

    same = None
    if not ard and len(photos) > 1:
        if yes("Use the SAME tilt and distance for every photo?"):
            print("\nEnter the values (Enter skips a value):")
            same = (ask("  Camera tilt (pitch) in degrees, up is +", 0.0, float),
                    ask("  Distance to the pole in meters", None, float),
                    ask("  Camera height above ground in meters", 1.5, float))

    from ultralytics import YOLO
    model = YOLO(model_path)
    csv_path = "allResults.csv"
    new_csv = not os.path.exists(csv_path)
    fields = ["photoName", "objectType", "confidencePercent", "heightMeters", "heightFeet",
              "plusMinusMeters", "owner", "isOnThisPole", "notes"]

    for photo in photos:
        print("\n" + "=" * 60)
        print(f" PHOTO: {os.path.basename(photo)}")
        print("=" * 60)
        saved = os.path.splitext(photo)[0] + ".json"
        pitch = dist = None
        cam_h = 1.5
        if os.path.exists(saved):
            print("   Using the saved sensor reading next to this photo.")
        elif not ard:
            if same:
                pitch, dist, cam_h = same
            else:
                print("Enter the values for this photo (Enter skips a value):")
                pitch = ask("  Camera tilt (pitch) in degrees, up is +", 0.0, float)
                dist = ask("  Distance to the pole in meters", None, float)
                cam_h = ask("  Camera height above ground in meters", 1.5, float)

        args = argparse.Namespace(pitch=pitch, distance=dist, cam_height=cam_h, conf=conf, imgsz=960,
                                  clock_offset=0.0)
        out, pil, dets, pole = analyze(photo, [model], args, ard, batch=False)

        s = out["sensors"]
        print(f"\n  dataUsed: tilt {s['pitch']:+.1f} deg ({s['pitch_src']}), "
              + (f"distance {s['dist']:.2f} m ({s['dist_src']})" if s["dist"] else "distance estimated from camera height"))
        if "pole" in out:
            p = out["pole"]
            print(f"\n  POLE HEIGHT:  {p['height_m']:.2f} m  +/- {p['sigma_m']:.2f}   ({p['height_ft']} ft)"
                  f"   confidenceLevel: {out['confidence']}")
        rows = build_rows(photo, out, dets, pole)
        print_table(rows)
        for f in out["flags"]:
            print(f"  FLAG: {f}")

        save(out, pil, dets, pole, photo)
        summary = {
            "photoName": os.path.basename(photo),
            "dataUsed": {"tiltSource": s["pitch_src"], "tiltDegrees": s["pitch"],
                         "distanceSource": s["dist_src"] or "cameraHeightEstimate", "distanceMeters": s["dist"]},
            "poleHeightMeters": out.get("pole", {}).get("height_m"),
            "poleHeightFeet": out.get("pole", {}).get("height_ft"),
            "poleHeightPlusMinusMeters": out.get("pole", {}).get("sigma_m"),
            "confidenceLevel": out.get("confidence"),
            "detections": rows,
            "flags": out["flags"],
        }
        stem = os.path.splitext(photo)[0]
        with open(stem + "_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new_csv:
                w.writeheader()
                new_csv = False
            w.writerows(rows)
        print(f"  Saved {os.path.basename(stem)}_summary.json")

    if ard:
        ard.close()
    print(f"\nDone. Every object from every photo is also in {csv_path} (opens in Excel).")


if __name__ == "__main__":
    main()
