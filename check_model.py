"""Check how well your trained model works on your own photos.

  python check_model.py photos
  python check_model.py photos --conf 0.2      (show weaker detections too)

For every photo it saves photos/checks/<name>_check.jpg with every detection highlighted
and labeled, prints what it found, and ends with a per-class summary.
"""
import argparse, collections, glob, os
import cv2, numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--model", default="best.pt")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--conf", type=float, default=0.1)
a = ap.parse_args()

if not os.path.exists(a.model):
    raise SystemExit(f"Model file not found: {a.model}  (put best.pt in this folder)")
model = YOLO(a.model)
print("Classes the model knows:", list(model.names.values()))

photos = [p for p in sorted(glob.glob(os.path.join(a.folder, "*")))
          if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
          and not os.path.splitext(p)[0].endswith(("_heights", "_check"))]
if not photos:
    raise SystemExit(f"No photos in {a.folder}")

out_dir = os.path.join(a.folder, "checks")
os.makedirs(out_dir, exist_ok=True)
seen_in = collections.Counter()
confs = collections.defaultdict(list)
palette = [(255, 255, 255), (40, 40, 230), (0, 200, 0), (255, 160, 0), (0, 200, 255), (200, 0, 200)]


def draw_label(img, text, x, y, color, sc):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc, max(1, int(2 * sc))
    (tw, tht), base = cv2.getTextSize(text, font, fs, th)
    H, W = img.shape[:2]
    x = int(min(max(0, x), W - tw - 8))
    y = int(max(tht + 8, y))
    cv2.rectangle(img, (x, y - tht - 8), (x + tw + 8, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), font, fs, (0, 0, 0) if sum(color) > 380 else (255, 255, 255), th, cv2.LINE_AA)


for p in photos:
    pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    r = model(pil, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    sc = max(img.shape[:2]) / 1500
    boxes = list(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()))
    polys = r.masks.xy if getattr(r, "masks", None) is not None else None   # segmentation model -> outlines
    overlay = img.copy()
    for k, (b, s, c) in enumerate(boxes):
        col = palette[int(c) % len(palette)]
        if polys is not None and k < len(polys) and len(polys[k]) >= 3:
            cv2.fillPoly(overlay, [np.asarray(polys[k], dtype=np.int32)], col)
        else:
            cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    found = collections.Counter()
    for k, (b, s, c) in enumerate(boxes):
        name = model.names[int(c)]
        found[name] += 1
        confs[name].append(s)
        col = palette[int(c) % len(palette)]
        if polys is not None and k < len(polys) and len(polys[k]) >= 3:
            cv2.polylines(img, [np.asarray(polys[k], dtype=np.int32)], True, col, max(2, int(3 * sc)), cv2.LINE_AA)
        else:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, max(2, int(3 * sc)))
        draw_label(img, f"{name} {s:.2f}", b[0], b[1] - 4, col, sc)
    for name in found:
        seen_in[name] += 1
    stem = os.path.splitext(os.path.basename(p))[0]
    cv2.imwrite(os.path.join(out_dir, stem + "_check.jpg"), img)
    summary = ", ".join(f"{n} x{k}" for n, k in found.items()) or "NOTHING DETECTED"
    print(f"{os.path.basename(p):<30} {summary}")

n = len(photos)
print("\n" + "=" * 60)
print(f"SUMMARY over {n} photos (conf >= {a.conf})")
print("=" * 60)
for name in model.names.values():
    k = seen_in[name]
    avg = sum(confs[name]) / len(confs[name]) if confs[name] else 0
    print(f"  {name:<12} found in {k}/{n} photos   avg confidence {avg:.2f}")
print(f"\nOpen {out_dir} and look at the boxes.")
print("Good model: pole found in nearly every photo, boxes tight from ground to top,")
print("transformers/crossarms boxed when they're visible, few boxes on random objects.")
