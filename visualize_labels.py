"""Draw YOLO bounding boxes on a few images to verify conversion."""
import random
from pathlib import Path
import cv2

DATASET   = Path("data/hagrid_yolo")
SPLIT     = "train"          # change to val / test as needed
N_SAMPLES = 8                # how many images to check
OUT_DIR   = Path("bbox_check")
OUT_DIR.mkdir(exist_ok=True)

COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}   # green=palm, red=two
NAMES  = {0: "palm", 1: "two"}

img_dir = DATASET / "images" / SPLIT
lbl_dir = DATASET / "labels" / SPLIT

samples = random.sample(list(lbl_dir.glob("*.txt")), N_SAMPLES)

for lf in samples:
    img_path = img_dir / f"{lf.stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read {img_path}")
        continue

    H, W = img.shape[:2]

    for line in lf.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls        = int(parts[0])
        cx, cy, w, h = [float(x) for x in parts[1:]]

        # Convert normalized center form -> pixel corners
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        color = COLORS.get(cls, (255, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, NAMES.get(cls, str(cls)),
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = OUT_DIR / f"{lf.stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path}")

print(f"\nDone. Open the '{OUT_DIR}' folder to inspect.")