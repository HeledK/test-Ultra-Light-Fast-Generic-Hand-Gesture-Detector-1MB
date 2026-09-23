#!/usr/bin/env python3
"""
Convert HaGRID JSON annotations (2-class: palm, two) to YOLO format.

HaGRID bbox: [top_left_x, top_left_y, width, height], all normalized [0,1]
YOLO  bbox:  class_id center_x center_y width height, all normalized [0,1]

Layout produced:
    dst/
        images/{train,val,test}/*.jpg   (symlinks or copies)
        labels/{train,val,test}/*.txt
        data.yaml
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

CLASS_MAP = {"palm": 0, "two": 1}


def coco_to_yolo(bbox):
    """[x_tl, y_tl, w, h] (normalized) -> [cx, cy, w, h] (normalized)."""
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    # Clamp for safety (a few HaGRID boxes occasionally drift slightly past 1.0)
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w  = min(max(w,  0.0), 1.0)
    h  = min(max(h,  0.0), 1.0)
    return cx, cy, w, h


def link_or_copy(src: Path, dst: Path, use_symlink: bool) -> str:
    """Returns one of 'exists', 'symlink', 'copy'."""
    if dst.exists() or dst.is_symlink():
        return "exists"
    if use_symlink:
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except (OSError, NotImplementedError):
            # Windows without Developer Mode / admin -> fall back to copy
            pass
    shutil.copy2(src, dst)
    return "copy"


def process_split(sources, split_name, out_root: Path, use_symlink: bool):
    """sources: list of (json_path, image_dir) tuples."""
    images_out = out_root / "images" / split_name
    labels_out = out_root / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Merge per uuid in case an image appears in multiple JSONs
    combined = {}  # uuid -> {"img_src": Path, "annots": [(cls, cx, cy, w, h), ...]}
    missing_imgs = 0
    skipped_labels = 0

    for json_file, img_dir in sources:
        print(f"  Loading {json_file.name} ...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for uuid, entry in data.items():
            bboxes = entry.get("bboxes", []) or []
            labels = entry.get("labels", []) or []
            if len(bboxes) != len(labels):
                continue

            annots = []
            for bb, lab in zip(bboxes, labels):
                if lab not in CLASS_MAP:
                    skipped_labels += 1
                    continue
                cx, cy, w, h = coco_to_yolo(bb)
                if w <= 0 or h <= 0:
                    continue
                annots.append((CLASS_MAP[lab], cx, cy, w, h))

            if not annots:
                continue

            img_path = img_dir / f"{uuid}.jpg"
            if not img_path.exists():
                missing_imgs += 1
                continue

            if uuid in combined:
                combined[uuid]["annots"].extend(annots)
            else:
                combined[uuid] = {"img_src": img_path, "annots": annots}

    # Write images + labels
    stats = {"symlink": 0, "copy": 0, "exists": 0}
    for uuid, info in combined.items():
        result = link_or_copy(info["img_src"], images_out / f"{uuid}.jpg", use_symlink)
        stats[result] += 1

        with open(labels_out / f"{uuid}.txt", "w", encoding="utf-8") as f:
            for cls, cx, cy, w, h in info["annots"]:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"  {split_name}: {len(combined)} images "
          f"(symlink={stats['symlink']}, copy={stats['copy']}, exists={stats['exists']})")
    if missing_imgs:
        print(f"    note: {missing_imgs} JSON entries had no matching .jpg on disk")
    if skipped_labels:
        print(f"    note: {skipped_labels} bboxes with non-{{palm,two}} labels skipped")

    return len(combined)


def write_yaml(out_root: Path, yaml_path: Path, splits_present):
    lines = [
        "# HaGRID 2-class (palm, two) YOLO dataset",
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits_present:
        lines.append("test: images/test")
    lines += ["", "names:", "  0: palm", "  1: two", ""]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote dataset config: {yaml_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/wider_face_add_lm_10_10/dataset",
                    help="Root of HaGRID-style dataset")
    ap.add_argument("--dst", default="data/hagrid_yolo",
                    help="Output YOLO-format dataset root")
    ap.add_argument("--no-symlink", action="store_true",
                    help="Always copy images (use if symlinks fail on Windows)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    use_symlink = not args.no_symlink

    if not src.is_dir():
        print(f"ERROR: source directory {src} not found", file=sys.stderr)
        sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [
            (src / "palm_train.json", src / "train" / "palm"),
            (src / "two_train.json",  src / "train" / "two"),
        ],
        "val": [
            (src / "palm_val.json", src / "val" / "palm"),
            (src / "two_val.json",  src / "val" / "two"),
        ],
        "test": [
            (src / "palm_test.json", src / "test" / "palm"),
            (src / "two_test.json",  src / "test" / "two"),
        ],
    }

    present = []
    for name, sources in splits.items():
        if not all(j.exists() for j, _ in sources):
            missing = [str(j) for j, _ in sources if not j.exists()]
            print(f"Skipping {name}: missing {missing}")
            continue
        print(f"Processing {name} ...")
        if process_split(sources, name, dst, use_symlink) > 0:
            present.append(name)

    write_yaml(dst, dst / "data.yaml", present)
    print(f"\nDone. Dataset at: {dst.resolve()}")
    print(f"Next: yolo detect train data={(dst / 'data.yaml').as_posix()} model=yolo11n.pt ...")


if __name__ == "__main__":
    main()
