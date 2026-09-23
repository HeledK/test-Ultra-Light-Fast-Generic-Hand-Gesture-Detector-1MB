"""
hagrid_dataset.py
-----------------
Drop-in replacement for VOCDataset that reads HaGRID JSON annotations
instead of VOC XML files.

Expected folder structure (relative to train.py):
    data/
    └── wider_face_add_lm_10_10/
        └── dataset/
            ├── palm_train.json
            ├── palm_val.json
            ├── two_train.json
            ├── two_val.json
            ├── train/
            │   ├── palm/   ← training images for palm
            │   └── two/    ← training images for two
            └── val/
                ├── palm/
                └── two/

JSON structure per entry:
    "uuid": {
        "bboxes": [top_left_x, top_left_y, width, height],   # normalised [0,1]
        "labels": ["palm"],
        ...
    }

Bounding boxes are stored as [top_left_x, top_left_y, width, height] normalised and converted
to absolute pixel [x1, y1, x2, y2] at read time, matching VOCDataset output.
"""

import json
import logging
import pathlib
import cv2
import numpy as np

from PIL import Image
class HaGRIDDataset:
    """
    HaGRID JSON dataset. API-compatible with VOCDataset so it can be used
    as a direct replacement in train.py without changing any other code.
    """

    CLASS_NAMES = ('BACKGROUND', 'palm', 'two')

    # JSON filenames for each split
    JSON_FILES = {
        'train': ['palm_train.json', 'two_train.json'],
        'val':   ['palm_val.json',   'two_val.json'],
    }

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

    def __init__(self, root, transform=None, target_transform=None,
                 is_test=False, keep_difficult=False, label_file=None):
        """
        Args:
            root        : path to the dataset/ folder containing JSON files
                          and train/ val/ image subfolders
            is_test     : if True, use val split (test split is never used
                          during training as per project requirements)
            transform   : image augmentation (TrainAugmentation or TestTransform)
            target_transform : box/label transform (MatchPrior)
        """
        self.root             = pathlib.Path(root)
        self.transform        = transform
        self.target_transform = target_transform
        self.keep_difficult   = keep_difficult   # kept for API compatibility

        self.class_names = self.CLASS_NAMES
        self.class_dict  = {name: i for i, name in enumerate(self.class_names)}

        split = 'val' if is_test else 'train'
        self.samples = self._load_samples(split)

        logging.info(f"HaGRIDDataset [{split}]: {len(self.samples)} images — "
                     f"classes: {self.class_names}")

    # ── Internal loading ──────────────────────────────────────────────────────

    def _load_samples(self, split: str) -> list:
        """
        Parse JSON files for the given split and return a flat list of:
            {
                'image_path': Path,
                'boxes':      np.ndarray  shape (N, 4)  absolute x1y1x2y2
                'labels':     np.ndarray  shape (N,)    int64 class indices
            }
        Images without any valid annotations are skipped.
        """
        image_dir = self.root / split   # e.g.  dataset/train/

        # Build uuid → filepath index for this split
        uuid_map = self._build_uuid_map(image_dir)
        logging.info(f"  Indexed {len(uuid_map):,} images in {image_dir}")

        samples  = []
        skipped  = 0

        for json_name in self.JSON_FILES[split]:
            json_path = self.root / json_name
            if not json_path.exists():
                logging.warning(f"  JSON not found, skipping: {json_path}")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            for uuid, entry in data.items():
                img_path = uuid_map.get(uuid)
                if img_path is None:
                    skipped += 1
                    continue

                boxes, labels = self._parse_entry(entry, img_path)
                if len(boxes) == 0:
                    skipped += 1
                    continue

                samples.append({
                    'image_path': img_path,
                    'boxes':      boxes,
                    'labels':     labels,
                })

        if skipped:
            logging.info(f"  Skipped {skipped} entries "
                         f"(image not found or no valid annotations)")
        return samples

    def _build_uuid_map(self, image_dir: pathlib.Path) -> dict:
        """
        Scan image_dir recursively (covers palm/ and two/ subfolders) and
        build uuid → Path mapping.
        Plain filenames: uuid.jpg
        No Roboflow affixes used per project requirements.
        """
        uuid_map = {}
        for ext in self.IMG_EXTENSIONS:
            for img_path in image_dir.rglob(f'*{ext}'):
                uuid = img_path.stem   # filename without extension = uuid
                uuid_map[uuid] = img_path
        return uuid_map

    def _parse_entry(self, entry: dict, img_path: pathlib.Path):
        """
        Convert one JSON entry to absolute-pixel boxes and integer labels.
        HaGRID bbox format: [top_left_x, top_left_y, width, height] normalized [0, 1] (COCO format)
        Output format:      [x1, y1, x2, y2] absolute pixels
        """
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        raw_boxes  = entry.get('bboxes', [])
        raw_labels = entry.get('labels', [])

        boxes  = []
        labels = []

        for bbox, label_str in zip(raw_boxes, raw_labels):
            label_str = label_str.lower().strip()
            if label_str not in self.class_dict:
                continue

            # HaGRID format: [top_left_x, top_left_y, width, height] normalized
            tlx, tly, bw, bh = bbox

            x1 = tlx * img_w
            y1 = tly * img_h
            x2 = (tlx + bw) * img_w
            y2 = (tly + bh) * img_h

            # Clamp to image boundaries
            x1 = max(0.0, x1);  y1 = max(0.0, y1)
            x2 = min(float(img_w - 1), x2)
            y2 = min(float(img_h - 1), y2)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[label_str])

        boxes_arr  = np.array(boxes,  dtype=np.float32).reshape(-1, 4)
        labels_arr = np.array(labels, dtype=np.int64).reshape(-1)
        return boxes_arr, labels_arr
    # ── Public API (matches VOCDataset) ───────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        boxes  = sample['boxes'].copy()
        labels = sample['labels'].copy()
        image  = self._read_image(sample['image_path'])

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def get_image(self, index):
        img_path = self.samples[index]['image_path']
        image    = self._read_image(img_path)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        sample = self.samples[index]
        return (str(sample['image_path']),
                (sample['boxes'].copy(),
                 sample['labels'].copy(),
                 np.zeros(len(sample['labels']), dtype=np.uint8)))  # no difficult flag

    def _read_image(self, img_path: pathlib.Path) -> np.ndarray:
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"cv2.imread failed for {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
