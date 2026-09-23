import json
import cv2
import numpy as np
import pathlib
from vision.ssd.config.fd_config import define_img_size
define_img_size(320)
from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing1 import TrainAugmentation

train_transform = TrainAugmentation(fd_config.image_size, fd_config.image_mean, fd_config.image_std)

root = pathlib.Path('data/wider_face_add_lm_10_10/dataset')
CLASS_NAMES = ('BACKGROUND', 'palm', 'two')
class_dict = {name: i for i, name in enumerate(CLASS_NAMES)}

# Pick a few samples from each JSON
samples_to_show = []
for json_name, folder in [('palm_train.json', 'palm'), ('two_train.json', 'two')]:
    with open(root / json_name) as f:
        data = json.load(f)
    count = 0
    for uuid, entry in data.items():
        if count >= 3:
            break
        img_path = root / 'train' / folder / f'{uuid}.jpg'
        if not img_path.exists():
            continue
        samples_to_show.append((img_path, entry))
        count += 1

for img_path, entry in samples_to_show:
    # Load image
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    
    # Parse boxes (same logic as fixed _parse_entry)
    boxes = []
    labels = []
    for bbox, label_str in zip(entry['bboxes'], entry['labels']):
        label_str = label_str.lower().strip()
        if label_str not in class_dict:
            continue
        tlx, tly, bw, bh = bbox
        x1 = tlx * img_w
        y1 = tly * img_h
        x2 = (tlx + bw) * img_w
        y2 = (tly + bh) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(class_dict[label_str])
    
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.array(labels, dtype=np.int64)
    
    # Run through the full training transform
    image_t, boxes_t, labels_t = train_transform(img_rgb, boxes, labels)
    
    # Denormalize for visualization
    img_np = image_t.permute(1, 2, 0).numpy()
    img_np = (img_np * 128 + 127).clip(0, 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    h, w = img_np.shape[:2]
    for box, label in zip(boxes_t, labels_t):
        x1, y1, x2, y2 = box
        if x2 <= 1.0:  # percent coords
            x1, y1, x2, y2 = x1*w, y1*h, x2*w, y2*h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, CLASS_NAMES[label], (x1, max(0, y1-5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow(f'{img_path.stem}', img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()