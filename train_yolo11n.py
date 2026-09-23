from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.yaml")
    model.train(
        data="data/hagrid_yolo/data.yaml",
        epochs=40,
        imgsz=320,
        batch=16,
        workers=6,
        optimizer="SGD",
        lr0=0.02,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        cos_lr=True,
        amp=False,
        cache=False,
        project="runs/hagrid",
        name="yolo11n_320_scratch",
    )