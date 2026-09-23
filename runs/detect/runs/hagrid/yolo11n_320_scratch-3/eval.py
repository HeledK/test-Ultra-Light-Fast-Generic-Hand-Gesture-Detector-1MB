from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"weights/best.pt")
    metrics = model.val(
        data=r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\data\hagrid_yolo\data.yaml",
        split="test",
        imgsz=320,
    )