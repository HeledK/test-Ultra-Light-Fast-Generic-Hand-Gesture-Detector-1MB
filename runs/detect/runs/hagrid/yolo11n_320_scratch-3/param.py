from ultralytics import YOLO

model = YOLO(r"weights/best.pt")
print(model.info())