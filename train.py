from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(data="config.yaml", imgsz=640, epochs=10, batch=8)
