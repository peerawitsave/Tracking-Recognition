from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.predict(source="video01.mp4", show=True, 
              save=True, line_width=2, 
              show_labels=True, show_conf=True,
              classes=[0, 1], conf=0.60)