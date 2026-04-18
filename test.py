from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="Dataset/non_split/image/origin_strawberry_299.jpg",
    save=True
)