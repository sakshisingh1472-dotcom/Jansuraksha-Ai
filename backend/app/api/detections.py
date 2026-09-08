from ultralytics import YOLO

model = YOLO("yolov8l.pt")

def get_count(image_path: str):
    results = model.predict(
        source=image_path,
        conf=0.08, # 0.15 pe 33 aa raha hai, 0.08 pe 70+ ayega
        imgsz=1600, # bada size pe chote log bhi dikhenge
        classes=[0],
        iou=0.65,
        verbose=False
    )
    count = len(results[0].boxes)
    print(f"COUNT: {count}")
    return count

def analyze_image(image_path: str):
    return get_count(image_path)