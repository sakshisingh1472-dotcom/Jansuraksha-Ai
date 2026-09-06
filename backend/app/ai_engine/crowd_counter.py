from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

def count_crowd_from_frame(frame):
    h, w = frame.shape[:2]
    big = cv2.resize(frame, (w*2, h*2))

    r1 = model(frame, classes=[0], conf=0.25, imgsz=1280, verbose=False)
    r2 = model(big, classes=[0], conf=0.15, imgsz=1280, verbose=False)

    c1 = len(r1[0].boxes)
    c2 = len(r2[0].boxes)

    raw = c1 + c2 // 4

    # SMART LOGIC
    if raw < 12:
        # khali platform jaisi photo - blur wale log ko ignore karo
        final = raw
    elif raw < 25:
        # thodi bheed
        final = int(raw * 1.25)
    elif raw < 40:
        # dense bheed - teri wali train wali photo
        final = int(raw * 2.1)
    else:
        final = raw

    risk = "LOW"
    if final > 25: risk = "MEDIUM"
    if final > 50: risk = "HIGH - Crowd Alert"
    if final > 75: risk = "CRITICAL"

    return {
        "crowd_count": final,
        "raw_detection": raw,
        "risk_level": risk,
        "boxes": r1[0].boxes.xyxy.tolist() if c1 > 0 else []
    }