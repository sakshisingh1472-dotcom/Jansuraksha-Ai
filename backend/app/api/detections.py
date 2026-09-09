from fastapi import APIRouter, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from collections import Counter

router = APIRouter()

print("Loading models...")
crowd_model = YOLO("yolov8l.pt")
vehicle_model = YOLO("yolov8m.pt")
print("Models loaded!")

PERSON = 0
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def count_persons_sahi(model, frame):
    h, w = frame.shape[:2]
    all_boxes = []
    all_scores = []

    # 1. Full image - ekdum low conf
    r = model(frame, conf=0.08, iou=0.7, imgsz=1920, verbose=False)[0]
    for box in r.boxes:
        if int(box.cls[0]) == PERSON:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            all_boxes.append([x1,y1,x2,y2])
            all_scores.append(float(box.conf[0]))

    # 2. Tiling - 640 size se poori image ko scan karo
    tile_size = 640
    stride = 480 # 160px overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            if x2 - x < 200 or y2 - y < 200: continue
            tile = frame[y:y2, x:x2]
            rt = model(tile, conf=0.08, iou=0.7, imgsz=640, verbose=False)[0]
            for box in rt.boxes:
                if int(box.cls[0]) == PERSON:
                    bx1,by1,bx2,by2 = box.xyxy[0].tolist()
                    all_boxes.append([bx1+x, by1+y, bx2+x, by2+y])
                    all_scores.append(float(box.conf[0]))

    if not all_boxes:
        return 0

    # Light NMS - duplicate hatana, count kam nahi karna
    boxes_np = np.array(all_boxes)
    # x,y,w,h me convert
    bboxes = [[int(x[0]), int(x[1]), int(x[2]-x[0]), int(x[3]-x[1])] for x in boxes_np]
    indices = cv2.dnn.NMSBoxes(bboxes, all_scores, 0.08, 0.5)

    if len(indices) == 0:
        return len(all_boxes) # NMS fail hua toh saare gin lo

    # indices is [[i], [j]] format
    final_count = len(indices)
    return final_count

def analyze_image_only(frame):
    crowd = count_persons_sahi(crowd_model, frame)
    # Vehicle bhi isi frame se
    v_results = vehicle_model(frame, conf=0.35, iou=0.45, verbose=False)[0]
    vehs = []
    for b in v_results.boxes:
        cls = int(b.cls[0])
        if cls in VEHICLES:
            vehs.append(VEHICLES[cls])
    return {
        "crowd_count": crowd,
        "vehicle_count": len(vehs),
        "vehicle_breakdown": dict(Counter(vehs)),
        "total_objects": crowd + len(vehs),
        "status": "HIGH" if crowd > 60 else "MEDIUM" if crowd > 25 else "NORMAL"
    }

def analyze_video_fast(frame):
    r = vehicle_model(frame, conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
    c = 0; v_list = []
    for b in r.boxes:
        cls = int(b.cls[0])
        if cls == PERSON: c+=1
        elif cls in VEHICLES: v_list.append(VEHICLES[cls])
    return {"crowd_count": c, "vehicle_count": len(v_list), "vehicle_breakdown": dict(Counter(v_list))}

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return analyze_image_only(frame)

@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            path = tmp.name
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_c = 0; max_v = 0; last = None; idx = 0
        step = 30 if total_frames > 300 else 15
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % step == 0:
                res = analyze_video_fast(frame)
                max_c = max(max_c, res["crowd_count"])
                max_v = max(max_v, res["vehicle_count"])
                last = res
            idx += 1
        cap.release()
        return {"max_crowd_count": max_c, "max_vehicle_count": max_v, "last_frame_analysis": last}
    except Exception as e:
        print("VIDEO ERROR:", e)
        return {"max_crowd_count": 0, "max_vehicle_count": 0, "error": str(e), "last_frame_analysis": {}}