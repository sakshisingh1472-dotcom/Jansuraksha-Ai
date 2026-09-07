from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import tempfile
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Tera wahi best model - yolov8l.pt
model = YOLO("yolov8l.pt")

def get_exact_count_with_tiling(image):
    # Tiling logic - door ke chote log bhi count honge, border CCTV ke liye best
    h, w = image.shape[:2]
    count = 0
    all_boxes = []

    # 640x640 ke tiles me kaat ke check karega - exact count
    for y in range(0, h, 640):
        for x in range(0, w, 640):
            tile = image[y:y+640, x:x+640]
            if tile.size == 0: continue
            results = model.predict(source=tile, imgsz=640, conf=0.25, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0]) == 0: # 0 = person
                    count += 1
                    # global coordinates
                    bx = box.xyxy[0].cpu().numpy()
                    all_boxes.append([bx[0]+x, bx[1]+y, bx[2]+x, bx[3]+y])

    # Border draw
    annotated = image.copy()
    for bx in all_boxes:
        cv2.rectangle(annotated, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0,0,255), 2)

    _, buffer = cv2.imencode('.jpg', annotated)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return count, f"data:image/jpeg;base64,{b64}"

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        count, b64_img = get_exact_count_with_tiling(img)
        status = "RED ALERT - BORDER BREACH" if count > 10 else "SAFE" if count < 5 else "WARNING"
        return {"people_count": count, "status": status, "border_image": b64_img}
    except Exception as e:
        return {"error": str(e), "people_count": 0, "status": "ERROR"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # video temp save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        max_count = 0
        frame_count = 0
        last_annotated = None

        while True:
            ret, frame = cap.read()
            if not ret: break
            # Har 15th frame ka exact count lega - fast + exact
            if frame_count % 15 == 0:
                count, b64_img = get_exact_count_with_tiling(frame)
                if count > max_count:
                    max_count = count
                    last_annotated = b64_img
            frame_count += 1

        cap.release()
        os.remove(tmp_path)

        status = "RED ALERT - BORDER BREACH" if max_count > 10 else "SAFE" if max_count < 5 else "WARNING"
        return {"people_count": max_count, "status": status, "border_image": last_annotated, "frames_checked": frame_count}
    except Exception as e:
        return {"error": str(e), "people_count": 0, "status": "ERROR"}