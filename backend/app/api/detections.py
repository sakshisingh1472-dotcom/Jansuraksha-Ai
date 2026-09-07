from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
import os

router = APIRouter()
MODEL_PATH = "../yolov8l.pt" if os.path.exists("../yolov8l.pt") else "yolov8l.pt"
model = YOLO(MODEL_PATH)

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    # Sahi 4 tukde
    tiles = [
        img[0:H//2, 0:W//2],
        img[0:H//2, W//2:W],
        img[H//2:H, 0:W//2],
        img[H//2:H, W//2:W]
    ]

    total = 0
    for crop in tiles:
        results = model(crop, classes=[0], conf=0.15, imgsz=1280, verbose=False)
        total += len(results[0].boxes)

    return {
        "people_count": total,
        "is_crowded": total > 40,
        "status": "Crowd Alert! 🚨" if total > 40 else "Normal"
    }

@router.post("/demo")
async def demo(data: dict):
    return {"received": data}