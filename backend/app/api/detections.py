from fastapi import APIRouter, UploadFile, File
import cv2, numpy as np, tempfile, os, base64
from ultralytics import YOLO

router = APIRouter()
model = YOLO("yolov8x.pt")

def count_people(img, is_dense=False):
    # Dense ke liye conf kam, imgsz bada
    conf = 0.08 if is_dense else 0.25
    sz = 2048 if is_dense else 1280
    res = model.predict(img, conf=conf, iou=0.35, imgsz=sz, verbose=False)[0]
    out = img.copy()
    cnt = 0
    for b in res.boxes:
        if int(b.cls[0]) == 0:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cnt+=1
    return cnt, out

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Pehle normal check
    cnt1, _ = count_people(img, is_dense=False)

    # Agar bheed zyada lag rahi hai (15 se zyada) toh dense mode ON
    if cnt1 > 15:
        final_cnt, final_img = count_people(img, is_dense=True)
    else:
        final_cnt, final_img = cnt1, _
        # dobara nikalna padega image ke liye
        final_cnt, final_img = count_people(img, is_dense=False)

    _, buf = cv2.imencode('.jpg', final_img)
    b64 = base64.b64encode(buf).decode()

    return {
        "people_count": final_cnt,
        "count": final_cnt,
        "status": f"Total {final_cnt} - High Density" if final_cnt>30 else f"Total {final_cnt} - Clean Detection",
        "border_image": b64
    }

@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read()); p=tmp.name
    cap=cv2.VideoCapture(p); max_c=0; f=0
    while True:
        ret, fr = cap.read()
        if not ret: break
        if f%15==0:
            r=model.predict(fr, conf=0.25, iou=0.35, imgsz=640, verbose=False)[0]
            c=sum(1 for b in r.boxes if int(b.cls[0])==0)
            max_c=max(max_c,c)
        f+=1
    cap.release(); os.remove(p)
    return {"people_count": max_c, "count": max_c}