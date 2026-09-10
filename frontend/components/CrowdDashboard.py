import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="Jansuraksha AI - Precision Analytics Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ Jansuraksha AI - Precision Analytics Engine")

@st.cache_resource
def load_detection_model():
    return YOLO("yolov8m.pt")

try:
    model = load_detection_model()
except Exception:
    model = YOLO("yolov8n.pt")

if 'bg_subtractor' not in st.session_state:
    st.session_state.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

st.sidebar.header("🕹️ Source Selection")
source_mode = st.sidebar.radio("Input Type:", ["Image Upload", "Live CCTV / Webcam", "Video Upload"])

st.sidebar.header("⚙️ Precision & Motion Tuning")
base_conf = st.sidebar.slider("Detection Sensitivity (Confidence)", 0.05, 0.80, 0.15)
iou_thresh = st.sidebar.slider("Overlap Filtering (IoU Threshold)", 0.10, 0.70, 0.45)
is_dense_crowd = st.sidebar.checkbox("🔥 High-Density Patch Mode (For Dense Crowds)", value=True)
filter_motion_only = st.sidebar.checkbox("⚡ Motion-Only Filter (Ignore Stationary/Static)", value=False)
motion_thresh = st.sidebar.slider("Motion Sensitivity Threshold", 5, 50, 15)

st.sidebar.header("🚨 Detection Targets")
detect_people = st.sidebar.checkbox("Detect People", value=True)
detect_vehicles = st.sidebar.checkbox("Detect Vehicles (Cars/Bikes/Buses)", value=True)

CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def check_motion(fg_mask, box):
    x1, y1, x2, y2 = box
    roi_mask = fg_mask[y1:y2, x1:x2]
    if roi_mask.size == 0:
        return True
    motion_ratio = (np.count_nonzero(roi_mask) / float(roi_mask.size)) * 100
    return motion_ratio >= motion_thresh

def process_frame(frame, fg_mask=None):
    h, w, _ = frame.shape
    crowd_count = 0
    vehicle_breakdown = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
    annotated_frame = frame.copy()
    boxes_to_draw = []

    if is_dense_crowd and detect_people:
        grid_rows, grid_cols = 3, 3
        cell_h, cell_w = h // grid_rows, w // grid_cols
        detected_centers = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                sy1, sy2 = r * cell_h, (r + 1) * cell_h if r < grid_rows - 1 else h
                sx1, sx2 = c * cell_w, (c + 1) * cell_w if c < grid_cols - 1 else w
                
                crop = frame[sy1:sy2, sx1:sx2]
                results = model(crop, conf=base_conf, iou=iou_thresh, imgsz=1280, verbose=False)

                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        full_x1, full_y1 = x1 + sx1, y1 + sy1
                        full_x2, full_y2 = x2 + sx1, y2 + sy1
                        cx, cy = (full_x1 + full_x2) // 2, (full_y1 + full_y2) // 2

                        if filter_motion_only and fg_mask is not None:
                            if not check_motion(fg_mask, (full_x1, full_y1, full_x2, full_y2)):
                                continue

                        is_duplicate = False
                        for (dcx, dcy) in detected_centers:
                            if abs(cx - dcx) < 20 and abs(cy - dcy) < 20:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            if cls_id == 0 and detect_people:
                                crowd_count += 1
                                detected_centers.append((cx, cy))
                                boxes_to_draw.append((full_x1, full_y1, full_x2, full_y2, (0, 255, 0), "Person"))
                            elif cls_id in CLASS_NAMES and cls_id != 0 and detect_vehicles:
                                v_type = CLASS_NAMES[cls_id]
                                vehicle_breakdown[v_type] += 1
                                detected_centers.append((cx, cy))
                                boxes_to_draw.append((full_x1, full_y1, full_x2, full_y2, (255, 165, 0), v_type))
    else:
        results = model(frame, conf=base_conf, iou=iou_thresh, imgsz=1280, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if filter_motion_only and fg_mask is not None:
                    if not check_motion(fg_mask, (x1, y1, x2, y2)):
                        continue

                if cls_id == 0 and detect_people:
                    crowd_count += 1
                    boxes_to_draw.append((x1, y1, x2, y2, (0, 255, 0), "Person"))
                elif cls_id in CLASS_NAMES and cls_id != 0 and detect_vehicles:
                    v_type = CLASS_NAMES[cls_id]
                    vehicle_breakdown[v_type] += 1
                    boxes_to_draw.append((x1, y1, x2, y2, (255, 165, 0), v_type))

    for (bx1, by1, bx2, by2, color, label) in boxes_to_draw:
        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(annotated_frame, label, (bx1, max(by1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    total_vehicles = sum(vehicle_breakdown.values())
    return annotated_frame, crowd_count, total_vehicles, vehicle_breakdown

col_display, col_metrics = st.columns([3, 1])

with col_metrics:
    st.subheader("📊 Live Telemetry")
    metric_crowd = st.empty()
    metric_vehicle = st.empty()
    st.markdown("---")
    st.subheader("🚗 Vehicle Categorization")
    metric_breakdown = st.empty()

with col_display:
    if source_mode == "Image Upload":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, motion_mask = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
            
            processed_img, p_cnt, v_cnt, v_breakdown = process_frame(frame, fg_mask=motion_mask)
            
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            metric_crowd.metric("👥 Total Crowd Count", p_cnt)
            metric_vehicle.metric("🚘 Total Vehicles Detected", v_cnt)
            metric_breakdown.json(v_breakdown)

    elif source_mode == "Video Upload":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                fg_mask = st.session_state.bg_subtractor.apply(frame)
                processed_img, p_cnt, v_cnt, v_breakdown = process_frame(frame, fg_mask=fg_mask)
                
                st_frame.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                metric_crowd.metric("👥 Total Crowd Count", p_cnt)
                metric_vehicle.metric("🚘 Total Vehicles Detected", v_cnt)
                metric_breakdown.json(v_breakdown)
            cap.release()

    elif source_mode == "Live CCTV / Webcam":
        rtsp_input = st.text_input("RTSP / IP Camera Stream Link (Leave blank for default webcam '0'):")
        cam_source = rtsp_input.strip() if rtsp_input.strip() != "" else 0

        if st.button("🔴 Start Live Feed"):
            cap = cv2.VideoCapture(cam_source)
            st_frame = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera Feed Offline.")
                    break
                
                fg_mask = st.session_state.bg_subtractor.apply(frame)
                processed_img, p_cnt, v_cnt, v_breakdown = process_frame(frame, fg_mask=fg_mask)
                
                st_frame.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                metric_crowd.metric("👥 Total Crowd Count", p_cnt)
                metric_vehicle.metric("🚘 Total Vehicles Detected", v_cnt)
                metric_breakdown.json(v_breakdown)
            cap.release()