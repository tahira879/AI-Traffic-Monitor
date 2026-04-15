import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import os
from collections import defaultdict

# --- CONFIGURATION ---
st.set_page_config(page_title="Traffic AI Demo", layout="wide")

@st.cache_resource
def load_model():
    # 'yolov8n.pt' automatically downloads from Ultralytics servers (No custom training needed)
    # This detects: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
    model = YOLO('yolov8n.pt')
    st.success("✅ Model Loaded (Using Standard YOLOv8)")
    return model

model = load_model()

# Excel File setup
EXCEL_FILE = "traffic_violations.xlsx"
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["ID", "Type", "Plate", "Speed", "Violation", "Time"])
    df.to_excel(EXCEL_FILE, index=False)

# --- HELPER FUNCTIONS ---
def calculate_speed(prev_center, curr_center, fps, ppm=8): # ppm=8 is a guess value
    distance_pixels = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
    distance_meters = distance_pixels / ppm
    time_seconds = 1 / fps
    speed_mps = distance_meters / time_seconds
    speed_kmh = speed_mps * 3.6
    return int(speed_kmh)

def process_video(video_path, signal_red):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker_history = defaultdict(list)
    vehicle_crossed_line = {}
    stop_line_y = int(height * 0.6)  
    violations_data = []
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO Tracking
        results = model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                id = ids[i]
                cls_id = classes[i]
                class_name = model.names[cls_id]
                
                # Filter only vehicles (Ignore people etc for this demo)
                if class_name not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    continue

                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Speed Calc
                if id in tracker_history and len(tracker_history[id]) > 0:
                    prev_center = tracker_history[id][-1]
                    speed = calculate_speed(prev_center, center, fps)
                else:
                    speed = 0
                
                tracker_history[id].append(center)
                if len(tracker_history[id]) > 10:
                    tracker_history[id].pop(0)

                # Violation Logic
                is_violation = False
                plate_text = "UNKNOWN" # Removed EasyOCR to avoid errors
                
                if signal_red and (y1 < stop_line_y < y2):
                    if id not in vehicle_crossed_line:
                        vehicle_crossed_line[id] = True
                        is_violation = True
                        
                        new_row = {
                            "ID": id, "Type": class_name, "Plate": plate_text, 
                            "Speed": speed, "Violation": "Signal Break", "Time": time.strftime("%H:%M:%S")
                        }
                        violations_data.append(new_row)

                # Visuals
                color = (0, 255, 0) # Green
                label = f"{class_name} {speed}km/h"

                if is_violation:
                    color = (0, 0, 255) # Red
                    label = f"VIOLATION! {class_name}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "SIGNAL VIOLATION", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Line
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), (0, 255, 255), 2)
        cv2.putText(frame, "STOP LINE", (10, stop_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR")

    cap.release()
    out.release()
    
    if violations_data:
        df_new = pd.DataFrame(violations_data)
        df_existing = pd.read_excel(EXCEL_FILE)
        df_final = pd.concat([df_existing, df_new])
        df_final.to_excel(EXCEL_FILE, index=False)
    
    return "output_video.mp4"

# --- UI ---
st.title("🚦 AI Traffic Monitoring System (Demo Mode)")
st.sidebar.header("Settings")

signal_red = st.sidebar.checkbox("Signal Status (RED = Violation)", value=False)
video_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])

if video_file is not None:
    temp_video = "temp_input.mp4"
    with open(temp_video, "wb") as f:
        f.write(video_file.getbuffer())
    
    st.video(temp_video)
    
    if st.button("Start Detection"):
        st.info("Processing... This may take a moment.")
        output_file = process_video(temp_video, signal_red)
        
        st.success("Done!")
        st.video(output_file)
        
        with open(EXCEL_FILE, "rb") as f:
            st.download_button(
                label="Download Excel Report",
                data=f,
                file_name="violations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )