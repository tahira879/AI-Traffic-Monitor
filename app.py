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
    import gc # Import garbage collector to clear memory
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Reduce video resolution for processing to save RAM
    # New dimensions (Half the original size)
    new_width = 640
    new_height = 480
    
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    tracker_history = defaultdict(list)
    vehicle_crossed_line = {}
    # Adjust stop line according to new height
    stop_line_y = int(new_height * 0.6)  
    violations_data = []
    stframe = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame immediately to save RAM during processing
        frame = cv2.resize(frame, (new_width, new_height))

        # --- MAIN CHANGE HERE ---
        # imgsz=320 forces the model to process at very low resolution (saving huge memory)
        results = model.track(frame, persist=True, verbose=False, imgsz=320)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                id = ids[i]
                cls_id = classes[i]
                class_name = model.names[cls_id]
                
                if class_name not in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    continue

                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if id in tracker_history and len(tracker_history[id]) > 0:
                    prev_center = tracker_history[id][-1]
                    speed = calculate_speed(prev_center, center, fps)
                else:
                    speed = 0
                
                tracker_history[id].append(center)
                if len(tracker_history[id]) > 10:
                    tracker_history[id].pop(0)

                is_violation = False
                plate_text = "UNKNOWN"
                
                if signal_red and (y1 < stop_line_y < y2):
                    if id not in vehicle_crossed_line:
                        vehicle_crossed_line[id] = True
                        is_violation = True
                        
                        new_row = {
                            "ID": id, "Type": class_name, "Plate": plate_text, 
                            "Speed": speed, "Violation": "Signal Break", "Time": time.strftime("%H:%M:%S")
                        }
                        violations_data.append(new_row)

                color = (0, 255, 0)
                label = f"{class_name} {speed}km/h"

                if is_violation:
                    color = (0, 0, 255)
                    label = f"VIOLATION! {class_name}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "SIGNAL VIOLATION", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, (0, stop_line_y), (new_width, stop_line_y), (0, 255, 255), 2)
        cv2.putText(frame, "STOP LINE", (10, stop_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR")
        
        # --- MEMORY CLEANUP ---
        # Clear 'results' variable to free RAM
        del results
        gc.collect()
        
        frame_count += 1

    cap.release()
    out.release()
    
    if violations_data:
        df_new = pd.DataFrame(violations_data)
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_final = pd.concat([df_existing, df_new])
        else:
            df_final = df_new
        df_final.to_excel(EXCEL_FILE, index=False)
    
    return "output_video.mp4"
