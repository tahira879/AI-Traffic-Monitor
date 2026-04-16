from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt') 

    # ⚠️ IMPORTANT: CHANGE "Vehicle-Detection-3" TO YOUR FOLDER NAME
    # The path must point to the YAML file inside the downloaded folder
    dataset_path = 'Vehicle-Detection-3/data.yaml' 
    
    results = model.train(
        data=dataset_path,     # Use the variable here
        epochs=300,            # Assignment requirement
        imgsz=640,
        batch=16,
        name='traffic_monitor'
    )

    print("✅ Training Completed! Check 'runs/detect/traffic_monitor/weights/best.pt'")

if __name__ == "__main__":
    train_model()