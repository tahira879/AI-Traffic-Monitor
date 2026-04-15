from roboflow import Roboflow

def download_dataset():
    try:
        # REPLACE WITH YOUR ACTUAL API KEY
        rf = Roboflow(api_key="YOUR_API_KEY_HERE") 
        
        # This is a public vehicle dataset example. 
        # If you have your own project, change workspace/project names.
        project = rf.workspace("moceans-lab-0wuek").project("vehicle-detection-n6nuk")
        version = project.version(3)
        dataset = version.download("yolov8")
        
        print(f"✅ Dataset downloaded to: {dataset.location}")
        print(f"👉 Please copy the folder name: {dataset.name}")
        return dataset.location
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    download_dataset()