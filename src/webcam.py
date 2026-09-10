import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Clear terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Dynamically resolve relative paths from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "weights" / "yolov10_purestream_best.pt"

CONFIDENCE_THRESHOLD = 0.50  
CAMERA_SOURCE = 1            

if __name__ == '__main__':
    if not MODEL_PATH.exists():
        print(f"Error: Model weights not found at '{MODEL_PATH}'")
        sys.exit(1)

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))
    
    print("\n--- Loaded Class Labels ---")
    print(model.names)
    print("---------------------------\n")

    print(f"Starting stream on source {CAMERA_SOURCE} (Confidence: {CONFIDENCE_THRESHOLD * 100:.0f}%)...")
    
    results = model.predict(
        source=CAMERA_SOURCE,
        show=True,
        stream=True,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    for r in results:
        # Silencing Pylance type warnings for dynamic Ultralytics result objects
        boxes = r.boxes  # type: ignore
        if boxes is None or len(boxes) == 0:
            print("[STATUS] Null / Background (No objects detected)")
        else:
            for box in boxes:
                exact_confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]

                print(f"[DETECTED] Class: {class_name:<20} | Confidence: {exact_confidence * 100:.2f}%")