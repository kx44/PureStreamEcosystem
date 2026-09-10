from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
BASE_WEIGHTS = PROJECT_ROOT / "weights" / "yolov10n.pt"

def main():
    print(f"Initializing YOLOv10 from: {BASE_WEIGHTS}")
    # Passing the string path; Ultralytics will auto-download if missing
    model = YOLO(str(BASE_WEIGHTS))

    print(f"Starting training using config: {DATA_CONFIG}")
    model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        patience=20,
        batch=-1,
        imgsz=640,
        device=0,
        cos_lr=True,
        project=str(PROJECT_ROOT / "runs"),
        name="yolov10n_purestream"
    )

if __name__ == '__main__':
    main()