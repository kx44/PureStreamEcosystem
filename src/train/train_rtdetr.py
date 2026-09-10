from pathlib import Path
from ultralytics import RTDETR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
BASE_WEIGHTS = PROJECT_ROOT / "weights" / "rtdetr-l.pt"

def run_training():
    print(f"Initializing RT-DETR from: {BASE_WEIGHTS}")
    model = RTDETR(str(BASE_WEIGHTS))

    print(f"Starting training with config: {DATA_CONFIG}")
    model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,
        batch=-1,           # Auto-batch size for your RTX 5070 Ti VRAM
        optimizer="AdamW",  # Mandatory for Transformer stability
        cos_lr=True,        # Cosine learning rate scheduling
        deterministic=False,# Required for RT-DETR grid_sample CUDA compatibility
        patience=20,
        device=0,
        project=str(PROJECT_ROOT / "runs"),
        name="rtdetr_purestream"
    )

if __name__ == "__main__":
    run_training()