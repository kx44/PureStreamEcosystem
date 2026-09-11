from pathlib import Path
from ultralytics import RTDETR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
BASE_WEIGHTS = PROJECT_ROOT / "weights" / "rtdetr-l.pt"


def run_training():
    # Always start from the pretrained backbone, never from the diverged
    # rtdetr_purestream checkpoint -- its weights have been NaN since epoch 4.
    print(f"Initializing RT-DETR from: {BASE_WEIGHTS}")
    model = RTDETR(str(BASE_WEIGHTS))

    print(f"Starting training with config: {DATA_CONFIG}")
    model.train(
        data=str(DATA_CONFIG),
        epochs=100,
        imgsz=640,

        # --- stability: these three are what killed the previous run ---
        lr0=0.0001,          # RT-DETR paper value. 0.01 (the YOLO/SGD default)
                             # is ~100x too high for AdamW on a transformer.
        amp=False,           # fp16 + AdamW is a known NaN source on RT-DETR.
                             # Costs ~20-30% speed, buys numerical stability.
        batch=8,             # explicit. batch=-1 auto-picked 3, which is far
                             # too noisy for a transformer (and used only 3 of
                             # 12 GB VRAM anyway).

        warmup_epochs=5.0,   # gentler ramp than the default 3.0
        warmup_bias_lr=0.0,  # default 0.1 is wildly high for AdamW bias groups
        weight_decay=0.0001,  # DETR-family standard (default 0.0005 is YOLO's)

        optimizer="AdamW",
        cos_lr=True,
        deterministic=False,  # required for RT-DETR grid_sample CUDA compat
        patience=20,
        device=0,
        project=str(PROJECT_ROOT / "runs"),
        name="rtdetr_purestream_v2",  # fresh run; do not touch the dead one
    )


if __name__ == "__main__":
    run_training()
