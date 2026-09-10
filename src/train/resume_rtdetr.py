from pathlib import Path
from ultralytics import RTDETR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "runs" / "rtdetr_purestream" / "weights" / "last.pt"

def resume():
    model = RTDETR(str(CHECKPOINT))
    model.train(resume=True)

if __name__ == "__main__":
    resume()