"""Live webcam demo for the PureStream detector.

Hotkeys (the video window must have focus):
    r   Rig: detect the object in the current frame and assign it to a class.
        The frame and its YOLO label are saved under data/rigged/ so the object
        is learned on the next training run (see README, "Rigging new objects").
    q   Quit (Esc also works).
"""

import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Clear terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Dynamically resolve relative paths from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "weights" / "yolov10_purestream_best.pt"
RIGGED_DIR = PROJECT_ROOT / "data" / "rigged"

CONFIDENCE_THRESHOLD = 0.50
CAMERA_SOURCE = 1

RIG_KEY = ord('r')
QUIT_KEYS = (ord('q'), 27)
# A new object is by definition one the model is unsure about, so rigging
# searches far below the display threshold before falling back to a hand-drawn box.
RIG_SEARCH_CONFIDENCE = 0.05
WINDOW_NAME = "PureStream"


def propose_box(model, frame):
    """Return (x1, y1, x2, y2, class_id or None) for the most likely object in frame."""
    result = model.predict(frame, conf=RIG_SEARCH_CONFIDENCE, verbose=False)[0]
    boxes = result.boxes  # type: ignore
    if boxes is not None and len(boxes) > 0:
        best = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[best].tolist())
        return x1, y1, x2, y2, int(boxes.cls[best].item())
    return draw_box(frame)


def draw_box(frame):
    """Let the user drag a box around the object. Returns None if cancelled."""
    print("[RIG] Drag a box around the object, then press Enter (c to cancel).")
    x, y, w, h = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=True)
    if w == 0 or h == 0:
        return None
    return float(x), float(y), float(x + w), float(y + h), None


def ask_class(names, guess):
    """Prompt in the terminal for a class. Returns a class id, 'redraw', or None to skip."""
    print("\n--- Rig to which class? ---")
    for class_id, name in names.items():
        print(f"  {class_id:>2}  {name}")
    default = f" [Enter = {names[guess]}]" if guess is not None else ""
    while True:
        answer = input(f"Class number or name{default}, d = redraw box, s = skip: ").strip()
        if answer == "" and guess is not None:
            return guess
        if answer.lower() == "s":
            return None
        if answer.lower() == "d":
            return "redraw"
        if answer.isdigit() and int(answer) in names:
            return int(answer)
        matches = [cid for cid, name in names.items() if name == answer]
        if matches:
            return matches[0]
        print(f"  '{answer}' is not a class. Use a number from the list.")


def save_rigged(frame, box, class_id):
    """Write frame + YOLO-format label to data/rigged/. Returns the image path."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (max(0.0, min(v, limit)) for v, limit in
                      zip(box, (width, height, width, height)))
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height

    stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    image_path = RIGGED_DIR / "images" / f"{stem}.jpg"
    label_path = RIGGED_DIR / "labels" / f"{stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), frame)
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return image_path


def rig(model, frame):
    """Detect the object in frame, ask which class it belongs to, and save it."""
    proposal = propose_box(model, frame)
    while True:
        if proposal is None:
            print("[RIG] Cancelled.")
            return
        *box, guess = proposal

        preview = frame.copy()
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(preview, "RIGGING - answer in terminal", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(1)

        choice = ask_class(model.names, guess)
        if choice == "redraw":
            proposal = draw_box(frame)
            continue
        if choice is None:
            print("[RIG] Skipped.")
            return
        image_path = save_rigged(frame, box, choice)
        print(f"[RIG] Rigged as '{model.names[choice]}' -> {image_path.relative_to(PROJECT_ROOT)}\n")
        return


if __name__ == '__main__':
    if not MODEL_PATH.exists():
        print(f"Error: Model weights not found at '{MODEL_PATH}'")
        sys.exit(1)

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("\n--- Loaded Class Labels ---")
    print(model.names)
    print("---------------------------\n")

    capture = cv2.VideoCapture(CAMERA_SOURCE)
    if not capture.isOpened():
        print(f"Error: Could not open camera source {CAMERA_SOURCE}")
        sys.exit(1)

    print(f"Starting stream on source {CAMERA_SOURCE} (Confidence: {CONFIDENCE_THRESHOLD * 100:.0f}%)...")
    print("Hotkeys: [r] rig the object in view to a class   [q] quit\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: Lost camera feed.")
                break

            r = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            cv2.imshow(WINDOW_NAME, r.plot())

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

            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
            if key == RIG_KEY:
                rig(model, frame)
    finally:
        capture.release()
        cv2.destroyAllWindows()
