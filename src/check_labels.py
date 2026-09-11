"""Scan the YOLO-format label files for the defects that make DETR-style
training produce NaN losses.

Degenerate boxes (zero or near-zero width/height) are the classic cause: the
GIoU loss divides by box area, so a zero-area GT box yields inf -> NaN. The
RT-DETR denoising path (get_cdn_group -> xywh2xyxy) is where that usually
surfaces as a hard crash.

Usage:  python src/check_labels.py
"""

from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPLITS = ("train", "valid", "test")

NUM_CLASSES = 22
MIN_SIDE = 1e-3  # ~0.64 px at 640. Anything below this is effectively zero-area.


def bbox_from_polygon(coords):
    xs, ys = coords[0::2], coords[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def scan_split(split):
    label_dir = DATA_ROOT / split / "labels"
    if not label_dir.is_dir():
        print(f"  [skip] {label_dir} does not exist")
        return Counter(), []

    counts = Counter()
    examples = []

    def note(kind, path, detail):
        counts[kind] += 1
        if sum(1 for k, _, _ in examples if k == kind) < 3:
            examples.append((kind, path.name, detail))

    for path in sorted(label_dir.glob("*.txt")):
        counts["files"] += 1
        for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            counts["boxes"] += 1

            try:
                cls = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                note("malformed", path, f"line {lineno}: {line[:60]}")
                continue

            if not 0 <= cls < NUM_CLASSES:
                note("bad_class", path, f"line {lineno}: class {cls}")

            if len(coords) == 4:
                cx, cy, w, h = coords
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                counts["polygons"] += 1
                cx, cy, w, h = bbox_from_polygon(coords)
            else:
                note("malformed", path, f"line {lineno}: {len(coords)} coords")
                continue

            if w <= 0 or h <= 0:
                note("zero_area", path, f"line {lineno}: w={w:.6g} h={h:.6g}")
            elif w < MIN_SIDE or h < MIN_SIDE:
                note("near_zero", path, f"line {lineno}: w={w:.6g} h={h:.6g}")

            if any(v < 0 or v > 1 for v in (cx, cy, w, h)):
                note("out_of_bounds", path, f"line {lineno}: {cx:.4g} {cy:.4g} {w:.4g} {h:.4g}")

            if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 \
                    or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                note("spills_edge", path, f"line {lineno}: {cx:.4g} {cy:.4g} {w:.4g} {h:.4g}")

    return counts, examples


def main():
    print(f"Scanning {DATA_ROOT}\n")
    grand = Counter()

    for split in SPLITS:
        print(f"{split}:")
        counts, examples = scan_split(split)
        grand.update(counts)

        if counts:
            print(f"  files={counts['files']}  boxes={counts['boxes']}"
                  f"  polygons={counts['polygons']}")
            for kind in ("zero_area", "near_zero", "out_of_bounds",
                         "spills_edge", "bad_class", "malformed"):
                if counts[kind]:
                    print(f"  {kind:<14} {counts[kind]}")
            for kind, name, detail in examples:
                print(f"      e.g. [{kind}] {name} -> {detail}")
        print()

    fatal = grand["zero_area"] + grand["near_zero"]
    print("-" * 60)
    print(f"TOTAL boxes: {grand['boxes']}")
    print(f"zero/near-zero area (NaN risk): {fatal}")
    print(f"out of bounds: {grand['out_of_bounds']}")
    if fatal:
        print("\n-> Degenerate boxes present. These can produce NaN in the GIoU")
        print("   loss regardless of learning rate. Remove or fix them.")
    else:
        print("\n-> No degenerate boxes. NaN was almost certainly the learning rate.")


if __name__ == "__main__":
    main()
