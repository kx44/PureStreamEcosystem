"""Repair the label defects that check_labels.py reports.

Two operations, applied per annotation line:

  1. Clamp   -- coordinates outside [0, 1] are clipped back to the image.
                A box that is merely a few pixels over the edge is a valid
                annotation, so it is fixed rather than discarded.
  2. Drop    -- any box that is degenerate (zero or near-zero width/height),
                either originally or after clamping, is removed. These cannot
                be repaired: a zero-area box has no valid extent, and it makes
                the GIoU loss divide by zero, which is where NaN comes from.

Modified files are backed up alongside the original as <name>.txt.bak.
The labels.cache files are deleted afterwards -- ultralytics will happily
reuse a stale cache and silently ignore every repair otherwise.

Usage:
    python src/fix_labels.py --dry-run     # report only, change nothing
    python src/fix_labels.py               # apply
"""

import argparse
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPLITS = ("train", "valid", "test")

MIN_SIDE = 1e-3  # ~0.64 px at 640


def clamp01(v):
    return min(1.0, max(0.0, v))


def fix_box(cx, cy, w, h):
    """Clamp to image bounds via xyxy. Returns None if degenerate."""
    x0, x1 = clamp01(cx - w / 2), clamp01(cx + w / 2)
    y0, y1 = clamp01(cy - h / 2), clamp01(cy + h / 2)
    nw, nh = x1 - x0, y1 - y0
    if nw < MIN_SIDE or nh < MIN_SIDE:
        return None
    return (x0 + x1) / 2, (y0 + y1) / 2, nw, nh


def fix_polygon(coords):
    """Clamp each vertex. Returns None if the resulting bbox is degenerate."""
    out = [clamp01(v) for v in coords]
    xs, ys = out[0::2], out[1::2]
    if (max(xs) - min(xs)) < MIN_SIDE or (max(ys) - min(ys)) < MIN_SIDE:
        return None
    return out


def fmt(v):
    return f"{v:.6f}".rstrip("0").rstrip(".")


def process_file(path, stats, dry_run):
    original = path.read_text()
    kept, changed = [], False

    for raw in original.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()

        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            stats["dropped_malformed"] += 1
            changed = True
            continue

        if len(coords) == 4:
            fixed = fix_box(*coords)
            if fixed is None:
                stats["dropped_degenerate"] += 1
                changed = True
                continue
            if any(abs(a - b) > 1e-9 for a, b in zip(fixed, coords)):
                stats["clamped"] += 1
                changed = True
            kept.append(f"{cls} " + " ".join(fmt(v) for v in fixed))

        elif len(coords) >= 6 and len(coords) % 2 == 0:
            fixed = fix_polygon(coords)
            if fixed is None:
                stats["dropped_degenerate"] += 1
                changed = True
                continue
            if any(abs(a - b) > 1e-9 for a, b in zip(fixed, coords)):
                stats["clamped"] += 1
                changed = True
            kept.append(f"{cls} " + " ".join(fmt(v) for v in fixed))

        else:
            stats["dropped_malformed"] += 1
            changed = True

    if not changed:
        return

    stats["files_modified"] += 1
    if not kept:
        stats["files_emptied"] += 1

    if dry_run:
        return

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(original)
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    print(f"{'DRY RUN - ' if args.dry_run else ''}Repairing labels under {DATA_ROOT}\n")
    total = Counter()

    for split in SPLITS:
        label_dir = DATA_ROOT / split / "labels"
        if not label_dir.is_dir():
            print(f"{split}: [skip] {label_dir} not found")
            continue

        stats = Counter()
        for path in sorted(label_dir.glob("*.txt")):
            process_file(path, stats, args.dry_run)

        print(f"{split}:")
        print(f"  files modified      {stats['files_modified']}")
        print(f"  boxes clamped       {stats['clamped']}")
        print(f"  boxes dropped       {stats['dropped_degenerate']}")
        if stats["dropped_malformed"]:
            print(f"  malformed dropped   {stats['dropped_malformed']}")
        if stats["files_emptied"]:
            print(f"  files now empty     {stats['files_emptied']}  "
                  f"(treated as background images -- this is fine)")
        total.update(stats)
        print()

    print("-" * 60)
    print(f"files modified {total['files_modified']}   "
          f"clamped {total['clamped']}   dropped {total['dropped_degenerate']}")

    if args.dry_run:
        print("\nDry run - nothing written. Re-run without --dry-run to apply.")
        return

    removed = 0
    for split in SPLITS:
        cache = DATA_ROOT / split / "labels.cache"
        if cache.exists():
            cache.unlink()
            removed += 1
    print(f"\nDeleted {removed} labels.cache file(s) so ultralytics re-scans.")
    print("Originals backed up as <name>.txt.bak")
    print("Verify with: python src/check_labels.py")


if __name__ == "__main__":
    main()
