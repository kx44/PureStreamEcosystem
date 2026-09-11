# PureStream Ecosystem

**Decentralized, edge-AI waste triage system that retrofits existing HDB refuse chutes**, turning them into high-purity recycling hubs. Built for the NTU Chua Thian Poh Pinnacle Prize (AI for Good and Smart Cities).

**Team:** Soh Kai Xuan · Teo Qing Yang · Angelina Kee · Zoe Seah Jia Ying · Htut Myat Min

## The problem

Singapore's Semakau landfill is projected to hit capacity by 2035, and the household recycling rate is stuck at ~11-12%. Four root causes: contamination from "wish-cycling," the inconvenience of walking to centralized blue bins vs. the chute, commingling of clean and dirty waste once collected, and resident apathy from anonymous, shared bins.

## The solution

Pure-Stream retrofits existing chutes with:
- **Privacy-first identification** — BLE sensor + mobile app, camera-free authentication as residents approach the chute.
- **Mechanical debagging** — spinning impeller blades split bags open for inspection.
- **Edge-AI machine vision sorting** — a locked-down, in-chute camera grades items in real time and diverts them into the correct purity stream (see `src/` below).
- **Rewards loop** — good sorting behavior is tracked per-household and rebated, replacing anonymity with accountability.

## Business case (from pitch deck)

- Market: smart waste management, ~14-22% CAGR, ~USD 48B Asia-Pacific opportunity.
- Target: lift household recycling rate from ~11% toward 40%, cut waste-collector operating costs ~26%.
- Unit economics: ~$7,134 capex/unit (actuators & motors, vision camera + edge AI, mechanical debagger), ~$467 opex/unit (hardware/AI maintenance, electricity, cloud DB).
- Ask: $100k to cross from working prototype to an HDB pilot.

## This repo: the edge-AI vision component

This codebase is the object-detection pipeline behind the "camera" step above.

- **Dataset**: custom-annotated, 22 classes covering grade A/B recyclables (cardboard, glass, paper, plastic bags/bottles/containers) and trash (food waste, wrappers, facemasks, metal, styrofoam, etc.) — see `configs/data.yaml`. Sourced via Roboflow (`soh-kai-xuan/pure-stream-class-merged`).
- **Models**: `src/train/train_yolov10.py` (YOLOv10n) and `src/train/train_rtdetr.py` (RT-DETR, AdamW + cosine LR) — both 100 epochs, auto-batch, tuned for an RTX 5070 Ti. `src/train/resume_rtdetr.py` resumes from the last RT-DETR checkpoint.
- **Live demo**: `src/webcam.py` runs the trained YOLOv10 weights (`weights/yolov10_purestream_best.pt`) against a webcam/camera feed at 50% confidence and prints detected classes in real time.

## Deployment target

One NVIDIA Jetson module per retrofitted chute, running the detector under TensorRT.

**Recommended: Jetson Orin Nano 8GB** (67 TOPS, 102 GB/s, 7–25 W, ~US$399 as of Jul 2026).

Rationale, and why not the larger parts:

| Module | TOPS | Bandwidth | Power | Price (Jul 2026) |
|---|---|---|---|---|
| Orin Nano 4GB | 34 | 51 GB/s | 7–25 W | $349 |
| **Orin Nano 8GB** | **67** | **102 GB/s** | **7–25 W** | **$399** |
| Orin NX 8GB | 117 | 102 GB/s | 10–40 W | $649 |
| Orin NX 16GB | 157 | 102 GB/s | 10–40 W | $999 |
| AGX Orin 32GB | 200 | 205 GB/s | 15–40 W | $1,799 |
| AGX Orin 64GB | 275 | 205 GB/s | 15–60 W | $2,999 |

- **Not the 4GB Nano.** Half the memory bandwidth (51 vs 102 GB/s), and RT-DETR's attention layers are bandwidth-bound, so the gap is worse than the TOPS figures suggest.
- **Not AGX Orin or Thor.** Power alone breaks the opex model: AGX Orin 64GB at 60 W draws ~526 kWh/yr ≈ S$158/yr at S$0.30/kWh — more than double the $77/unit/yr electricity line in the business case, before the actuators draw anything. Orin Nano 8GB at ~15 W is ~131 kWh/yr ≈ S$39/yr, which fits.
- **Orin NX 16GB** is the upgrade path if a unit ever needs multiple camera streams or a larger model. It is not needed for single-camera, single-shot inspection.

NVIDIA raised Jetson prices across the board in July 2026 (33–101% depending on SKU). The $1,400/unit "Vision Camera & Edge AI" line in the pitch deck predates that increase and should be re-checked against current quotes.

### Open dependency: capture mechanics

The required inference rate depends entirely on whether the item is moving when inspected. An item in free fall past a 0.5 m field of view is in frame for under 40 ms, which no Jetson can catch reliably — and motion blur would degrade it regardless. The workable designs are either (a) the staggered baffle system arrests the item so inspection happens at near-zero velocity, or (b) a break-beam sensor hardware-triggers a global-shutter camera with strobe, and inference runs once per triggered frame. Both make Orin Nano 8GB sufficient. **This decision should be settled before hardware is ordered.**

## Setup

The dataset and model weights are deliberately **not tracked in git** (see `.gitignore`) — the trained checkpoints alone are ~264 MB each, well past GitHub's 100 MB per-file limit. A fresh clone therefore needs three things restored:

1. **Dataset** — download `pure-stream-class-merged` from Roboflow (workspace `soh-kai-xuan`, CC BY 4.0) in YOLO format and extract it into `data/`, so the layout matches `configs/data.yaml`:

   ```
   data/
     train/images   train/labels
     valid/images   valid/labels
     test/images    test/labels
   ```

2. **Base weights** — `weights/rtdetr-l.pt` and `weights/yolov10n.pt`. Ultralytics auto-downloads these on first run if they're missing.

3. **Trained weights** — `runs/rtdetr_purestream/weights/best.pt` (~264 MB). Not in the repo; distribute via GitHub Releases (2 GB/file limit) or a shared drive.

### Dataset notes

The Roboflow export is in **YOLOv8 detection** format, which is also the correct format for RT-DETR — ultralytics uses the same dataset loader for both. Two caveats seen in training logs:

- `Box and segment counts should be equal (segments=10571, boxes=60722)` — the export mixes polygon and box annotations. Ultralytics discards the polygons and trains on boxes, so it is not fatal, but a detection-only re-export would remove the ambiguity.
- `ignoring corrupt image/label: non-normalized or out of bounds coordinates` — at least one label has a coordinate > 1.

Run `python src/check_labels.py` to scan all three splits for out-of-bounds, zero-area, and near-zero-area boxes, then `python src/fix_labels.py` to repair them (`--dry-run` to preview). The repair clamps salvageable out-of-bounds boxes back to the image, drops boxes that are degenerate even after clamping, backs up each modified file as `<name>.txt.bak`, and deletes the `labels.cache` files so ultralytics re-scans.

**Deleting `labels.cache` is not optional.** Ultralytics reuses a stale cache without complaint, so edited labels are silently ignored otherwise.

Baseline scan (Sept 2026, 88,463 boxes across all splits):

| Finding | Count | Action |
|---|---|---|
| Zero-area boxes (`w=0` or `h=0`) | 17 | Dropped — unrepairable, and a genuine NaN source in the GIoU loss |
| Out-of-bounds coordinates | 2 | One clamped, one dropped (`cx=1.241`) |
| Boxes touching/marginally crossing the border | 483 | Benign — sub-pixel annotation rounding on objects that fill the frame. Snapped to the border. |

Note that the 17 degenerate boxes were **not** the cause of the first run's divergence — epochs 1–3 trained cleanly on exactly this data. They are a second, latent failure mode worth removing before committing to a multi-day run.

If Roboflow-side augmentation is enabled, note that ultralytics also augments on the fly (`mosaic`, `fliplr`, `scale`, `hsv_*`, `erasing` are all active by default). Baking augmentations into the export stacks them and can push boxes out of bounds; prefer augmenting at training time only.

## Training

`src/train/train_rtdetr.py` trains RT-DETR-l; `src/train/train_yolov10.py` trains YOLOv10n. Outputs go to `runs/<name>/`.

### Hyperparameters that matter

RT-DETR is a transformer and must **not** use the ultralytics defaults, which are tuned for SGD on YOLO:

| Setting | Default | Use | Why |
|---|---|---|---|
| `lr0` | 0.01 | **0.0001** | RT-DETR paper value for AdamW. The default is ~100x too high. |
| `amp` | True | **False** | fp16 + AdamW is a known NaN source on DETR-style decoders. |
| `batch` | -1 | **8** | Auto-batch picked 3, which is too noisy for a transformer (and used only 3 of 12 GB). |
| `warmup_bias_lr` | 0.1 | **0.0** | Wildly high for AdamW bias groups. |
| `weight_decay` | 0.0005 | **0.0001** | DETR-family standard. |

### Post-mortem: the `rtdetr_purestream` run (abandoned)

The first run diverged and was discarded. Recorded here so it is not repeated:

```
epoch 1   giou 1.235  cls 0.872   lr 0.0033   healthy
epoch 2   giou 1.080  cls 1.007   lr 0.0067   healthy
epoch 3   giou 1.072  cls 1.015   lr 0.0099   warmup completes
epoch 4   nan         nan         lr 0.0099   diverged
```

It died the moment the 3-epoch warmup finished and `lr0=0.01` took full effect. From epoch 13 on, `results.csv` logs bit-identical losses (`2.71739 / 82.7451 / 4.64592`, val cls `805.628`) because the trainer was repeatedly restoring the same NaN checkpoint. mAP was 0 from epoch 11.

**A checkpoint that has produced NaN cannot be resumed.** `src/train/resume_rtdetr.py` points at `runs/rtdetr_purestream/weights/last.pt` and will only reproduce the failure — restart from the pretrained `weights/rtdetr-l.pt` instead.

### Status

Pushed to a private GitHub remote. Code, config, and docs are versioned; data and weights are not.

## Related docs

- Full written report: Google Doc — "Chua Thian Poh" (report tab)
- Pitch deck: Canva — "Pure Stream Ecosystem Final Pitch Deck"
- System flow / ideation: Canva — "CTP Winning Idea"
