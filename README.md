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

### Status

Local git repo — **not yet pushed to a GitHub remote**.

## Related docs

- Full written report: Google Doc — "Chua Thian Poh" (report tab)
- Pitch deck: Canva — "Pure Stream Ecosystem Final Pitch Deck"
- System flow / ideation: Canva — "CTP Winning Idea"
