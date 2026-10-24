#!/bin/bash
# Train YOLOv5n-ReLU at 320x320 on the trash dataset (run from the repo root).
#   sw/yolo/train.sh <path/to/data.yaml> [epochs]
# Hyperparameters follow the author's original YOLOv5s run (Trash_Detection/Hyp.yaml).
# Initialized from COCO-pretrained yolov5n.pt: layer shapes match exactly and only the
# activation differs (SiLU -> ReLU), so fine-tuning converges far faster than scratch.
set -euo pipefail

DATA=$(realpath "$1")
EPOCHS=${2:-60}
ROOT=$(pwd)

cd build/yolov5
../../.venv-yolo/bin/python train.py \
    --img 320 \
    --batch 64 \
    --epochs "$EPOCHS" \
    --data "$DATA" \
    --cfg "$ROOT/sw/yolo/yolov5n-relu.yaml" \
    --weights yolov5n.pt \
    --hyp "$ROOT/sw/yolo/hyp_trash.yaml" \
    --project "$ROOT/build/yolo_runs" \
    --name yolov5n_relu_320 \
    --cache ram \
    --workers 8 \
    --exist-ok
