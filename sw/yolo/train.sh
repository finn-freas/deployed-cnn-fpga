#!/bin/bash
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
