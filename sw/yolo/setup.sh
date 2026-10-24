#!/bin/bash
# Set up the YOLOv5 training environment (run from the repo root).
#   - clones ultralytics/yolov5 at a pinned commit into build/yolov5
#   - creates .venv-yolo (Python 3.12) with CUDA PyTorch + yolov5 requirements
# RTX 50-series (Blackwell) GPUs need CUDA >= 12.8 wheels, hence the cu130 index.
set -euo pipefail

YOLOV5_COMMIT=af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce
TORCH_INDEX=https://download.pytorch.org/whl/cu130
# Tolerate slow connection setup (e.g. networks where IPv6 stalls before IPv4 fallback).
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}

if [ ! -d build/yolov5/.git ]; then
    git clone -q https://github.com/ultralytics/yolov5 build/yolov5
fi
git -C build/yolov5 checkout -q "$YOLOV5_COMMIT"

if [ ! -x .venv-yolo/bin/python ]; then
    .venv/bin/uv venv -q -p 3.12 .venv-yolo
fi
.venv/bin/uv pip install -q -p .venv-yolo/bin/python torch torchvision --index-url "$TORCH_INDEX"
# yolov5 requirements minus torch/torchvision (already installed from the CUDA index)
grep -vE '^(torch|torchvision)[>=<]' build/yolov5/requirements.txt > build/yolov5/requirements-notorch.txt
.venv/bin/uv pip install -q -p .venv-yolo/bin/python -r build/yolov5/requirements-notorch.txt onnx onnxsim
.venv-yolo/bin/python -c 'import torch, torchvision; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
