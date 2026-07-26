# macOS and Apple Silicon

This project was originally developed in a Windows CUDA environment. On macOS, especially Apple Silicon, use CPU or PyTorch MPS instead of CUDA.

## Recommended Environment

- macOS 12.3 or newer for MPS.
- Apple Silicon Python, not a Rosetta x86 Python, when using MPS.
- Python 3.10 to 3.12 is supported; 3.10 is recommended because CI uses it.
- A local virtual environment.

Official references:

- PyTorch local installation: https://docs.pytorch.org/get-started/locally/
- PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps.html

## Setup

```bash
git clone https://github.com/YfengJ/steel-defect-detection.git
cd steel-defect-detection

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.10` is not available, install Python 3.10 with Homebrew or pyenv first.
Python 3.13 is not currently supported by the project's PyTorch compatibility range.
Run commands from the repository root so Python imports the vendored Ultralytics
8.0.182 source under `ultralytics/`.

## Check MPS Availability

```bash
python - <<'PY'
import platform
import torch

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
PY
```

If `mps available` is `False`, use `--device cpu`.

## Runtime Smoke Checks

The automated smoke test builds the bundled YOLOv8 architecture and predicts a
64 x 64 synthetic image without downloading weights or datasets:

```bash
python -m pytest tests/test_runtime_smoke.py -q
```

For an Apple Silicon MPS check, run:

```bash
python - <<'PY'
import numpy as np
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")
image = np.zeros((64, 64, 3), dtype=np.uint8)
results = model.predict(image, imgsz=64, device="mps", verbose=False)
print("MPS prediction results:", len(results))
PY
```

The v0.1.1 maintenance pass verified this no-weight MPS prediction on Apple
Silicon with PyTorch 2.5.1. Full training still depends on your dataset, model
weights, memory, and the operations used by your selected model.

## Prepare Dataset and Weights

The repository does not include NEU-DET images, labels, or trained model weights. See [dataset.md](dataset.md) for the expected layout.

Recommended local paths:

```text
datasets/NEU-DET/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Download YOLOv8 pretrained weights yourself, for example `yolov8n.pt`, or place your own trained `best.pt` under a local path such as `weights/`.
Only use checkpoints from sources you trust; see [../SECURITY.md](../SECURITY.md).

## Train on Apple Silicon

```bash
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 8 --device mps
```

If training is unstable or an operation is not implemented on MPS, fall back to CPU:

```bash
python train.py --model yolov8n.pt --data dataset.yaml --epochs 5 --batch 4 --device cpu
```

## Validate and Predict

```bash
python val.py --model runs/detect/train_result/weights/best.pt --data dataset.yaml
python predict.py --model runs/detect/train_result/weights/best.pt --source path/to/image.jpg
```

## Launch GUI

```bash
python ui.py
```

If Tkinter or fonts behave differently on macOS, start with command-line training and prediction first, then open the GUI after the environment is confirmed.
