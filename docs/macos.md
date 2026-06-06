# macOS and Apple Silicon

This project was originally developed in a Windows CUDA environment. On macOS, especially Apple Silicon, use CPU or PyTorch MPS instead of CUDA.

## Recommended Environment

- macOS 12.3 or newer for MPS.
- Apple Silicon Python, not a Rosetta x86 Python, when using MPS.
- Python 3.10 is recommended because the CI workflow uses Python 3.10.
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
