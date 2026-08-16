# Troubleshooting

## Dataset Path Points To Another Checkout

If an error mentions a missing dataset under an unrelated worktree or user
directory, first update to v0.1.3 or newer. Earlier scripts passed relative
dataset roots directly to Ultralytics 8.0.182, which resolves them against its
global `datasets_dir` setting. Current `train.py` and `val.py` generate a
temporary absolute-path configuration without modifying your source YAML.

Also confirm the configured paths locally:

```bash
python train.py --model yolov8s.pt --data dataset.yaml --epochs 1 --device cpu
```

## NumPy Has No Attribute `trapz`

Ultralytics 8.0.182 uses `numpy.trapz`, which was removed in NumPy 2.4. Install
the repository's tested requirement range rather than an unconstrained latest
NumPy:

```bash
python -m pip install --upgrade --force-reinstall "numpy>=1.22.2,<2.4"
python -m pip install -r requirements.txt
```

Use a fresh virtual environment if another package keeps upgrading NumPy.

## `ModuleNotFoundError: No module named 'ultralytics'`

Install dependencies again inside the active virtual environment:

```bash
python -m pip install -r requirements.txt
```

Then confirm:

```bash
python - <<'PY'
from ultralytics import YOLO
print("ultralytics import ok")
PY
```

## `Dataset 'dataset.yaml' images not found`

Check that `dataset.yaml` points to your local dataset root and that the `images/train`, `images/val`, and `labels` directories exist. See [dataset.md](dataset.md).

## `MPS is not available`

MPS only works on supported macOS/PyTorch/Apple Silicon combinations. Check availability:

```bash
python - <<'PY'
import torch
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
PY
```

If it prints `False`, use `--device cpu`.

## CUDA Errors on macOS

CUDA is for NVIDIA GPUs and is not available on Apple Silicon. Use:

```bash
python train.py --model yolov8n.pt --data dataset.yaml --device mps
```

or:

```bash
python train.py --model yolov8n.pt --data dataset.yaml --device cpu
```

## Training Runs Out of Memory

Lower batch size and image size:

```bash
python train.py --model yolov8n.pt --data dataset.yaml --epochs 20 --batch 4 --imgsz 512 --device mps
```

For CPU-only machines, start with `--epochs 1 --batch 2 --imgsz 416` to confirm the pipeline.

## GUI Does Not Start on macOS

Install dependencies and confirm Tkinter is available:

```bash
python -m pip install -r requirements.txt
python - <<'PY'
import tkinter
print("tkinter ok")
PY
```

If the GUI still fails, run `train.py` or `predict.py` from the command line and include the full traceback in a bug report.
