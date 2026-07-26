# Troubleshooting

## `ModuleNotFoundError: No module named 'ultralytics'`

This repository vendors Ultralytics under `ultralytics/`. First confirm that you
are running commands from the repository root, then install dependencies inside
the active virtual environment:

```bash
python -m pip install -r requirements.txt
```

Then confirm:

```bash
python - <<'PY'
import ultralytics
from ultralytics import YOLO

print("ultralytics import ok:", ultralytics.__file__)
PY
```

The printed path should point inside this cloned repository. Do not install a
second PyPI copy of Ultralytics over the vendored source.

## `找不到模型权重`

The CLI and GUI require a local model file. Download a compatible YOLOv8 weight
or use your own trained `best.pt`, then pass its real path:

```bash
python predict.py --model weights/best.pt --source path/to/image.jpg
```

Weights are intentionally excluded from git. Only load checkpoints from sources
you trust; see [../SECURITY.md](../SECURITY.md).

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
