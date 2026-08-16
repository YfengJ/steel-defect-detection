# Steel Surface Defect Detection with YOLOv8

An open source YOLOv8 project for steel surface defect detection, training, validation, inference, and GUI demos.

[English](README.md) | [简体中文](README.zh-CN.md)

![Python](https://img.shields.io/badge/Python-3.10%20recommended-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![GUI](https://img.shields.io/badge/GUI-ttkbootstrap-green)
![Dataset](https://img.shields.io/badge/Dataset-NEU--DET-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)
[![CI](https://github.com/YfengJ/steel-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YfengJ/steel-defect-detection/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/YfengJ/steel-defect-detection)](https://github.com/YfengJ/steel-defect-detection/releases)

---

## Overview

This repository is a learning-oriented steel surface defect detection project
built on **YOLOv8**. It targets NEU-DET style steel defect datasets and
provides a practical workflow for model training, validation, image inference,
video inference, and a desktop GUI demo.

It is designed for students, computer vision beginners, and industrial vision
learners who want a real, runnable, and maintainable open source project rather
than an unmaintained collection of scripts.

> Datasets, training outputs, and model weights are not included in this
> repository. Prepare them locally by following [docs/dataset.md](docs/dataset.md).

![Steel defect detection result](screenshots/predict.png)

## Start Here

```bash
python -m pip install -r requirements.txt
python predict.py --model /path/to/trusted-best.pt --source /path/to/image.jpg --device cpu
```

The rendered result is saved under `runs/detect/`. Bring a trusted local weight
and an image you are allowed to process; the repository intentionally does not
bundle either. See the [sample inference guide](docs/sample_inference.md) for
CPU, Apple Silicon MPS, and NVIDIA CUDA examples.

## Current Status

- Current version: `v0.1.3`.
- CI runs focused Ruff checks, unit tests, CLI smoke tests, repository hygiene,
  and Python compilation on Python 3.10.
- The same health checks run automatically every Monday and can be started
  manually from GitHub Actions.
- Documentation exists for macOS, dataset preparation, model cards,
  sample inference, troubleshooting, support, security, and the roadmap.
- Datasets and model weights are not included in the repository; users should
  prepare them locally.
- The GUI checks local model, dataset, image, folder, and video paths before
  launching work.
- Apple Silicon MPS has been checked with PyTorch 2.5.1, a tensor operation,
  and one temporary image inference run.
- Planned v0.2.0 work focuses on reproducible training presets, deeper runtime
  smoke tests, and clearer experiment outputs.
- A fresh YOLOv8s/NEU-DET reproduction is being recorded transparently in the
  [experiment log](docs/experiments/yolov8s-neu-det-baseline.md). Historical
  metrics are not presented as verified because the original checkpoint and
  result files were lost during a device migration.

## Features

| Feature | Description |
| --- | --- |
| Single-image inference | Load one image and visualize detected defect boxes. |
| Batch inference | Process an image folder and generate summary reports. |
| Video inference | Run defect detection on video files or camera streams. |
| Model training | Train YOLOv8 models from the command line or GUI. |
| Model validation | Evaluate trained weights and inspect mAP metrics. |
| GUI demo | Use a `ttkbootstrap` desktop interface for common workflows. |
| Device selection | Choose CPU, Apple Silicon MPS, or NVIDIA CUDA for CLI prediction. |

## Documentation

- [macOS and Apple Silicon guide](docs/macos.md)
- [Sample inference with local files](docs/sample_inference.md)
- [Dataset preparation](docs/dataset.md)
- [Model card template](docs/model_card.md)
- [YOLOv8s baseline experiment log](docs/experiments/yolov8s-neu-det-baseline.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Roadmap](ROADMAP.md)
- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Support](SUPPORT.md)
- [Security policy](SECURITY.md)

## Screenshots

### Single-image inference

![Single-image inference](screenshots/predict.png)

### Batch inference

![Batch inference](screenshots/batch.png)

### Video inference

![Video inference](screenshots/video.png)

### Training

![Training](screenshots/train.png)

### Validation

![Validation](screenshots/val.png)

## Project Structure

```text
steel-defect-detection/
├── ui.py                  # Desktop GUI built with ttkbootstrap
├── train.py               # YOLOv8 training entrypoint
├── predict.py             # Image and batch prediction entrypoint
├── val.py                 # Validation entrypoint
├── video_predict.py       # Video inference helper
├── translate.py           # VOC XML to YOLO TXT conversion utility
├── path_validation.py     # Dependency-light GUI path checks
├── dataset.yaml           # Dataset configuration
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Focused lint and test dependencies
├── docs/                  # Project setup and maintenance docs
├── .github/               # CI, Dependabot, and issue templates
├── datasets/              # Local datasets, ignored by git
├── runs/                  # Local training/inference outputs, ignored by git
└── weights/ or *.pt        # Local model weights, ignored by git
```

## Maintained Scope And Upstream

This repository vendors **Ultralytics 8.0.182** so the original project remains
reproducible without silently changing its YOLO runtime. The vendored
`ultralytics/`, much of `docs/`, and generic `examples/` originate from the
[Ultralytics project](https://github.com/ultralytics/ultralytics) under
AGPL-3.0.

Project-specific maintenance focuses on:

- `train.py`, `val.py`, `predict.py`, and `video_predict.py`;
- the `ttkbootstrap` desktop workflow in `ui.py`;
- dataset conversion and path validation;
- CPU, CUDA, and Apple Silicon MPS setup and compatibility;
- focused tests, CI, release documentation, and contributor support.

Changes to the vendored YOLO core are intentionally conservative. A future
move to a pinned external package is tracked as a compatibility project rather
than being presented as original model architecture work.

## Defect Classes

The default configuration follows the six common NEU-DET steel surface defect classes:

| ID | Class | Meaning |
| --- | --- | --- |
| 0 | `crazing` | Fine crack-like surface patterns |
| 1 | `inclusion` | Non-metallic inclusion defects |
| 2 | `patches` | Irregular surface patches |
| 3 | `pitted_surface` | Pitting or corrosion-like marks |
| 4 | `rolled-in_scale` | Oxide scale rolled into the surface |
| 5 | `scratches` | Linear scratch defects |

## Quick Start

### 1. Requirements

- Python 3.10 recommended
- CPU, Apple Silicon MPS, or NVIDIA CUDA
- Windows, Linux, or macOS

macOS users should start with [docs/macos.md](docs/macos.md).

### 2. Install

```bash
git clone https://github.com/YfengJ/steel-defect-detection.git
cd steel-defect-detection

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Prepare these files locally:

- Dataset: see [docs/dataset.md](docs/dataset.md)
- Weights: download a YOLOv8 pretrained weight such as `yolov8n.pt`, or use your own trained `best.pt`

For a dataset-free first run with your own local image and trusted weight, use
[docs/sample_inference.md](docs/sample_inference.md).

### 3. Launch the GUI

```bash
python ui.py
```

The GUI includes tabs for image prediction, batch prediction, video inference, training, and validation.

## Command-line Usage

### Train

```bash
# CPU
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device cpu

# Apple Silicon MPS
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 8 --device mps

# NVIDIA CUDA
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device cuda
```

### Validate

```bash
python val.py --model runs/detect/train_result/weights/best.pt --data dataset.yaml
```

### Predict

```bash
python predict.py \
  --model runs/detect/train_result/weights/best.pt \
  --source path/to/image.jpg \
  --device cpu
```

## Dataset Preparation

This project can use the [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)
or another dataset with the same class mapping.

Expected local layout:

```text
datasets/NEU-DET/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
```

YOLO label files should use:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `0.0` to `1.0`.

If your annotations are VOC XML files, place them under `datasets/NEU-DET/annotations/` and run:

```bash
python translate.py
```

## Model Weights

Model weights are intentionally excluded from git. Keep large artifacts locally,
in cloud storage, or in GitHub Releases.

Only load weights from sources you trust. PyTorch checkpoints may contain
serialized Python objects.

Common local choices:

| Weight | Use case |
| --- | --- |
| `yolov8n.pt` | Fastest baseline for CPU or small experiments |
| `yolov8s.pt` | Better speed/accuracy balance |
| `yolov8m.pt` | Medium experiments when hardware allows |
| `best.pt` | Your trained defect detector |

## Dependency Updates

Dependabot is enabled for Python dependencies and GitHub Actions. Large
dependency jumps are reviewed conservatively because PyTorch, OpenCV, NumPy, and
Ultralytics compatibility can be sensitive across platforms. This repository
vendors Ultralytics `8.0.182`; PyTorch is capped below `2.6` because newer
`torch.load` defaults are incompatible with that checkpoint loader. NumPy is
capped below `2.4` because this vendored runtime still calls `numpy.trapz`.

## Release Notes

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for v0.1.3 changes, known limitations, and next plans.

## Community

Real bug reports, reproduction notes, and focused pull requests are welcome.
Share a tested environment or result in the
[reproduction reports discussion](https://github.com/YfengJ/steel-defect-detection/discussions/57).
If this project helps your study or research, consider starring the repository
to support continued maintenance. Please do not use star exchanges or other
artificial promotion.

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).

YOLOv8 is provided by [Ultralytics](https://github.com/ultralytics/ultralytics) and follows its upstream license terms.
