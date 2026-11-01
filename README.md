# Steel Surface Defect Detection with YOLOv8

Train, validate and run a YOLOv8 steel surface defect detector from a desktop
GUI or the command line.

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/YfengJ/steel-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YfengJ/steel-defect-detection/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/YfengJ/steel-defect-detection)](https://github.com/YfengJ/steel-defect-detection/releases)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics%208.0.182-purple)
![Dataset](https://img.shields.io/badge/Dataset-NEU--DET-orange)
![License](https://img.shields.io/badge/License-AGPL--3.0-red)

![Single-image detection in the desktop GUI](screenshots/predict.png)

This is a learning-oriented project for the six NEU-DET steel surface defect
classes. It covers the whole loop: convert VOC annotations, train, validate,
detect on images, folders and video, and inspect the results in a desktop app.
Datasets and weights are not bundled. You prepare them locally
([docs/dataset.md](docs/dataset.md)).

## Highlights

- **Desktop app** with a sidebar workspace, dark and light themes, and a
  CPU / Apple MPS / CUDA device picker on every page. It runs on Windows,
  macOS and Linux, with a one-click setup and launcher for Windows.
- **Single image:** switch between the original and the annotated image. Every
  detection is listed with its confidence and area.
- **Batch:** per-folder statistics, class and confidence charts, and a per-image
  table. Double-click a row to open its result image.
- **Video and camera:** live inference that drops stale frames, with FPS and
  per-frame defect counts.
- **Training:** each run writes to its own directory and draws live mAP and loss
  curves from `results.csv`. A run can be stopped from the app.
- **Validation:** overall precision, recall and mAP, plus per-class results
  parsed from `val.py`.
- **CLI scripts** for every step, tested in CI (Ruff, unit tests, CLI smoke tests,
  and a synthetic CPU train and validate run) on Linux and Windows.

## Quick Start

```bash
git clone https://github.com/YfengJ/steel-defect-detection.git
cd steel-defect-detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows you can instead run `scripts\windows\setup.bat` (add `cuda` for an
NVIDIA GPU) and then double-click `start_gui.bat`. See
[docs/windows.md](docs/windows.md).

Launch the GUI:

```bash
python ui.py
```

Or detect on one image from the command line:

```bash
python predict.py --model /path/to/trusted-best.pt --source /path/to/image.jpg --device cpu
```

Results go to `runs/detect/<name>/`. Only load weights you trust: PyTorch
checkpoints can contain serialized Python objects. Platform guides:
[Windows](docs/windows.md) and [macOS](docs/macos.md). For a dataset-free first run, see
[docs/sample_inference.md](docs/sample_inference.md).

## GUI Tour

| Batch detection | Video detection |
| --- | --- |
| ![Batch detection](screenshots/batch.png) | ![Video detection](screenshots/video.png) |
| **Training** | **Validation** |
| ![Training](screenshots/train.png) | ![Validation](screenshots/val.png) |

The light theme is one click away in the sidebar:

![Light theme](screenshots/light.png)

The screenshots use the v0.1.3 YOLOv8s baseline weight described below. The
images come from the NEU-DET validation split. The training page shows the
`results.csv` of that baseline run.

## Reproduced Baseline

YOLOv8s, 50 epochs, 640 px, batch 16. It was trained on Apple M5 (MPS) with a
local 1,440 / 360 image split, then validated separately on CPU against 822
labeled defect instances:

| Precision | Recall | mAP50 | mAP50-95 | mAP75 |
| ---: | ---: | ---: | ---: | ---: |
| 0.719 | 0.721 | 0.7637 | 0.4455 | 0.4530 |

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `crazing` | 0.625 | 0.383 | 0.472 | 0.1861 |
| `inclusion` | 0.716 | 0.767 | 0.816 | 0.4821 |
| `patches` | 0.741 | 0.891 | 0.904 | 0.6045 |
| `pitted_surface` | 0.825 | 0.788 | 0.847 | 0.5253 |
| `rolled-in_scale` | 0.611 | 0.567 | 0.630 | 0.2883 |
| `scratches` | 0.795 | 0.928 | 0.913 | 0.5864 |

These numbers describe one public-dataset split, not factory performance.
Details, limitations and the weight checksum are in the
[experiment log](docs/experiments/yolov8s-neu-det-baseline.md) and the
[model card](docs/experiments/yolov8s-neu-det-v0.1.3-model-card.md). The weight
itself is not published.

## Command-line Usage

### Train

```bash
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device cpu
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 8  --device mps
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device 0
```

| Option | Default | Notes |
| --- | --- | --- |
| `--imgsz` | `640` | Training image size |
| `--workers` | `2` | Set to `0` if Windows multiprocessing fails |
| `--project` / `--name` | `runs/detect` / `train_result` | An existing run directory is never reused; the run becomes `train_result2`, `train_result3`, … |
| `--exist-ok` | off | Reuse the directory on purpose. This overwrites the old weights and appends to the old `results.csv` |

The final weight path is printed when training ends.

### Validate

```bash
python val.py --model runs/detect/train_result/weights/best.pt --data dataset.yaml
python val.py --model best.pt --data dataset.yaml --split test --device cpu
```

`val.py` prints precision, recall, mAP50, mAP50-95 and mAP75, and the
per-class results. It also prints one `METRICS_JSON {...}` line for tools such as
the GUI.

### Predict

```bash
python predict.py --model best.pt --source path/to/image.jpg --conf 0.25 --device cpu
python predict.py --model best.pt --source path/to/folder/ --name my_batch
```

Annotated images and YOLO labels with confidences are saved under
`runs/detect/<name>/`.

### Video

```bash
python video_predict.py best.pt path/to/video.mp4 --output output.mp4 --conf 0.25
python video_predict.py best.pt 0          # camera index 0
```

## Dataset Preparation

Use the [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)
or any dataset with the same class mapping:

```text
datasets/NEU-DET/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── annotations/            # optional VOC XML
```

YOLO labels use `class_id x_center y_center width height`, normalized to
`0.0`–`1.0`. To convert VOC XML annotations, run:

```bash
python translate.py
```

This writes `labels/<split>/` for every existing `images/<split>` directory.
`dataset.yaml` resolves its `path` relative to the YAML file itself. See
[docs/dataset.md](docs/dataset.md) for the full guide.

| ID | Class | Meaning |
| --- | --- | --- |
| 0 | `crazing` | Fine crack-like surface patterns |
| 1 | `inclusion` | Non-metallic inclusions |
| 2 | `patches` | Irregular surface patches |
| 3 | `pitted_surface` | Pitting or corrosion-like marks |
| 4 | `rolled-in_scale` | Oxide scale rolled into the surface |
| 5 | `scratches` | Linear scratches |

## Project Structure

```text
steel-defect-detection/
├── ui.py                  # Desktop GUI (ttkbootstrap)
├── gui_logic.py           # Display-free GUI helpers: label parsing, batch stats, metrics
├── train.py               # Training CLI
├── val.py                 # Validation CLI
├── predict.py             # Image / folder prediction CLI
├── video_predict.py       # Video and camera inference CLI
├── translate.py           # VOC XML → YOLO TXT conversion
├── path_validation.py     # Local path and dataset YAML checks
├── cli_support.py         # Shared CLI helpers (UTF-8 console output)
├── dataset.yaml           # Default dataset configuration
├── start_gui.bat          # Windows GUI launcher (no console window)
├── scripts/windows/       # Windows one-click setup
├── requirements*.txt      # Runtime and contributor dependencies
├── tests/                 # Unit, GUI-callback, CLI and training smoke tests
├── docs/                  # Setup, dataset, experiment and maintenance docs
└── ultralytics/           # Vendored Ultralytics 8.0.182
```

`datasets/`, `runs/`, `weights/` and `*.pt` files are git-ignored.

## Compatibility Notes

- The vendored **Ultralytics 8.0.182** keeps the original runtime reproducible.
  `ultralytics/`, most of `docs/` and the generic `examples/` come from the
  [Ultralytics project](https://github.com/ultralytics/ultralytics) under
  AGPL-3.0. Changes to that core are deliberately conservative.
- **PyTorch < 2.6.** The newer `torch.load` default breaks this checkpoint
  loader.
- **NumPy < 2.4.** The vendored metrics code still calls `numpy.trapz`.
- **ttkbootstrap 1.10+ and 2.x** both work. The GUI registers its own themes.

## Documentation

- [Windows](docs/windows.md) · [macOS and Apple Silicon](docs/macos.md) ·
  [Sample inference](docs/sample_inference.md) ·
  [Dataset preparation](docs/dataset.md) ·
  [Troubleshooting](docs/troubleshooting.md)
- [Baseline experiment log](docs/experiments/yolov8s-neu-det-baseline.md) ·
  [v0.1.3 model card](docs/experiments/yolov8s-neu-det-v0.1.3-model-card.md) ·
  [Model card template](docs/model_card.md)
- [GUI smoke checklist](docs/gui-smoke.md) · [Roadmap](ROADMAP.md) ·
  [Release notes](RELEASE_NOTES.md)
- [Contributing](CONTRIBUTING.md) · [Code of conduct](CODE_OF_CONDUCT.md) ·
  [Support](SUPPORT.md) · [Security](SECURITY.md)

## Community

Bug reports, reproduction notes and focused pull requests are welcome. Share a
tested environment or result in the
[reproduction reports discussion](https://github.com/YfengJ/steel-defect-detection/discussions/57).
If the project helps your study, a star supports continued maintenance. Please
do not use star exchanges.

## License

[AGPL-3.0](LICENSE). YOLOv8 is provided by
[Ultralytics](https://github.com/ultralytics/ultralytics) under its upstream
license.
