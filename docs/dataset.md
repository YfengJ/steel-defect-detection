# Dataset Preparation

Datasets must be prepared locally. Do not commit dataset files, generated labels, training runs, or model weights to this repository.

## Expected Task

This project is configured for YOLO object detection of six NEU-DET steel surface defect classes:

| ID | Class |
| --- | --- |
| 0 | crazing |
| 1 | inclusion |
| 2 | patches |
| 3 | pitted_surface |
| 4 | rolled-in_scale |
| 5 | scratches |

## Recommended Layout

Use this local directory structure:

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

The `annotations/` directory is optional after labels have been converted to YOLO format.

## dataset.yaml

The default `dataset.yaml` expects the dataset under:

```yaml
path: datasets/NEU-DET
train: images/train
val: images/val
test: images/test
```

If your dataset is elsewhere, update `path` to an absolute path or a path relative to the repository root.

`train.py` and `val.py` normalize that root relative to the YAML file before
calling the vendored Ultralytics runtime. This prevents an old user-level
Ultralytics `datasets_dir` setting from redirecting a run to another checkout.

## Label Format

YOLO detection labels use one `.txt` file per image:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to `0.0` to `1.0`.

## VOC XML Conversion

If your annotations are VOC XML files, place them in `datasets/NEU-DET/annotations/` and run:

```bash
python translate.py
```

Review the generated label files before training.

## What Not to Commit

The repository `.gitignore` excludes common generated and large artifacts, including:

- `datasets/`
- `runs/`
- `weights/`
- `*.pt`
- `*.pth`
- `*.onnx`

Keep large files in local storage, cloud storage, GitHub Releases, or another artifact host instead of the source repository.
