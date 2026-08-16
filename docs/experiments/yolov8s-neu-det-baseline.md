# YOLOv8s NEU-DET Baseline Experiment

This page is the source of truth for the repository's fresh baseline run. It
exists because the original `best.pt` and result files were lost during a
device migration. The maintainer remembers an approximate historical mAP near
70%, but that recollection is **not a verified result** and is not used below.

## Status

- Dataset structure and label pairing: verified locally.
- One-epoch training and validation smoke test: completed.
- Full 50-epoch baseline: completed and independently validated.
- Published weight: not yet available.

## Reproduction Contract

| Field | Value |
| --- | --- |
| Architecture | YOLOv8s detection |
| Base checkpoint | `yolov8s.pt` from a trusted local source |
| Ultralytics runtime | Vendored `8.0.182` |
| Dataset | NEU-DET style six-class split, prepared locally |
| Train images / labels | 1,440 / 1,440 |
| Validation images / labels | 360 / 360 |
| Image size | 640 |
| Epochs | 50 completed |
| Batch size | 16 |
| Random seed | 0, deterministic mode enabled by the vendored trainer |
| Dataset and weights in git | No |

The local dataset scan reported three train images with one duplicate label
removed in memory and four background images across the train/validation
splits. No corrupt images were reported. These observations should be checked
against any independently prepared split.

## Commands

Apple Silicon MPS:

```bash
python train.py \
  --model /path/to/yolov8s.pt \
  --data dataset.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device mps
```

Validation:

```bash
python val.py \
  --model runs/detect/train_result/weights/best.pt \
  --data dataset.yaml
```

The CLI resolves the dataset root relative to the supplied YAML file. This
keeps a stale user-level Ultralytics `datasets_dir` setting from redirecting the
run to another checkout.

## Environment Under Verification

| Component | Value |
| --- | --- |
| Hardware | Apple M5 |
| Operating system | macOS 26.5.2 arm64 |
| Python | 3.11.15 |
| PyTorch | 2.5.1 |
| TorchVision | 0.20.1 |
| NumPy | 2.3.5 |
| Device | MPS |

## Results

The 50-epoch MPS run completed in 3.051 hours. The trainer's final validation
reported mAP50 0.764 and mAP50-95 0.445. A separate CPU invocation of `val.py`
against the generated local `best.pt` produced the following reproducible
summary:

| Precision | Recall | mAP50 | mAP50-95 | mAP75 |
| ---: | ---: | ---: | ---: | ---: |
| 0.719 | 0.721 | 0.7637 | 0.4455 | 0.4530 |

Per-class results from the independent CPU validation:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crazing` | 360 | 148 | 0.625 | 0.383 | 0.472 | 0.1861 |
| `inclusion` | 360 | 174 | 0.716 | 0.767 | 0.816 | 0.4821 |
| `patches` | 360 | 192 | 0.741 | 0.891 | 0.904 | 0.6045 |
| `pitted_surface` | 360 | 85 | 0.825 | 0.788 | 0.847 | 0.5253 |
| `rolled-in_scale` | 360 | 127 | 0.611 | 0.567 | 0.630 | 0.2883 |
| `scratches` | 360 | 96 | 0.795 | 0.928 | 0.913 | 0.5864 |

The independent CPU run measured 226.5 ms inference per image at 640 pixels;
the trainer's final MPS validation measured 24.5 ms per image. These timings
are local observations, not a controlled hardware benchmark.

Local `best.pt` SHA256:

```text
5449dc254a57a43499963d466b0cb5f5f7d6e45520166d08dc7afed2f618d3a6
```

The checkpoint is intentionally not committed or attached to the release while
dataset redistribution and derived-weight licensing remain undocumented. The
training output remains under ignored `runs/` storage.

## Limitations

- This public dataset baseline does not establish production performance.
- NEU-DET imagery may not represent a specific factory, camera, steel grade,
  lighting setup, or defect prevalence.
- MPS results are not directly interchangeable with CUDA throughput results.
- `crazing` recall and mAP were substantially weaker than the other classes in
  this split and need targeted data/annotation review before model comparison.
- PyTorch checkpoints must only be loaded from trusted sources.
