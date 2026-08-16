# YOLOv8s NEU-DET Baseline Experiment

This page is the source of truth for the repository's fresh baseline run. It
exists because the original `best.pt` and result files were lost during a
device migration. The maintainer remembers an approximate historical mAP near
70%, but that recollection is **not a verified result** and is not used below.

## Status

- Dataset structure and label pairing: verified locally.
- One-epoch training and validation smoke test: in progress for v0.1.3.
- Full 50-epoch baseline: pending completion.
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
| Epochs | 50 planned |
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
  --batch 8 \
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

Verified metrics will be added only after the full command completes and the
generated `results.csv` is reviewed. The final record must include mAP50,
mAP50-95, precision, recall, per-class mAP50-95, runtime, the training commit,
and a SHA256 checksum for any released weight.

## Limitations

- This public dataset baseline does not establish production performance.
- NEU-DET imagery may not represent a specific factory, camera, steel grade,
  lighting setup, or defect prevalence.
- MPS results are not directly interchangeable with CUDA throughput results.
- PyTorch checkpoints must only be loaded from trusted sources.
