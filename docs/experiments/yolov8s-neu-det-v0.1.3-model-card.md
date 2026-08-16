# Model Card: YOLOv8s NEU-DET v0.1.3 Baseline

## Model Details

| Field | Value |
| --- | --- |
| Architecture | YOLOv8s detection |
| Base checkpoint | Trusted local `yolov8s.pt` |
| Runtime | Vendored Ultralytics 8.0.182 |
| Training release | v0.1.3 |
| Local weight SHA256 | `5449dc254a57a43499963d466b0cb5f5f7d6e45520166d08dc7afed2f618d3a6` |
| Public weight location | Not published |

The weight is not distributed because dataset redistribution terms and the
license for derived weights have not been established in this repository.

## Intended Use

- Learning and reproducing a six-class steel surface defect workflow.
- Comparing local training, validation, and inference behavior.
- Identifying data and annotation improvements for NEU-DET-style experiments.

It is not intended for production quality decisions, safety-critical
inspection, or deployment without site-specific validation.

## Dataset And Training

The locally prepared split contains 1,440 training images and 360 validation
images. The six classes are `crazing`, `inclusion`, `patches`,
`pitted_surface`, `rolled-in_scale`, and `scratches`.

```bash
python train.py \
  --model yolov8s.pt \
  --data dataset.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device mps
```

Training used seed 0 and deterministic mode from the vendored trainer. The
50-epoch run completed in 3.051 hours on Apple M5.

## Evaluation

Independent CPU validation used 360 images with 822 annotated instances.

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

## Environment

- macOS 26.5.2 arm64, Apple M5
- Python 3.11.15
- PyTorch 2.5.1 and TorchVision 0.20.1
- NumPy 2.3.5
- MPS training and final trainer validation
- CPU independent validation

## Limitations

- The `crazing` class has the lowest recall and mAP by a substantial margin.
- Results depend on this exact local split and are not directly comparable to
  papers using different splits, preprocessing, or evaluation settings.
- Public benchmark imagery does not represent arbitrary factory cameras,
  lighting, materials, defect prevalence, or operating conditions.
- The project uses an older vendored runtime for compatibility.
- Serialized PyTorch checkpoints should only be loaded from trusted sources.

## Responsible Release Status

- [x] Environment, class order, split size, command, and metrics documented.
- [x] Weight checksum recorded locally.
- [x] Known limitations and weak-class behavior documented.
- [ ] Dataset source terms independently confirmed for redistribution.
- [ ] Derived-weight license confirmed.
- [ ] Weight published outside git.
