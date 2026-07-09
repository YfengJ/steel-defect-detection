# Model Card Template for Steel Defect Detection Weights

Use this template when publishing trained steel surface defect detection weights
through GitHub Releases or external storage.

This repository does not currently ship official trained weights. Keep model
weights, datasets, and generated `runs/` outputs outside the git repository.

## Model Details

| Field | Value |
| --- | --- |
| Model name | `<model-name>` |
| Version | `<version-or-release-tag>` |
| Architecture | YOLOv8 detect |
| Base checkpoint | `<for example: yolov8n.pt, yolov8s.pt, or custom>` |
| Training code commit | `<git commit SHA>` |
| Weight location | `<GitHub Release asset or external download URL>` |
| License | `<license for the released weights>` |

## Intended Use

Describe what the weights are meant to support.

- Learning and experimentation with steel surface defect detection.
- Reproducing a documented training and inference workflow.
- Comparing model behavior on NEU-DET style defect classes.

## Non-Goals

State what the weights should not be used for without additional validation.

- Production quality control decisions without local industrial validation.
- Safety-critical inspection workflows.
- Defect classes, camera setups, materials, or lighting conditions outside the
  documented training and evaluation scope.

## Dataset Summary

| Field | Value |
| --- | --- |
| Dataset name | `<dataset name>` |
| Dataset source | `<download page, paper, or internal source>` |
| Dataset license | `<dataset license or usage terms>` |
| Train images | `<count>` |
| Validation images | `<count>` |
| Test images | `<count, if used>` |
| Annotation format | YOLO TXT / converted from VOC XML / other |

Do not commit the dataset to this repository. Link to public dataset sources or
describe the local preparation process in [dataset.md](dataset.md).

## Class Mapping

Update this table if your trained weights use a different class order.

| ID | Class | Meaning |
| --- | --- | --- |
| 0 | `crazing` | Fine crack-like surface patterns |
| 1 | `inclusion` | Non-metallic inclusion defects |
| 2 | `patches` | Irregular surface patches |
| 3 | `pitted_surface` | Pitting or corrosion-like marks |
| 4 | `rolled-in_scale` | Oxide scale rolled into the surface |
| 5 | `scratches` | Linear scratch defects |

## Training Configuration

| Field | Value |
| --- | --- |
| Image size | `<imgsz>` |
| Epochs | `<epochs>` |
| Batch size | `<batch>` |
| Device | CPU / MPS / CUDA |
| Optimizer | `<optimizer, if customized>` |
| Key augmentations | `<brief summary>` |
| Dataset YAML | `<path or link>` |

Example command:

```bash
python train.py \
  --model yolov8n.pt \
  --data dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device cpu
```

## Evaluation

Report metrics on a held-out validation or test split. Do not report training
set metrics as if they measured generalization.

| Metric | Value | Notes |
| --- | --- | --- |
| mAP50 | `<value>` | `<dataset split>` |
| mAP50-95 | `<value>` | `<dataset split>` |
| Precision | `<value>` | `<confidence / IoU settings>` |
| Recall | `<value>` | `<confidence / IoU settings>` |
| Inference speed | `<value>` | `<device and image size>` |

Validation command:

```bash
python val.py --model path/to/best.pt --data dataset.yaml
```

## Platform Notes

Document the environment used to train and evaluate the weights.

- Python: `<version>`
- PyTorch: `<version>`
- Ultralytics: `<version>`
- Operating system: Windows / Linux / macOS
- Device: CPU / Apple Silicon MPS / NVIDIA CUDA
- GPU or chip model: `<hardware>`

## Known Limitations

Describe known failure cases honestly.

- Sensitivity to lighting, scale, camera angle, or surface finish.
- Defects that look similar across classes.
- Small, faint, or heavily occluded defects.
- Domain shift between public datasets and real factory imagery.

## Responsible Release Checklist

- [ ] Dataset source and license are documented.
- [ ] Class order matches `dataset.yaml`.
- [ ] Validation metrics are reported with the dataset split.
- [ ] Weight file is hosted outside git, such as a GitHub Release asset.
- [ ] Model version and training commit are recorded.
- [ ] Known limitations are listed.
- [ ] Users are told to validate locally before production use.
