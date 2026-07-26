# Release Notes

## v0.1.1 - Maintenance and Cross-Platform Baseline

This release turns the repository into a more reproducible open source baseline
without a large refactor of the YOLOv8 training or inference implementation.

### Current Functionality

- YOLOv8 training, validation, image and directory prediction, and video inference.
- A `ttkbootstrap` desktop GUI for common learning and demonstration workflows.
- VOC XML to YOLO label conversion for six NEU-DET style defect classes.
- CPU and NVIDIA CUDA workflows, plus Apple Silicon MPS selection for training.
- A vendored Ultralytics 8.0.182 runtime for reproducible project behavior.

### Maintenance and Reliability

- Restored valid source, Markdown, and YAML formatting with real line endings.
- Added early local-path validation to the CLI and GUI for weights, datasets,
  images, directories, and video sources.
- Preserved subprocess exit codes in the GUI and improved failure status handling.
- Ensured video capture and output resources are released after success or failure.
- Hardened VOC conversion for missing fields, invalid image sizes, and out-of-range boxes.
- Added 36 focused tests covering CLI parsing and failure paths, CPU inference,
  path validation, conversion, subprocess behavior, video cleanup, and maintenance contracts.
- Expanded CI on Python 3.10 to run Ruff, pytest, repository hygiene checks, and
  `python -m compileall .` on pushes and pull requests.
- Added artifact, large-file, line-ending, GitHub YAML, and maintained-document
  link checks without committing datasets or model weights.
- Clarified vendored-runtime packaging and updated project citation metadata,
  contribution guidance, security guidance, Dependabot policy, and bilingual
  README status information.

### Supported Platforms

- Windows and Linux with CPU or NVIDIA CUDA, subject to the installed PyTorch build.
- macOS with CPU.
- Apple Silicon macOS with PyTorch MPS when available.

The supported Python range is 3.10 to 3.12, with 3.10 as the CI baseline. The
maintenance verification also passed on an Apple Silicon Mac with Python 3.12,
PyTorch 2.5.1, CPU inference, and a no-weight MPS inference smoke check.

### Known Limitations

- NEU-DET data, pretrained/trained weights, and generated `runs/` outputs are not included.
- Real training and accuracy validation require a user-provided dataset and trusted weights.
- MPS training is not automated in GitHub Actions and may fall back to CPU for unsupported operations.
- The GUI remains a learning/demo interface and still needs broader manual platform testing.
- The vendored Ultralytics 8.0.182 checkpoint loader requires the current
  PyTorch/TorchVision compatibility bounds; major upgrades need explicit testing.
- PyTorch checkpoints may contain serialized Python objects. Load only weights
  from trusted sources.

### Next Plans

- Publish a reproducible sample inference workflow without bundling data or weights.
- Add a model-card template for any separately published steel-defect weights.
- Exercise a real Apple Silicon MPS training and validation workflow.
- Continue improving GUI feedback and add platform-focused smoke coverage.
