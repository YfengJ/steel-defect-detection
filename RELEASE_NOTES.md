# Release Notes

## v0.1.1 - Documentation and Maintenance Polish

This release focuses on making the repository easier to read, clone, check, and
maintain before broader feature work. It does not introduce a large refactor of
the core YOLOv8 training or inference behavior.

### Current Functionality

- YOLOv8-based steel surface defect detection workflow.
- Command-line entrypoints for training, validation, image/folder prediction,
  and video prediction.
- Desktop GUI demo built with `ttkbootstrap`.
- Default NEU-DET style class mapping for six steel surface defect categories.
- Local-only handling for datasets, model weights, and `runs/` outputs.

### Supported Platforms

- Windows with CPU or NVIDIA CUDA.
- Linux with CPU or NVIDIA CUDA.
- macOS with CPU.
- Apple Silicon macOS with MPS when the installed PyTorch build and local
  environment support it.
- Python 3.10 is the recommended baseline for local setup and CI.

### Maintenance Changes

- Restored real line endings for README, YAML, and project Python entrypoints so
  Markdown, YAML, and Python parsers can read them reliably.
- Cleaned English and Chinese README Markdown so GitHub renders headings,
  badges, tables, lists, and code blocks correctly.
- Fixed the CI workflow YAML and kept the initial check intentionally simple:
  install dependencies and run `python -m compileall .`.
- Tightened Dependabot queue limits to keep dependency review manageable.
- Added clearer command-line help text for `train.py`, `predict.py`, `val.py`,
  and `video_predict.py`.
- Delayed heavy YOLO/OpenCV imports in CLI entrypoints so `--help` works before
  a full runtime environment is installed.
- Applied low-risk dependency floor updates for `requests` and `tqdm`.

### Known Limitations

- The repository does not include datasets, trained model weights, or generated
  `runs/` outputs.
- CI currently performs repository hygiene checks, lightweight CLI smoke tests,
  and syntax compilation; full training, validation, and GUI smoke tests are
  still manual.
- Apple Silicon MPS behavior depends on the local PyTorch build, macOS version,
  and available operators.
- Major dependency upgrades for PyTorch/TorchVision, OpenCV, NumPy, SciPy,
  Pandas, Pillow, Seaborn, and Ultralytics need compatibility testing before
  merging.
- The GUI is useful for demos and learning workflows, but it is not yet a
  production inspection system.

### Next Plans

- Expand lightweight CLI smoke tests beyond help output and argument parsing.
- Document a small sample-data workflow for first-time users.
- Add a model card template for trained steel defect detection weights.
- Improve GUI error messages for missing datasets, missing weights, and unsupported devices.
- Continue reviewing dependency updates conservatively across Windows, Linux, and macOS.
