# Release Notes

## v0.1.2 - Runtime And Contributor Confidence

This patch release turns the documentation baseline into a tested local
workflow. It completes the three public maintenance issues that followed
v0.1.1, strengthens dependency compatibility, and adds missing contributor
governance without refactoring the vendored YOLO core.

### Highlights

- Added dependency-light validation for model, dataset, image, folder, and
  video paths selected in the GUI.
- Added `predict.py --device` for explicit CPU, Apple Silicon MPS, CUDA, or CUDA
  index selection.
- Added a sample inference guide that uses trusted user-provided weights and
  images without committing either one.
- Verified Apple Silicon MPS with a tensor operation and one temporary image
  inference using Python 3.11.15, PyTorch 2.5.1, TorchVision 0.20.1, and the
  vendored Ultralytics 8.0.182 source.
- Added project-specific contribution guidance, Contributor Covenant 3.0,
  CODEOWNERS, a pull request checklist, and a direct private security-reporting
  path.

### Compatibility Fixes

- Removed the redundant PyPI Ultralytics installation. Runtime imports now
  unambiguously use the repository's vendored 8.0.182 source.
- Capped PyTorch below 2.6 and TorchVision below 0.21. Ultralytics 8.0.182 uses
  the earlier `torch.load` behavior for trusted YOLO checkpoints and cannot load
  standard weights with the PyTorch 2.6+ default.
- Added dependency contract tests so the vendored/runtime version boundary
  cannot be removed accidentally.
- Fixed the repository hygiene checker so an unstaged YAML deletion does not
  crash local checks.
- Removed an inactive, outdated README translation workflow configuration.

### CI And Tests

- CI still uses Python 3.10 as the compatibility baseline.
- CI installs runtime and development dependencies, then runs focused Ruff
  checks, 20 dependency/path/repository unit tests, CLI smoke tests, repository
  hygiene, and `python -m compileall .`.
- Workflow permissions are read-only, superseded runs are cancelled, and the
  job has a 20-minute timeout.
- Local Apple Silicon verification additionally covered GUI import, MPS device
  discovery, an MPS tensor operation, trusted weight loading, and image
  inference to a temporary output directory.

### Supported Platforms

- Windows: CPU and NVIDIA CUDA workflows remain supported by the existing
  scripts and documentation.
- Linux: CPU and NVIDIA CUDA workflows remain supported by the existing scripts
  and documentation.
- macOS: CPU is the fallback path; Apple Silicon MPS completed the v0.1.2 local
  smoke check.

Windows and CUDA hardware were not available during this release's local
verification. Compatibility-sensitive dependency updates remain manual.

### Upgrade Notes

Create a fresh virtual environment when upgrading from v0.1.1 so the removed
PyPI Ultralytics package and new Torch bounds are resolved cleanly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Datasets, model weights, and generated `runs/` output are still intentionally
excluded. Only load `.pt` or `.pth` checkpoints from sources you trust.

### Known Limitations

- CI does not train a model, launch the desktop GUI, or run GPU inference.
- The MPS result validates one environment and one image workflow, not every
  model architecture or PyTorch operator.
- GUI subprocess completion currently needs clearer distinction between a
  successful exit and a failed command after launch.
- The project carries an older Ultralytics source snapshot; moving to a newer
  version requires an explicit packaging and compatibility decision.
- This is a learning and demonstration project, not a production inspection or
  safety system.

### Next Plans

- Add a repeatable CPU runtime fixture or scripted public sample download.
- Add training presets for CPU, MPS, and CUDA with small smoke configurations.
- Make GUI completion status reflect subprocess exit codes.
- Improve experiment output naming and release checks.
- Review whether the project should continue vendoring Ultralytics or move to a
  separately pinned package.

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
- Added a model card template for future trained steel defect detection weights.

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
- Improve GUI error messages for missing datasets, missing weights, and unsupported devices.
- Continue reviewing dependency updates conservatively across Windows, Linux, and macOS.
