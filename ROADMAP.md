# Roadmap

This roadmap keeps the project focused on a small, runnable YOLOv8 steel surface
defect detection workflow. Dates are intentionally not fixed because this is a
learning-oriented open source project maintained in spare time.

## v0.1.0 - Open Source Maintenance Baseline

- Document the project scope as a YOLOv8 steel surface defect detection project.
- Add macOS and Apple Silicon setup guidance.
- Clarify that datasets, trained weights, and generated outputs are not stored
  in the repository.
- Add basic GitHub Actions CI with dependency installation and Python syntax checks.
- Add issue templates for bugs, feature requests, and documentation fixes.
- Add security and support policies.
- Keep training, validation, prediction, and GUI scripts close to the current
  implementation.

## v0.2.0 - Reproducible Training Workflow

- Provide a small sample `dataset.yaml` and clearer dataset layout checks.
- Expand lightweight smoke tests beyond CLI help output to cover argument
  parsing and dataset path validation.
- Document recommended training presets for CPU, Apple Silicon MPS, and CUDA.
- Add example commands for prediction, validation, and batch inference.
- Improve generated output naming so repeated experiments are easier to compare.

## v0.3.0 - Usability and Release Quality

- Add a minimal demo dataset or scripted sample download that does not commit
  dataset files.
- Add model card documentation for any released trained weights.
- Add screenshots or short demo media generated from public sample images.
- Add a release checklist covering CI, docs, dataset instructions, and weight
  publishing.
- Improve GUI error messages for missing models, missing dataset paths, and
  unavailable devices.
