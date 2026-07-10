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

## v0.1.2 - Runtime And Contributor Confidence

- Add testable local path validation and actionable GUI setup errors.
- Add a sample inference workflow that uses trusted user-provided files.
- Verify one Apple Silicon MPS tensor operation and image inference outside git.
- Expose prediction device selection through `predict.py --device`.
- Protect the vendored checkpoint loader with tested PyTorch/TorchVision bounds.
- Run focused Ruff checks, unit tests, CLI smoke tests, repository hygiene, and
  compileall in CI.
- Add project-specific contribution guidance, a code of conduct, CODEOWNERS,
  and a pull request template.

## v0.2.0 - Reproducible Training Workflow

- Provide a small sample `dataset.yaml` and clearer dataset layout checks.
- Add a tiny runtime fixture or scripted public download for repeatable CPU
  inference without committing assets.
- Document recommended training presets for CPU, Apple Silicon MPS, and CUDA.
- Add focused tests for subprocess completion and output directory reporting.
- Improve generated output naming so repeated experiments are easier to compare.

## v0.3.0 - Usability and Release Quality

- Add a minimal demo dataset or scripted sample download that does not commit
  dataset files.
- Use the model card template for any released trained weights.
- Add screenshots or short demo media generated from public sample images.
- Add a release checklist covering CI, docs, dataset instructions, and weight
  publishing.
- Add a GUI device selector and clearer runtime failure state after subprocesses
  exit unsuccessfully.
- Decide whether to keep vendoring Ultralytics or migrate to a separately pinned
  package after compatibility tests and packaging review.
