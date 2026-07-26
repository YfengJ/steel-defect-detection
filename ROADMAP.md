# Roadmap

This roadmap keeps the project focused on a small, runnable YOLOv8 steel surface
defect detection workflow. Dates are intentionally not fixed because this is a
learning-oriented open source project maintained in spare time.

## v0.1.0 - Open Source Maintenance Baseline

- Document the project scope and supported learning use cases.
- Add macOS, dataset, troubleshooting, support, and security guidance.
- Exclude datasets, trained weights, and generated outputs from git.
- Add basic GitHub Actions CI, Dependabot, and issue templates.
- Preserve the existing training, validation, prediction, and GUI structure.

## v0.1.1 - Reliability Baseline

- Restore valid source, Markdown, and YAML formatting.
- Add focused CLI, path-validation, conversion, video, subprocess, and CPU runtime tests.
- Improve GUI errors for missing local datasets, weights, images, and video sources.
- Add repository hygiene, local documentation link, and large-artifact checks.
- Verify a no-weight YOLO inference path on CPU and Apple Silicon MPS.
- Align package, citation, release, and contribution metadata with this project.

## v0.2.0 - Reproducible Sample Workflow

- Add a scripted sample inference workflow without committing datasets or weights.
- Document recommended presets for CPU, Apple Silicon MPS, and CUDA.
- Add a small synthetic dataset fixture for training and validation smoke coverage.
- Exercise a real Apple Silicon MPS training and validation workflow.
- Improve generated output naming so repeated experiments are easier to compare.

## v0.3.0 - Usability and Release Quality

- Add a model-card template for separately released trained weights.
- Add a release checklist covering CI, docs, dataset instructions, and artifacts.
- Expand GUI error states and manual cross-platform verification.
- Publish public sample results with clear provenance and evaluation limitations.
