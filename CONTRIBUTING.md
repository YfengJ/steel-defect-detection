# Contributing

Thanks for helping improve this learning-oriented steel surface defect detection
project. Small, reproducible contributions are especially welcome.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## Good Contributions

- Fix a reproducible training, validation, inference, or GUI problem.
- Improve Windows, Linux, macOS, CPU, MPS, or CUDA setup guidance.
- Add focused tests that do not require committed datasets or model weights.
- Clarify dataset preparation, annotation conversion, or troubleshooting.
- Improve accessibility and actionable error messages in the desktop GUI.

Large changes to the vendored `ultralytics/` source tree should start with an
issue explaining the need, compatibility impact, and upstream relationship.

## Before Opening An Issue

Read [SUPPORT.md](SUPPORT.md) and search existing issues. Bug reports should
include:

- Operating system, Python version, PyTorch version, and device.
- The exact command or GUI action.
- The full error output.
- A dataset layout description without uploading the dataset.
- The model weight source without attaching large or untrusted checkpoint files.

Use GitHub private vulnerability reporting for security issues as described in
[SECURITY.md](SECURITY.md).

## Development Setup

```bash
git clone https://github.com/YfengJ/steel-defect-detection.git
cd steel-defect-detection

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The repository vendors Ultralytics 8.0.182 under `ultralytics/`. Do not install
another Ultralytics version on top of the local source when reproducing project
behavior.

## Required Checks

Run these before opening a pull request:

```bash
python -m ruff check \
  train.py predict.py val.py video_predict.py path_validation.py scripts tests

python -m pytest \
  tests/test_repository_hygiene.py \
  tests/test_path_validation.py \
  tests/test_dependency_contracts.py \
  -q

python tests/test_cli_smoke.py
python scripts/check_repository_hygiene.py
python -m compileall .
```

Python 3.10 is the CI baseline. Platform-specific changes should also include
the local environment and manual command used for verification.

## Dependency Policy

PyTorch, TorchVision, OpenCV, NumPy, SciPy, and the vendored Ultralytics source
are compatibility-sensitive. Do not upgrade them only to reach the newest
version. A dependency pull request should explain:

- Why the update is needed.
- Which CPU, MPS, or CUDA paths were tested.
- Whether model loading, one inference command, and CLI checks still pass.
- Any known platform or Python-version limitation.

The current PyTorch and TorchVision upper bounds protect compatibility with the
vendored Ultralytics 8.0.182 checkpoint loader.

## Documentation

`README.md` is the English default homepage and `README.zh-CN.md` is its Chinese
counterpart. Keep their shared status, links, commands, and feature descriptions
in sync. Project documentation under `docs/` may remain English unless a change
specifically adds a translated page.

## Files That Must Stay Out Of Git

Do not commit:

- Datasets or private images.
- `.pt`, `.pth`, ONNX, engine, or other model artifacts.
- `runs/`, `wandb/`, caches, virtual environments, or generated reports.
- Secrets, tokens, credentials, or proprietary configuration.

The repository hygiene check enforces the main artifact and large-file rules.

## Pull Requests

Keep each pull request focused and link a real issue when one exists. Complete
the pull request checklist, describe verification evidence, and update user
documentation for public behavior changes. Maintainers may ask for narrower
scope or platform evidence before merging compatibility-sensitive updates.

By contributing, you agree that your contribution is licensed under the
[AGPL-3.0 license](LICENSE).
