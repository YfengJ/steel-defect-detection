## Summary

Describe the problem and the smallest change that solves it.

## Related Issue

Closes #

## Verification

- [ ] `python -m ruff check train.py predict.py val.py video_predict.py path_validation.py scripts tests`
- [ ] `python -m pytest tests/test_repository_hygiene.py tests/test_path_validation.py tests/test_dependency_contracts.py -q`
- [ ] `python tests/test_cli_smoke.py`
- [ ] `python -m compileall .`

## Compatibility And Artifacts

- [ ] I did not add datasets, model weights, `runs/` output, caches, or large generated files.
- [ ] I documented any user-visible CLI, GUI, dataset, or platform change.
- [ ] I did not upgrade PyTorch, TorchVision, OpenCV, NumPy, or the vendored Ultralytics snapshot without CPU/MPS/CUDA compatibility evidence.
- [ ] I updated both `README.md` and `README.zh-CN.md` when shared project status changed.

## Notes

Add screenshots, logs, or compatibility details that help reviewers validate the change. Do not include private data or untrusted model files.
