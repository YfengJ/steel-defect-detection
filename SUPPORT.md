# Support

Thanks for trying this project. The quickest way to get useful help is to open a GitHub issue with enough context for someone else to reproduce the problem.

## Before Opening an Issue

- Check [README.md](README.md), [docs/macos.md](docs/macos.md), [docs/dataset.md](docs/dataset.md), and [docs/troubleshooting.md](docs/troubleshooting.md).
- Search existing issues for the same error message.
- Make sure your dataset and model weights are stored locally and are not committed to the repository.

## Include This Information

- Operating system and CPU/GPU, for example macOS on Apple M2, Windows with CUDA, or Linux CPU.
- Python version: `python --version`
- PyTorch device check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
PY
```

- The exact command you ran.
- The full traceback or console output.
- Whether the problem happens in `train.py`, `val.py`, `predict.py`, `video_predict.py`, or `ui.py`.
- Dataset layout, without uploading the dataset itself.
- Model weight source, without uploading large `.pt` or `.pth` files unless a maintainer explicitly asks for a small public reproduction.
