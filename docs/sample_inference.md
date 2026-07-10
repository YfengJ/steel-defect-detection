# Sample Inference With Local Files

This workflow verifies image inference without adding a dataset, model weight,
or generated output to the repository. You provide both the weight and image
from trusted local sources.

## 1. Create The Environment

```bash
git clone https://github.com/YfengJ/steel-defect-detection.git
cd steel-defect-detection

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Python 3.10 is the CI baseline. Python 3.11 is also used for the documented
Apple Silicon smoke check.

## 2. Prepare Trusted Local Inputs

Choose paths outside git for:

- A trusted YOLOv8 `.pt` weight, such as an official `yolov8n.pt` download or
  your own trained `best.pt`.
- An image you are allowed to process.

Do not load untrusted PyTorch checkpoint files. A `.pt` file can contain
serialized Python objects, so its source matters.

Example local layout:

```text
~/steel-defect-local/
├── weights/
│   └── best.pt
└── images/
    └── sample.jpg
```

These files remain outside the repository. Paths under `weights/`, `datasets/`,
and `runs/` are ignored if you choose to keep local artifacts inside the clone.

## 3. Run CPU Inference

Start with CPU because it is the most portable device:

```bash
python predict.py \
  --model ~/steel-defect-local/weights/best.pt \
  --source ~/steel-defect-local/images/sample.jpg \
  --device cpu \
  --project runs/detect \
  --name sample_cpu
```

The rendered image is written to `runs/detect/sample_cpu/`. YOLO labels with
confidence values are written under `runs/detect/sample_cpu/labels/`.

## 4. Use MPS Or CUDA

On Apple Silicon, first confirm MPS availability as described in
[macos.md](macos.md), then replace the device:

```bash
python predict.py \
  --model ~/steel-defect-local/weights/best.pt \
  --source ~/steel-defect-local/images/sample.jpg \
  --device mps \
  --project runs/detect \
  --name sample_mps
```

On an NVIDIA CUDA environment, use a CUDA index:

```bash
python predict.py \
  --model /path/to/best.pt \
  --source /path/to/sample.jpg \
  --device 0 \
  --project runs/detect \
  --name sample_cuda
```

If MPS or CUDA fails, rerun the same command with `--device cpu` and include the
full error output in a bug report.

## 5. Verify And Clean Up

Check the command contract at any time without loading a model:

```bash
python predict.py --help
```

Generated outputs are disposable and ignored by git:

```bash
rm -rf runs/detect/sample_cpu runs/detect/sample_mps
```

For dataset-based training, use [dataset.md](dataset.md). For common runtime
errors, use [troubleshooting.md](troubleshooting.md).
