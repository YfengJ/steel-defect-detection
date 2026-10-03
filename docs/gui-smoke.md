# GUI Smoke Checklist

Use trusted local weights and files you are allowed to process. Do not attach
datasets, checkpoints, or student/private images to test reports.

## Record The Environment

Record OS/version, CPU/GPU, Python, PyTorch, OpenCV, ttkbootstrap, and Tk
versions, plus the command used to launch `python ui.py`. State whether the
test is a GUI interaction or a headless callback test.

| Coverage | Status |
| --- | --- |
| macOS arm64 headless task callbacks | Verified locally; no window opened |
| Linux headless task callbacks | Included in Python 3.10 CI |
| Windows GUI interaction | Pending |
| Linux GUI interaction | Pending |
| macOS GUI interaction and camera permissions | Pending |
| GUI CUDA/MPS selection | Not yet exposed; use CLI `--device` |

## Check Each Tab

| Tab | Successful path | Failure and recovery |
| --- | --- | --- |
| Training | Run 1 epoch on a small local dataset and confirm the gauge stops. | Select a corrupt local `.pt` or enter an invalid epoch value. Confirm failure status, no success toast, and a stopped gauge; retry with valid input. |
| Validation | Validate a trusted checkpoint and confirm metrics appear. | Try an invalid checkpoint or unreadable dataset. Confirm failure status and readable logs; retry. |
| Image | Predict a valid image and confirm result/report rendering. | Try an unreadable image or checkpoint. Confirm failure status; retry. |
| Batch | Predict a local folder and confirm report generation. | Try a folder with unsupported/corrupt images. Confirm the task leaves its running state and reports the actual outcome; retry. |
| Video | Run a short local video, then stop and restart it. | Try an unreadable video or a camera without permission. Confirm failure status and restartability; repeated Start clicks must not create additional video workers. |

Check all five tabs with missing paths as well: they should reject the input
before background work starts. Watch that validation logs continue updating
while the window remains responsive.

## Video CLI Output Check

```bash
python video_predict.py /path/to/trusted.pt /path/to/short.mp4 --output output.mp4
```

Open the resulting file and check that frames play. Repeat with an output
inside a nonexistent directory; confirm an actionable error and exit status
1, then retry with a valid output directory.

## Report Results

Post the environment, tested rows, exact failure messages, and recovery result
in [issue #53](https://github.com/YfengJ/steel-defect-detection/issues/53).
Headless CI checks task state transitions and resource cleanup. It does not
establish cross-platform GUI, CUDA, MPS, or camera compatibility.
