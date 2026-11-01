# GUI Smoke Checklist

Use trusted local weights and files you are allowed to process. Do not attach
datasets, checkpoints, or student/private images to test reports.

## Record The Environment

Record OS/version, CPU/GPU, Python, PyTorch, OpenCV, ttkbootstrap, and Tk
versions, plus the command used to launch `python ui.py`. State whether the
test is a GUI interaction or a headless callback test.

| Coverage | Status |
| --- | --- |
| macOS arm64 headless task callbacks | Verified locally |
| Linux headless task callbacks and GUI logic | Included in Python 3.10 CI |
| Linux (Xvfb), ttkbootstrap 2.2, CPU | Verified for v0.2.0. A scripted run of image, batch, video and validation tasks against the baseline checkpoint produced the README screenshots |
| macOS arm64, Tk 9.0, ttkbootstrap 2.2, MPS | Verified for v0.2.0 with the same scripted image, batch, video and validation run |
| macOS arm64, ttkbootstrap 1.14 | Verified for v0.2.0: every page builds, and theme switching and the training view work |
| Manual mouse-driven click-through | Pending |
| Windows GUI interaction | Pending |
| macOS camera permissions | Pending |
| GUI CUDA selection | Device picker exposed; untested on CUDA hardware |

## Check Each Tab

| Tab | Successful path | Failure and recovery |
| --- | --- | --- |
| Training | Run 1–2 epochs on a small local dataset. Confirm that the epoch counter, progress bar and curves update after each epoch, and that **打开目录** opens the new run. | Select a corrupt local `.pt` or enter an invalid epoch value. Confirm the input is rejected or the run shows a failure status with the log opened, and no success notification. Start again, press **停止**, and confirm the status reads “训练已停止”. |
| Validation | Validate a trusted checkpoint and confirm the metric tiles and per-class table fill in. | Try an invalid checkpoint or unreadable dataset. Confirm failure status and readable logs; retry. |
| Image | Predict a valid image. Confirm the result view, the detection table, and the 原图/检测结果 switch. | Try an unreadable image or checkpoint. Confirm failure status; retry. |
| Batch | Predict a local folder. Confirm the tiles, both charts and the per-image table, and that double-clicking a row opens its result. | Try a folder with unsupported/corrupt images. Confirm the task leaves its running state and reports the actual outcome; retry. |
| Video | Run a short local video, then stop and restart it. | Try an unreadable video or a camera without permission. Confirm failure status and restartability; repeated Start clicks must not create additional video workers. |

Toggle the light/dark theme on every page and resize the window; images,
charts and tables should repaint. Check all five pages with missing paths as well: they should reject the input
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
