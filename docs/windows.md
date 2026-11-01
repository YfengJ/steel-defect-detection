# Windows

The project runs natively on Windows 10 and 11 (x64), on CPU or an NVIDIA GPU
with CUDA. The desktop GUI, every CLI script and the test suite are covered by
a `windows-latest` CI job.

## Requirements

- Python 3.10 or 3.11 (64-bit) from <https://www.python.org/downloads/windows/>.
  In the installer, tick **Add python.exe to PATH** and keep **tcl/tk and IDLE**
  selected; the GUI needs Tkinter.
- Optional: an NVIDIA GPU with driver 530 or newer for CUDA 12.1 builds of
  PyTorch.
- Git for Windows, or download the repository as a ZIP.

## One-click Setup

From the repository folder, in **Command Prompt** or by double-clicking:

```bat
scripts\windows\setup.bat
```

This creates `.venv`, installs `requirements.txt` and prints whether CUDA is
available. For an NVIDIA GPU, install the CUDA build of PyTorch instead:

```bat
scripts\windows\setup.bat cuda
```

Then start the GUI by double-clicking `start_gui.bat`. It runs `pythonw.exe`, so
no console window stays open. If the window does not appear, run
`start_gui.bat console` to see the startup error.

## Manual Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
# Optional, NVIDIA GPU: install CUDA PyTorch first
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python ui.py
```

If PowerShell refuses to run `activate`, allow local scripts for your user once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt
with `.venv\Scripts\activate.bat`.

## Check CUDA

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A version ending in `+cpu` means the CPU build is installed. Rerun
`scripts\windows\setup.bat cuda`, or install the CUDA wheel above. In the GUI,
the sidebar shows the detected GPU and every page lets you pick `0` (the first
GPU) or `cpu`.

## Train and Predict

```powershell
python train.py --model yolov8n.pt --data dataset.yaml --epochs 50 --batch 16 --device 0
python predict.py --model runs\detect\train_result\weights\best.pt --source path\to\image.jpg --device 0
python video_predict.py best.pt 0          # camera index 0
```

Paths may contain spaces or Chinese characters. Quote them in the shell.

## What the GUI Does Differently on Windows

- **High-DPI displays:** the app declares DPI awareness, so text stays sharp at
  125–200 % scaling, and the window is sized to fit the screen and taskbar.
- **No console windows:** training, validation and prediction run as child
  processes without flashing console windows, even when started from
  `pythonw.exe`.
- **Stop ends the whole process tree:** **停止** and closing the window use
  `taskkill /T`, so DataLoader worker processes do not linger and keep GPU
  memory.
- **Cameras:** camera indices open through DirectShow, which starts much faster
  than the default Media Foundation backend. The app falls back to the default
  backend if DirectShow cannot open the camera.
- **Fonts:** the UI uses Microsoft YaHei UI and Cascadia Mono or Consolas.
  Charts use Microsoft YaHei or SimHei.

## Common Problems

| Symptom | Fix |
| --- | --- |
| `RuntimeError: DataLoader worker ... exited unexpectedly` or a `freeze_support` error | Pass `--workers 0` to `train.py`. Lower `--batch` if memory is tight. |
| `CUDA out of memory` | Lower `--batch` (8 or 4) or `--imgsz`. Close other GPU programs. |
| `OSError: [WinError 126]` or `1114` when importing torch | Install the [Microsoft Visual C++ Redistributable (x64)](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist), then reboot. |
| `ModuleNotFoundError: No module named 'tkinter'` | Rerun the Python installer, choose **Modify**, and select **tcl/tk and IDLE**. |
| Camera cannot be opened | Allow desktop apps to use the camera under **Settings → Privacy & security → Camera**, close other apps that use it, or try index `1`. |
| Garbled Chinese in a console | Use Windows Terminal, or run `chcp 65001` first. The GUI log is not affected. |
