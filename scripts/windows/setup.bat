@echo off
rem Create .venv and install the project's dependencies on Windows.
rem   scripts\windows\setup.bat          CPU-only PyTorch
rem   scripts\windows\setup.bat cuda     PyTorch 2.5.1 built for CUDA 12.1 (NVIDIA driver 530 or newer)
setlocal
cd /d "%~dp0..\.."

if exist ".venv\Scripts\python.exe" (
    echo Using existing .venv
) else (
    echo Creating .venv ...
    py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Could not create .venv. Install Python 3.10 or 3.11 from https://www.python.org/downloads/windows/
    echo and tick "Add python.exe to PATH" and "tcl/tk and IDLE" in the installer.
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1

if /i "%~1"=="cuda" (
    echo Installing CUDA 12.1 PyTorch ...
    ".venv\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 || exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
".venv\Scripts\python.exe" -c "import torch, tkinter, ttkbootstrap; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" || exit /b 1

echo.
echo Setup finished. Start the GUI with start_gui.bat
endlocal
