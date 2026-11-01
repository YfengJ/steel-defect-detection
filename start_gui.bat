@echo off
rem Launch the desktop GUI without a console window.
rem   start_gui.bat            uses .venv (or venv) created by scripts\windows\setup.bat
rem   start_gui.bat console    keeps a console open to show startup errors
setlocal
cd /d "%~dp0"

set "PYDIR="
if exist ".venv\Scripts\python.exe" set "PYDIR=.venv\Scripts"
if not defined PYDIR if exist "venv\Scripts\python.exe" set "PYDIR=venv\Scripts"
if not defined PYDIR (
    echo No virtual environment found. Run scripts\windows\setup.bat first.
    pause
    exit /b 1
)

if /i "%~1"=="console" (
    "%PYDIR%\python.exe" ui.py
    pause
) else (
    start "" "%PYDIR%\pythonw.exe" ui.py
)
endlocal
