@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo       Launching Mask R-CNN Webcam Detection (GPU)
echo ========================================================

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Please wait for setup or run setup first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo [!] Process exited with code %ERRORLEVEL%
    pause
)
