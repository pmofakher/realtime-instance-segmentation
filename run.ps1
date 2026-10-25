# PowerShell launcher for Mask R-CNN GPU Webcam Detection
$ScriptDir = $PSScriptRoot
Set-Location -LiteralPath $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "[!] Error: Virtual environment python not found at $PythonExe" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "      Launching Mask R-CNN Webcam Detection (GPU)       " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

& $PythonExe main.py @args
