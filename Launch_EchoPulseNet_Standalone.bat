# Portable One-Click Zero-Dependency Standalone SaaS Launch Script for Windows
$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  ECHOPULSENET: STANDALONE ZERO-CONFIG SAAS PLATFORM LAUNCHER     " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOT

# 1. Check Python runtime
$PYTHON_BIN = "python"
if (Test-Path "C:\Users\elang\miniconda3\envs\dgpu-core\python.exe") {
    $PYTHON_BIN = "C:\Users\elang\miniconda3\envs\dgpu-core\python.exe"
}

Write-Host "[*] Launching Standalone EchoPulseNet SaaS Embedded Node..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON_BIN -ArgumentList "desktop_app.py" -NoNewWindow
