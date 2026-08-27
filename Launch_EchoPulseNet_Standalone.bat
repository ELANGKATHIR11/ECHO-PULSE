@echo off
title EchoPulseNet Autonomous Marine Sonar Intelligence Platform
color 0B
cd /d "%~dp0"

echo ==================================================================
echo   ECHOPULSENET: STANDALONE ZERO-CONFIG SAAS PLATFORM (x64)
echo   Autonomous Deep Ocean Sonar Target Classification Engine
echo ==================================================================
echo.

REM 1. Auto-create Desktop Shortcut if not present
powershell -NoProfile -Command ^
  "$desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop); " ^
  "$lnk = Join-Path $desktop 'EchoPulseNet Marine Sonar.lnk'; " ^
  "if (-not (Test-Path $lnk)) { " ^
  "  $ws = New-Object -ComObject WScript.Shell; " ^
  "  $s = $ws.CreateShortcut($lnk); " ^
  "  if (Test-Path '%~dp0dist_standalone\EchoPulseNet_Standalone\EchoPulseNet_Standalone.exe') { " ^
  "    $s.TargetPath = '%~dp0dist_standalone\EchoPulseNet_Standalone\EchoPulseNet_Standalone.exe'; " ^
  "  } else { " ^
  "    $s.TargetPath = '%~dp0Launch_EchoPulseNet_Standalone.bat'; " ^
  "  } " ^
  "  $s.WorkingDirectory = '%~dp0'; " ^
  "  $s.IconLocation = '%~dp0assets\app_icon.ico,0'; " ^
  "  $s.Description = 'EchoPulseNet Autonomous Marine Sonar Intelligence Platform'; " ^
  "  $s.Save(); " ^
  "  Write-Host '[OK] Desktop icon created at: ' $lnk; " ^
  "}"

REM 2. Determine best runtime
if exist "%~dp0dist_standalone\EchoPulseNet_Standalone\EchoPulseNet_Standalone.exe" (
    echo [*] Launching EchoPulseNet Native Standalone Executable...
    start "" "%~dp0dist_standalone\EchoPulseNet_Standalone\EchoPulseNet_Standalone.exe"
    exit /b 0
)

REM 3. Fallback to Python Embedded Desktop Runner
set "PY_EXEC=python"
if exist "C:\Users\elang\miniconda3\envs\dgpu-core\python.exe" (
    set "PY_EXEC=C:\Users\elang\miniconda3\envs\dgpu-core\python.exe"
)

echo [*] Starting EchoPulseNet Embedded Desktop Application...
"%PY_EXEC%" desktop_app.py
