@echo off
title EchoPulseNet Desktop Installer
color 0B
cd /d "%~dp0"

echo ====================================================================
echo   ECHOPULSENET: MARINE SONAR INTELLIGENCE PLATFORM (x64)
echo   Autonomous Deep Ocean Sonar Target Classification Engine
echo ====================================================================
echo.
echo [*] Installing EchoPulseNet Desktop Icon to Windows Desktop...

set "SCRIPT_DIR=%~dp0"
set "ICON_FILE=%SCRIPT_DIR%assets\app_icon.ico"
set "TARGET_EXE=%SCRIPT_DIR%Launch_EchoPulseNet_Standalone.bat"

if exist "%SCRIPT_DIR%dist_standalone\EchoPulseNet_Standalone\EchoPulseNet_Standalone.exe" (
    set "TARGET_EXE=%SCRIPT_DIR%dist_standalone\EchoPulseNet_Standalone\EchoPulseNet_Standalone.exe"
)

powershell -NoProfile -Command ^
  "$desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop); " ^
  "$shortcutPath = Join-Path $desktop 'EchoPulseNet Marine Sonar.lnk'; " ^
  "$ws = New-Object -ComObject WScript.Shell; " ^
  "$s = $ws.CreateShortcut($shortcutPath); " ^
  "$s.TargetPath = '%TARGET_EXE%'; " ^
  "$s.WorkingDirectory = '%SCRIPT_DIR%'; " ^
  "$s.IconLocation = '%ICON_FILE%,0'; " ^
  "$s.Description = 'EchoPulseNet Autonomous Marine Sonar Intelligence Platform'; " ^
  "$s.Save(); " ^
  "Write-Host '[SUCCESS] Desktop shortcut installed at: ' $shortcutPath -ForegroundColor Green"

echo.
echo [*] You can now launch EchoPulseNet directly from your Desktop!
echo.
pause
