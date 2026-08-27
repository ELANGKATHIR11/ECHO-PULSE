@echo off
title EchoPulseNet Marine Sonar Intelligence Platform
cd /d "%~dp0"

echo =================================================================
echo   ECHOPULSENET: NATIVE EDGE DESKTOP LAUNCHER
echo   (PostgreSQL + PyTorch RTX 5060 + Unified Single-Server + Desktop UI)
echo =================================================================

REM 1. Ensure PostgreSQL is started
echo [*] Checking PostgreSQL 18 Service...
net start postgresql-x64-18 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] PostgreSQL service is active.
) else (
    echo [*] Starting PostgreSQL directly via pg_ctl...
    "F:\Program Files\PostgreSQL\18\bin\pg_ctl.exe" start -D "F:\Program Files\PostgreSQL\18\data" -w >nul 2>&1
)

REM 2. Launch Desktop Application via Electron
echo [*] Launching Native Desktop App...
npx electron .

