# EchoPulseNet - Single Command Native Launcher for Windows PowerShell
param(
    [switch]$NoBrowser = $false,
    [string]$BackendPort = "8000",
    [string]$FrontendPort = "3000"
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  ECHOPULSENET: LOCAL-FIRST MARINE SONAR INTELLIGENCE LAUNCHER   " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Verify conda environment dgpu-core
Write-Host "[*] Checking Python and CUDA acceleration..." -ForegroundColor Yellow
$pyVer = python -c "import sys, torch; print(f'Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"})')"
Write-Host "    $pyVer" -ForegroundColor Green

# 2. Verify directories
Write-Host "[*] Ensuring local data and models directories exist..." -ForegroundColor Yellow
@("data/raw", "data/downloaded", "data/validated", "data/unified", "models_checkpoints", "reports", "uploads", "cache") | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}

# 3. Launch Backend
Write-Host "[*] Launching FastAPI Backend on http://127.0.0.1:$BackendPort..." -ForegroundColor Yellow
$backendProcess = Start-Process python -ArgumentList "-m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort" -WorkingDirectory "$PSScriptRoot\..\backend" -PassThru -WindowStyle Hidden
Write-Host "    FastAPI Backend started (PID: $($backendProcess.Id))" -ForegroundColor Green

# 4. Launch Frontend
Write-Host "[*] Launching Vite Frontend on http://localhost:$FrontendPort..." -ForegroundColor Yellow
$frontendProcess = Start-Process npm -ArgumentList "run dev" -WorkingDirectory "$PSScriptRoot\.." -PassThru -WindowStyle Hidden
Write-Host "    Vite Frontend started (PID: $($frontendProcess.Id))" -ForegroundColor Green

# 5. Wait for backend health check
Write-Host "[*] Waiting for services to initialize..." -ForegroundColor Yellow
$maxAttempts = 15
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Seconds 1
    $attempt++
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/v1/system/telemetry" -TimeoutSec 2 -ErrorAction Stop
        if ($response.backendStatus -eq "ONLINE") {
            $healthy = $true
            break
        }
    } catch {
        # continue waiting
    }
}

if ($healthy) {
    Write-Host "[SUCCESS] EchoPulseNet services are ONLINE and HEALTHY!" -ForegroundColor Green
    Write-Host "  - Web Dashboard:  http://localhost:$FrontendPort" -ForegroundColor Cyan
    Write-Host "  - API Swagger UI: http://127.0.0.1:$BackendPort/api/v1/docs" -ForegroundColor Cyan
    Write-Host "  - Mode:           100% LOCAL-FIRST (OFFLINE CAPABLE)" -ForegroundColor Green

    if (-not $NoBrowser) {
        Start-Process "http://localhost:$FrontendPort"
    }
} else {
    Write-Host "[WARNING] Backend took longer than expected to respond, but processes are active." -ForegroundColor Yellow
}

Write-Host "`nPress Ctrl+C to stop all local services or close this window." -ForegroundColor Gray
