param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VenvPython = ".venv\Scripts\python.exe"
$InstallScript = ".\scripts\install.ps1"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Test-PortAvailable {
    param([int]$Candidate)

    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $listener = [System.Net.Sockets.TcpListener]::new($address, $Candidate)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Get-AvailablePort {
    param([int]$Preferred)

    for ($candidate = $Preferred; $candidate -le ($Preferred + 20); $candidate++) {
        if (Test-PortAvailable $candidate) {
            return $candidate
        }
    }

    throw "No available local port found from $Preferred to $($Preferred + 20)."
}

function Ensure-InstallerExists {
    if (-not (Test-Path $InstallScript)) {
        throw "Installer script was not found: $InstallScript"
    }
}

function Test-RuntimeReady {
    if (-not (Test-Path $VenvPython)) {
        return $false
    }

    & $VenvPython -c "import fastapi, uvicorn, cv2, numpy, imageio_ffmpeg; from app.ocr_engines import get_engine_status; status = get_engine_status('paddle'); raise SystemExit(0 if status.available else 1)"
    return $LASTEXITCODE -eq 0
}

function Ensure-Runtime {
    if ((Test-Path ".venv\.runtime-installed") -and (Test-Path $VenvPython)) {
        return
    }

    if (Test-RuntimeReady) {
        Write-Step "Runtime already installed"
        "recommended" | Out-File -Encoding ascii ".venv\.runtime-installed"
        return
    }

    Write-Step "Installing recommended runtime"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $InstallScript -Profile recommended
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Close any running Video Subtitle OCR windows, then run install.bat and choose Recommended."
    }
}

Ensure-InstallerExists
$ActualPort = Get-AvailablePort $Port

if ($DryRun) {
    Write-Step "Dry run"
    Write-Host "Preferred port: $Port"
    Write-Host "Selected port:  $ActualPort"
    Write-Host "Python:         $VenvPython"
    Write-Host "Install marker: .venv\.runtime-installed"
    return
}

Ensure-Runtime

if (-not (Test-Path $VenvPython)) {
    throw "Python virtual environment was not created correctly: $VenvPython"
}

if ($ActualPort -ne $Port) {
    Write-Host "Port $Port is busy. Using port $ActualPort instead."
}

$Url = "http://127.0.0.1:$ActualPort"
Write-Step "Starting web service"
Write-Host "Open $Url in your browser."

if (-not $NoBrowser) {
    Start-Process $Url
}

$env:PYTHONUTF8 = "1"
$PaddleCache = Join-Path (Get-Location) "data\models\paddlex"
New-Item -ItemType Directory -Force -Path $PaddleCache | Out-Null
$env:PADDLE_PDX_CACHE_HOME = $PaddleCache
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port $ActualPort
