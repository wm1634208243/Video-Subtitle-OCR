param(
    [ValidateSet("recommended", "full", "core", "openvino", "onnxruntime", "easyocr", "tesseract", "dev")]
    [string]$Profile = "recommended",
    [switch]$SkipTesseractSystemInstall,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

Set-Location $Root

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Invoke-CommandLine {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    $display = "$FilePath $($Arguments -join ' ')"
    if ($DryRun) {
        Write-Host "[dry-run] $display"
        return
    }

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $display"
    }
}

function Get-SystemPython {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw "Python was not found. Install Python 3.10-3.12 and add it to PATH."
}

function Ensure-Venv {
    if (Test-Path $VenvPython) {
        Write-Step "Using existing virtual environment"
        return
    }

    Write-Step "Creating virtual environment"
    $systemPython = Get-SystemPython
    if ((Split-Path -Leaf $systemPython) -ieq "py.exe") {
        Invoke-CommandLine $systemPython @("-3", "-m", "venv", ".venv")
    } else {
        Invoke-CommandLine $systemPython @("-m", "venv", ".venv")
    }
}

function Install-Requirements {
    param([string]$FileName)

    Write-Step "Installing $FileName"
    Invoke-CommandLine $VenvPython @("-m", "pip", "install", "-U", "pip")
    Invoke-CommandLine $VenvPython @("-m", "pip", "install", "-r", $FileName)
}

function Install-TesseractExecutable {
    if ($SkipTesseractSystemInstall) {
        Write-Host "Skipping system Tesseract installation."
        return
    }

    if (Get-Command tesseract -ErrorAction SilentlyContinue) {
        Write-Host "Tesseract executable is already available in PATH."
        return
    }

    if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
        Write-Warning "Install the Tesseract executable with your OS package manager, then rerun this profile if needed."
        return
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Warning "winget was not found. Install Tesseract OCR manually, then make sure tesseract.exe is in PATH."
        return
    }

    Write-Step "Installing Tesseract OCR executable with winget"
    Invoke-CommandLine $winget.Source @("install", "--id", "UB-Mannheim.TesseractOCR", "-e", "--source", "winget")
    Write-Host "If Chinese OCR is needed, make sure chi_sim traineddata is installed with Tesseract."
}

function Set-InstallMarker {
    param([string]$Name)

    if ($DryRun) {
        Write-Host "[dry-run] write .venv\.runtime-installed = $Name"
        return
    }

    $Name | Out-File -Encoding ascii ".venv\.runtime-installed"
}

Ensure-Venv

switch ($Profile) {
    "recommended" {
        Install-Requirements "requirements.txt"
        Set-InstallMarker "recommended"
    }
    "full" {
        Install-Requirements "requirements.txt"
        Install-Requirements "requirements-openvino.txt"
        Install-Requirements "requirements-onnxruntime.txt"
        Install-Requirements "requirements-easyocr.txt"
        Install-Requirements "requirements-tesseract.txt"
        Install-TesseractExecutable
        Set-InstallMarker "full"
    }
    "core" {
        Install-Requirements "requirements-core.txt"
    }
    "openvino" {
        Install-Requirements "requirements-openvino.txt"
    }
    "onnxruntime" {
        Install-Requirements "requirements-onnxruntime.txt"
    }
    "easyocr" {
        Install-Requirements "requirements-easyocr.txt"
    }
    "tesseract" {
        Install-Requirements "requirements-tesseract.txt"
        Install-TesseractExecutable
    }
    "dev" {
        Install-Requirements "requirements-dev.txt"
    }
}

Write-Step "Done"
Write-Host "Start the app with start.bat or start.ps1."
