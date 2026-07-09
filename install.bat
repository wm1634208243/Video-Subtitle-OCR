@echo off
setlocal
cd /d "%~dp0"

echo.
echo Video Subtitle OCR installer
echo.
echo 1. Recommended: PaddleOCR CPU (best default)
echo 2. EasyOCR only (optional, downloads PyTorch)
echo 3. Tesseract only (optional, installs Python wrapper and checks system OCR)
echo 4. Full: PaddleOCR + EasyOCR + Tesseract
echo 5. Developer: lightweight deps for tests/CI
echo 6. Core only: web app without OCR engines
echo.
set /p CHOICE=Choose an option [1-6]:

if "%CHOICE%"=="1" set PROFILE=recommended
if "%CHOICE%"=="2" set PROFILE=easyocr
if "%CHOICE%"=="3" set PROFILE=tesseract
if "%CHOICE%"=="4" set PROFILE=full
if "%CHOICE%"=="5" set PROFILE=dev
if "%CHOICE%"=="6" set PROFILE=core

if "%PROFILE%"=="" (
  echo Invalid option.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -Profile %PROFILE%
exit /b %ERRORLEVEL%
