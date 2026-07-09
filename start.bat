@echo off
setlocal
cd /d "%~dp0"

echo.
echo Starting Video Subtitle OCR...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo Start failed with exit code %EXIT_CODE%.
  echo Keep this window open and check the error message above.
) else (
  echo Service stopped.
)
echo.
pause
exit /b %EXIT_CODE%
