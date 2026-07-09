#!/usr/bin/env bash
set -euo pipefail

PROFILE="recommended"
SKIP_TESSERACT_SYSTEM_INSTALL=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile|-p)
      PROFILE="${2:-}"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --skip-tesseract-system-install)
      SKIP_TESSERACT_SYSTEM_INSTALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/install.sh --profile recommended

Profiles:
  recommended  Install PaddleOCR CPU, the default OCR engine.
  full         Install PaddleOCR, OpenVINO, ONNXRuntime, EasyOCR, and Tesseract support.
  core         Install the web app and processing basics without OCR engines.
  openvino     Install RapidOCR OpenVINO backend.
  onnxruntime  Install RapidOCR ONNXRuntime backend.
  easyocr      Install EasyOCR only.
  tesseract    Install pytesseract and try to install the system Tesseract binary.
  dev          Install lightweight dependencies for tests and CI.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "$PROFILE" in
  recommended|full|core|openvino|onnxruntime|easyocr|tesseract|dev) ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON=".venv/bin/python"

step() {
  printf '\n==> %s\n' "$1"
}

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

system_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "Python was not found. Install Python 3.10-3.12 and add it to PATH." >&2
  exit 1
}

ensure_venv() {
  if [ -x "$VENV_PYTHON" ]; then
    step "Using existing virtual environment"
    return
  fi

  step "Creating virtual environment"
  py="$(system_python)"
  run_cmd "$py" -m venv .venv
}

install_requirements() {
  file="$1"
  step "Installing $file"
  run_cmd "$VENV_PYTHON" -m pip install -U pip
  run_cmd "$VENV_PYTHON" -m pip install -r "$file"
}

sudo_if_needed() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_tesseract_executable() {
  if [ "$SKIP_TESSERACT_SYSTEM_INSTALL" -eq 1 ]; then
    echo "Skipping system Tesseract installation."
    return
  fi

  if command -v tesseract >/dev/null 2>&1; then
    echo "Tesseract executable is already available in PATH."
    return
  fi

  os_name="$(uname -s)"
  case "$os_name" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        step "Installing Tesseract OCR executable with Homebrew"
        run_cmd brew install tesseract tesseract-lang
      else
        echo "Homebrew was not found. Install Tesseract manually: brew install tesseract tesseract-lang" >&2
      fi
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        step "Installing Tesseract OCR executable with apt"
        run_cmd sudo_if_needed apt-get update
        run_cmd sudo_if_needed apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
      elif command -v dnf >/dev/null 2>&1; then
        step "Installing Tesseract OCR executable with dnf"
        run_cmd sudo_if_needed dnf install -y tesseract tesseract-langpack-chi_sim
      elif command -v pacman >/dev/null 2>&1; then
        step "Installing Tesseract OCR executable with pacman"
        run_cmd sudo_if_needed pacman -Sy --needed tesseract tesseract-data-chi_sim
      elif command -v zypper >/dev/null 2>&1; then
        step "Installing Tesseract OCR executable with zypper"
        run_cmd sudo_if_needed zypper install -y tesseract-ocr tesseract-ocr-traineddata-chinese-simplified
      else
        echo "No supported package manager was found. Install Tesseract OCR manually and rerun this profile if needed." >&2
      fi
      ;;
    *)
      echo "Unsupported OS for automatic Tesseract installation: $os_name" >&2
      ;;
  esac
}

set_install_marker() {
  name="$1"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] write .venv/.runtime-installed = $name"
    return
  fi
  printf '%s\n' "$name" > ".venv/.runtime-installed"
}

ensure_venv

case "$PROFILE" in
  recommended)
    install_requirements requirements.txt
    set_install_marker recommended
    ;;
  full)
    install_requirements requirements.txt
    install_requirements requirements-openvino.txt
    install_requirements requirements-onnxruntime.txt
    install_requirements requirements-easyocr.txt
    install_requirements requirements-tesseract.txt
    install_tesseract_executable
    set_install_marker full
    ;;
  core)
    install_requirements requirements-core.txt
    ;;
  openvino)
    install_requirements requirements-openvino.txt
    ;;
  onnxruntime)
    install_requirements requirements-onnxruntime.txt
    ;;
  easyocr)
    install_requirements requirements-easyocr.txt
    ;;
  tesseract)
    install_requirements requirements-tesseract.txt
    install_tesseract_executable
    ;;
  dev)
    install_requirements requirements-dev.txt
    ;;
esac

step "Done"
echo "Start the app with ./start.sh."
