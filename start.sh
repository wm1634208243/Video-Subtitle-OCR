#!/usr/bin/env bash
set -euo pipefail

PORT=8000
NO_BROWSER=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --no-browser)
      NO_BROWSER=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./start.sh [--port 8000] [--no-browser] [--dry-run]

Create the local Python runtime when needed, install the recommended OCR stack,
and start Video Subtitle OCR at http://127.0.0.1:<port>.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")"

VENV_PYTHON=".venv/bin/python"
INSTALL_SCRIPT="./scripts/install.sh"

step() {
  printf '\n==> %s\n' "$1"
}

find_available_port() {
  "$VENV_PYTHON" - "$1" <<'PY'
import socket
import sys

preferred = int(sys.argv[1])
for port in range(preferred, preferred + 21):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit(f"No available local port found from {preferred} to {preferred + 20}.")
PY
}

ensure_installer_exists() {
  if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "Installer script was not found: $INSTALL_SCRIPT" >&2
    exit 1
  fi
}

runtime_ready() {
  if [ ! -x "$VENV_PYTHON" ]; then
    return 1
  fi

  "$VENV_PYTHON" - <<'PY'
import fastapi
import imageio_ffmpeg
import numpy
import uvicorn
import cv2
from app.ocr_engines import get_engine_status

raise SystemExit(0 if get_engine_status("paddle").available else 1)
PY
}

ensure_runtime() {
  if [ -f ".venv/.runtime-installed" ] && [ -x "$VENV_PYTHON" ]; then
    return
  fi

  if runtime_ready; then
    step "Runtime already installed"
    printf 'recommended\n' > ".venv/.runtime-installed"
    return
  fi

  step "Installing recommended runtime"
  bash "$INSTALL_SCRIPT" --profile recommended
}

open_browser() {
  url="$1"
  if [ "$NO_BROWSER" -eq 1 ]; then
    return
  fi

  case "$(uname -s)" in
    Darwin)
      open "$url" >/dev/null 2>&1 || true
      ;;
    Linux)
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 || true
      fi
      ;;
  esac
}

ensure_installer_exists

if [ ! -x "$VENV_PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then
    :
  elif command -v python >/dev/null 2>&1; then
    :
  else
    echo "Python 3.10-3.12 was not found. Install Python first, then rerun this script." >&2
    exit 1
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  step "Dry run"
  echo "Preferred port: $PORT"
  echo "Python:         $VENV_PYTHON"
  echo "Install marker: .venv/.runtime-installed"
  exit 0
fi

ensure_runtime
ACTUAL_PORT="$(find_available_port "$PORT")"

if [ "$ACTUAL_PORT" != "$PORT" ]; then
  echo "Port $PORT is busy. Using port $ACTUAL_PORT instead."
fi

URL="http://127.0.0.1:$ACTUAL_PORT"
step "Starting web service"
echo "Open $URL in your browser."
open_browser "$URL"

export PYTHONUTF8=1
export PADDLE_PDX_CACHE_HOME="$(pwd)/data/models/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
mkdir -p "$PADDLE_PDX_CACHE_HOME"

exec "$VENV_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port "$ACTUAL_PORT"
