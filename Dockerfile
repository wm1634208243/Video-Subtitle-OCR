FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_CACHE_HOME=/app/data/models/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app
ARG INSTALL_EASYOCR=false
ARG INSTALL_OPENVINO=false
ARG INSTALL_ONNXRUNTIME=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements.txt requirements-openvino.txt requirements-onnxruntime.txt requirements-easyocr.txt requirements-tesseract.txt ./
RUN python -m pip install -U pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r requirements-tesseract.txt \
    && if [ "$INSTALL_OPENVINO" = "true" ]; then python -m pip install -r requirements-openvino.txt; fi \
    && if [ "$INSTALL_ONNXRUNTIME" = "true" ]; then python -m pip install -r requirements-onnxruntime.txt; fi \
    && if [ "$INSTALL_EASYOCR" = "true" ]; then python -m pip install -r requirements-easyocr.txt; fi

COPY app ./app
COPY scripts ./scripts
COPY start.sh README.md LICENSE pyproject.toml ./

RUN mkdir -p /app/data/models/paddlex /app/data/jobs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
