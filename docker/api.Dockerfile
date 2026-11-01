FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

# System deps: tesseract-ocr + poppler-utils for scanned-PDF OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch from the PyTorch CPU wheel index keeps the image ~1 GB smaller
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        --no-cache-dir -r requirements.txt

# Copy the code the API imports
COPY common/        ./common/
COPY backend/       ./backend/
COPY calibration/   ./calibration/
COPY training/      ./training/

# Model + thresholds
COPY models/ ./models/
COPY experiments/calibration/thresholds.json ./experiments/calibration/thresholds.json

ENV DETECTOR_MODE=hf \
    MODEL_CKPT=/app/models/e3_lora_r8_lr5e-4 \
    THRESHOLDS_PATH=/app/experiments/calibration/thresholds.json \
    CORS_ORIGINS=http://localhost:8501,http://ui:8501 \
    MAX_UPLOAD_MB=10 \
    MAX_TEXT_CHARS=50000 \
    MIN_TEXT_WORDS=20

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0)"

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
