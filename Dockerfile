# syntax=docker/dockerfile:1
# KTP Extractor API -- CPU-only PaddleOCR (PP-OCRv5) + FastAPI.
# Works on Hugging Face Spaces (Docker SDK, app_port 7860) and any platform
# that injects $PORT (Cloud Run, Railway, Render, Fly.io, ...).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime libs for OpenCV (libGL, glib) and Paddle's OpenMP (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with UID 1000 (required by Hugging Face Spaces).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OMP_NUM_THREADS=2 \
    KTP_CPU_THREADS=2 \
    KTP_OCR_LANG=id \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    GLOG_minloglevel=3 \
    PORT=7860

WORKDIR $HOME/app

# Dependencies first -> cached layer as long as requirements.txt is unchanged.
COPY --chown=user:user requirements.txt .
RUN pip install --user --upgrade pip \
    && pip install --user -r requirements.txt

COPY --chown=user:user . .

# Download PP-OCRv5 models with the EXACT runtime engine config and run one
# real inference on the bundled sample. Fails the build (not the deploy) if
# models/deps are broken, and removes the cold-start model download.
RUN python scripts/smoke_test.py ktp1.jpg

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'), timeout=4)" || exit 1

# Single worker: the OCR model is loaded once per process and inference is
# serialized (OCR_LOCK); scale horizontally with more replicas instead.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
