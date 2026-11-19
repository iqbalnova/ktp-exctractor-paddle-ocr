# syntax=docker/dockerfile:1
# KTP Extractor API -- CPU-only PaddleOCR (PP-OCRv5) + FastAPI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime libs for OpenCV (libGL, glib) and Paddle's OpenMP (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with UID 1000 (compatible with Hugging Face Spaces & standard PaaS).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PADDLE_PDX_CACHE_HOME=/home/user/.paddlex \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    KTP_CPU_THREADS=1 \
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

# Pre-download PP-OCRv5 models and verify extraction at build time, then ensure
# cache directory is readable/writable even if runtime overrides UID.
RUN python scripts/smoke_test.py ktp1.jpg \
    && chmod -R a+rwX /home/user/.paddlex

EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
