FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

RUN mkdir -p $HOME/app
WORKDIR $HOME/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download model (tanpa bergantung pada kode proyek, ter-cache sebagai layer)
# Parameter HARUS sama dengan yang dipakai KTPExtractor di runtime
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='id')"

COPY --chown=user:user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]