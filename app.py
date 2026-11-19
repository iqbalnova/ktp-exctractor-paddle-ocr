"""
FastAPI application for Indonesian KTP OCR extraction using PaddleOCR.

Menjalankan server:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # atau: python app.py

Environment variables (opsional):
    KTP_MAX_UPLOAD_MB    Batas ukuran gambar dalam MB (default 8)
    KTP_CPU_THREADS      Jumlah thread CPU untuk OCR (default 2)
    KTP_OCR_LANG         Bahasa OCR (default "id")
    KTP_CORS_ORIGINS     Daftar origin dipisah koma (default "*" / semua origin)
    KTP_DISABLE_DOCS     "1" untuk mematikan /docs dan /redoc
    KTP_HOST / PORT      Host & port saat dijalankan via `python app.py`
    KTP_RELOAD           "1" untuk auto-reload saat development
"""
from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ktp_extractor import KTPExtractor, KTPExtractionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ktp_api")


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("KTP_MAX_UPLOAD_MB", "8")) * 1024 * 1024
DOCS_ENABLED = os.getenv("KTP_DISABLE_DOCS", "0") != "1"
_raw_cors = os.getenv("KTP_CORS_ORIGINS", "*").strip()
CORS_ORIGINS = [o.strip() for o in _raw_cors.split(",") if o.strip()] or ["*"]

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# PaddleOCR berat di CPU: proses satu gambar pada satu waktu.
OCR_LOCK = threading.Lock()


def _ensure_extractor(app: FastAPI) -> KTPExtractor:
    """Lazy/thread-safe initializer so Uvicorn can bind port immediately on
    startup while the OCR engine warms up in the background."""
    extractor = getattr(app.state, "extractor", None)
    if extractor is not None:
        return extractor
    cpu_threads = int(os.getenv("KTP_CPU_THREADS", "2"))
    lang = os.getenv("KTP_OCR_LANG", "id")
    logger.info("Initializing KTPExtractor with PaddleOCR PP-OCRv5 (lang=%s, threads=%d)...", lang, cpu_threads)
    extractor = KTPExtractor(lang=lang, cpu_threads=cpu_threads)
    app.state.extractor = extractor
    logger.info("KTPExtractor initialized and ready to serve requests.")
    return extractor


def _background_warmup(app: FastAPI) -> None:
    try:
        with OCR_LOCK:
            _ensure_extractor(app)
    except Exception:
        logger.exception("Background OCR warmup failed; will retry on first request.")


# ---------------------------------------------------------------------------
# App & lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bind port immediately and warm up PaddleOCR in a background thread so
    container deployment probes never time out waiting for model load."""
    app.state.extractor = None
    threading.Thread(target=_background_warmup, args=(app,), daemon=True).start()
    yield


app = FastAPI(
    title="KTP Extractor API",
    description="Indonesian Identity Card (KTP) text extraction and field normalization API powered by PaddleOCR.",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Base64ExtractRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64 encoded image string (with or without 'data:image/...;base64,' prefix).",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _looks_like_image(data: bytes) -> bool:
    """Cek magic bytes sederhana agar file sembarangan tidak diproses."""
    return (
        data.startswith(b"\xff\xd8\xff")  # JPEG
        or data.startswith(b"\x89PNG\r\n\x1a\n")  # PNG
        or data.startswith(b"BM")  # BMP
        or data.startswith((b"II*\x00", b"MM\x00*"))  # TIFF
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")  # WEBP
    )


def _write_temp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Gagal menghapus file sementara.")


def _process_extraction(file_path: str) -> dict:
    """Run extraction and return structured response with execution timing."""
    start_time = time.perf_counter()
    try:
        with OCR_LOCK:
            extractor = _ensure_extractor(app)
            record = extractor.extract(file_path)
    except KTPExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"KTP extraction failed: {exc}",
        ) from exc
    except Exception:
        logger.exception("Unexpected error during extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info("Extraction selesai dalam %.0f ms", elapsed_ms)
    return {
        "success": True,
        "elapsed_ms": elapsed_ms,
        "data": record.to_dict(),
    }


async def _extract_bytes(data: bytes, suffix: str) -> dict:
    """Validasi bytes gambar, tulis ke file sementara, jalankan OCR di threadpool."""
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File kosong.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File terlalu besar (maks {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    if not _looks_like_image(data):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File bukan gambar yang didukung.")

    tmp_path = _write_temp(data, suffix)
    try:
        return await run_in_threadpool(_process_extraction, tmp_path)
    finally:
        _safe_remove(tmp_path)


# ---------------------------------------------------------------------------
# Endpoint umum
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect to Swagger UI documentation (atau /health jika docs dimatikan)."""
    return RedirectResponse(url="/docs" if DOCS_ENABLED else "/health")


@app.get("/info", tags=["General"])
def service_info():
    """Service metadata and API information."""
    return {
        "service": "KTP Extractor API",
        "version": "1.1.0",
        "status": "online",
        "docs_url": "/docs" if DOCS_ENABLED else None,
        "endpoints": {
            "extract_file": "POST /api/v1/extract (multipart/form-data)",
            "extract_base64": "POST /api/v1/extract-base64 (application/json)",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint for monitoring."""
    is_ready = getattr(app.state, "extractor", None) is not None
    return {
        "status": "healthy",
        "ocr_engine_ready": is_ready,
    }


# ---------------------------------------------------------------------------
# Endpoint ekstraksi (publik, tanpa API key)
# ---------------------------------------------------------------------------
@app.post(
    "/api/v1/extract",
    tags=["Extraction"],
    summary="Extract KTP fields from uploaded image file",
)
async def extract_from_file(
    response: Response,
    file: UploadFile = File(..., description="KTP image file (JPG, PNG, WEBP, etc.)"),
):
    """Upload a KTP image file as multipart/form-data to extract structured fields."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    result = await _extract_bytes(content, ext or ".jpg")
    response.headers["Cache-Control"] = "no-store"
    return result


@app.post(
    "/api/v1/extract-base64",
    tags=["Extraction"],
    summary="Extract KTP fields from base64 encoded image",
)
async def extract_from_base64(payload: Base64ExtractRequest, response: Response):
    """Submit a base64 encoded KTP image string in a JSON payload."""
    raw_b64 = payload.image_base64.strip()
    if raw_b64.startswith("data:") and "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    if len(raw_b64) > (MAX_UPLOAD_BYTES * 4) // 3 + 8:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File terlalu besar (maks {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    try:
        image_bytes = base64.b64decode(raw_b64)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 payload.",
        )

    result = await _extract_bytes(image_bytes, ".jpg")
    response.headers["Cache-Control"] = "no-store"
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("KTP_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("KTP_RELOAD", "0") == "1",
    )
