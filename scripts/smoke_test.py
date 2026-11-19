"""
Build-time / post-deploy smoke test.

Runs the real OCR pipeline on the bundled sample ``ktp1.jpg`` with exactly
the same engine settings the API uses, so that:
  * the PP-OCRv5 models are downloaded into the image layer (no cold-start
    download on the first request), and
  * a broken dependency combo (e.g. paddleocr pulling PP-OCRv6 models that
    paddlepaddle 3.0.0 cannot load) fails the BUILD instead of the deploy.

Usage:  python scripts/smoke_test.py [image]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ktp_extractor import KTPExtractor  # noqa: E402


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "ktp1.jpg")
    cpu_threads = int(os.getenv("KTP_CPU_THREADS", "2"))
    lang = os.getenv("KTP_OCR_LANG", "id")

    t0 = time.perf_counter()
    extractor = KTPExtractor(lang=lang, cpu_threads=cpu_threads)
    t1 = time.perf_counter()
    record = extractor.extract(image)
    t2 = time.perf_counter()

    print(f"model load: {t1 - t0:.1f}s, inference: {t2 - t1:.1f}s")
    print(f"nik={record.nik.value!r} nama={record.nama.value!r} decision={record.decision}")

    expected = ("nik", "nama", "alamat", "rt_rw", "kewarganegaraan")
    missing = [f for f in expected if not getattr(record, f).found]
    if missing:
        print(f"SMOKE TEST FAILED: fields not extracted from sample: {missing}", file=sys.stderr)
        return 1
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
