"""
Command-line entry point.

Usage:
    python -m ktp_extractor.cli path/to/ktp.jpg
    python -m ktp_extractor.cli path/to/ktp.jpg --pretty
    python -m ktp_extractor.cli path/to/ktp.jpg --lang en --cpu-threads 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

# Suppress noisy C++ extension warnings and bypass remote model host check for speed
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "3")

from .extractor import KTPExtractor
from .models import KTPExtractionError


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured fields from a KTP image.")
    parser.add_argument("image", help="Path to a KTP image file (jpg/png).")
    parser.add_argument("--lang", default="id", help="PaddleOCR language model (default: id).")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON output.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        try:
            import paddlex.utils.logging as pdx_logging
            pdx_logging.setup_logging("ERROR")
        except Exception:
            pass
        logging.getLogger("paddlex").setLevel(logging.ERROR)
        logging.getLogger("ppocr").setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        extractor = KTPExtractor(lang=args.lang, cpu_threads=args.cpu_threads)
        record = extractor.extract(args.image)
    except KTPExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2 if args.pretty else None))

    if record.missing_fields:
        print(f"\nMissing fields: {', '.join(record.missing_fields)}", file=sys.stderr)
    if record.low_confidence_fields:
        print(f"Low-confidence fields (review recommended): "
              f"{', '.join(record.low_confidence_fields)}", file=sys.stderr)
    if record.nik_valid is False:
        print(f"WARNING: NIK failed structural validation: "
              f"{record.nik_validation_notes}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
