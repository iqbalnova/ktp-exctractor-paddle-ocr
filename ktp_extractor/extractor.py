"""
KTPExtractor: end-to-end KTP field extraction.

Pipeline:
    image -> OCR (PP-OCRv5, CPU) -> text boxes -> rows (reading order)
          -> label:value matching per row -> field cleanup & validation
          -> KTPRecord
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .fields import (
    FIELD_LABELS, PROVINCE_KEYWORD, REGENCY_KEYWORDS,
    scan_row_for_labels,
)
from .layout import group_into_rows, row_text, row_confidence
from .models import FieldResult, KTPRecord, TextBox
from .ocr_engine import PaddleOCREngine
from .validators import normalize_rt_rw, normalize_whitespace, validate_nik

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.75

# Rows before the NIK row are assumed to be the province/regency header.
# This caps how far down we'll look for that header so a stray early
# mis-detection elsewhere on the card can't be mistaken for it.
_HEADER_SEARCH_LIMIT_ROWS = 4


class KTPExtractor:
    def __init__(self, engine: Optional[PaddleOCREngine] = None, **engine_kwargs) -> None:
        """
        Args:
            engine: inject a pre-built PaddleOCREngine (or a test double).
                If omitted, one is constructed from engine_kwargs, which are
                passed straight through to PaddleOCREngine.__init__.
        """
        self._engine = engine or PaddleOCREngine(**engine_kwargs)

    def extract(self, image_path: str) -> KTPRecord:
        boxes = self._engine.run(image_path)
        return self.extract_from_boxes(boxes)

    def extract_from_boxes(self, boxes: List[TextBox]) -> KTPRecord:
        """Split out from ``extract`` so the field-matching logic can be
        unit-tested with synthetic boxes, without running real OCR.
        """
        rows = group_into_rows(boxes)
        texts = [row_text(r) for r in rows]
        record = KTPRecord(raw_text_lines=texts)

        header_limit = min(_HEADER_SEARCH_LIMIT_ROWS, len(rows))
        self._extract_header_field(rows, texts, header_limit, PROVINCE_KEYWORD, record.provinsi)
        self._extract_header_field(rows, texts, header_limit, REGENCY_KEYWORDS, record.kabupaten_kota)

        # Pass 1: scan every row for one or more label:value segments. This
        # handles both the common one-field-per-row layout and the case of
        # two short fields printed side by side (Jenis Kelamin / Gol. Darah).
        for row_idx, text in enumerate(texts):
            for label, value in scan_row_for_labels(text, FIELD_LABELS):
                target: FieldResult = getattr(record, label.key)
                if target.found:
                    continue  # first occurrence wins; don't overwrite
                if not value:
                    continue  # handled in pass 2 below
                target.value = normalize_whitespace(value)
                target.found = True
                target.confidence = row_confidence(rows[row_idx])
                target.source_row_text = text

        # Pass 2: a label that matched but left an empty value on its own
        # row usually means the card's tilt pushed the value onto a
        # visually separate row. Fall back to the next row's full text,
        # but only if that row doesn't itself contain another field's label
        # (otherwise we'd steal that field's value instead).
        for row_idx, text in enumerate(texts):
            segments = scan_row_for_labels(text, FIELD_LABELS)
            if len(segments) != 1:
                continue
            label, value = segments[0]
            target = getattr(record, label.key)
            if target.found or value:
                continue
            next_idx = row_idx + 1
            if next_idx >= len(texts):
                continue
            if scan_row_for_labels(texts[next_idx], FIELD_LABELS):
                continue  # next row is itself a label row; don't steal it
            target.value = normalize_whitespace(texts[next_idx])
            target.found = True
            target.confidence = row_confidence(rows[next_idx])
            target.source_row_text = texts[next_idx]

        self._postprocess(record)
        self._score(record)
        return record

    @staticmethod
    def _extract_header_field(rows, texts, limit, keyword_or_tuple, target: FieldResult) -> None:
        keywords = (keyword_or_tuple,) if isinstance(keyword_or_tuple, str) else keyword_or_tuple
        for i in range(limit):
            upper = texts[i].upper()
            if any(kw in upper for kw in keywords):
                value = texts[i]
                for kw in keywords:
                    value = value.replace(kw, "").strip()
                    value = value.replace(kw.title(), "").strip()
                target.value = normalize_whitespace(value) or None
                target.found = bool(target.value)
                target.confidence = row_confidence(rows[i])
                target.source_row_text = texts[i]
                return

    @staticmethod
    def _postprocess(record: KTPRecord) -> None:
        if record.rt_rw.found and record.rt_rw.value:
            record.rt_rw.value = normalize_rt_rw(record.rt_rw.value)

        if record.jenis_kelamin.found and record.jenis_kelamin.value:
            v = record.jenis_kelamin.value.upper()
            if "PEREMPUAN" in v or v.startswith("P"):
                record.jenis_kelamin.value = "PEREMPUAN"
            elif "LAKI" in v or v.startswith("L"):
                record.jenis_kelamin.value = "LAKI-LAKI"

        if record.nik.found and record.nik.value:
            is_valid, cleaned, notes = validate_nik(record.nik.value)
            record.nik.value = cleaned
            record.nik_valid = is_valid
            record.nik_validation_notes = notes
            if not is_valid:
                logger.info("NIK failed validation: %s", notes)
        else:
            record.nik_valid = False
            record.nik_validation_notes = ["NIK field was not detected at all."]

    @staticmethod
    def _score(record: KTPRecord) -> None:
        checked_fields = [
            ("provinsi", record.provinsi), ("kabupaten_kota", record.kabupaten_kota),
            ("nik", record.nik), ("nama", record.nama),
            ("tempat_tgl_lahir", record.tempat_tgl_lahir),
            ("jenis_kelamin", record.jenis_kelamin), ("gol_darah", record.gol_darah),
            ("alamat", record.alamat), ("rt_rw", record.rt_rw),
            ("kel_desa", record.kel_desa), ("kecamatan", record.kecamatan),
            ("agama", record.agama), ("status_perkawinan", record.status_perkawinan),
            ("pekerjaan", record.pekerjaan), ("kewarganegaraan", record.kewarganegaraan),
            ("berlaku_hingga", record.berlaku_hingga),
        ]

        found = [(name, f) for name, f in checked_fields if f.found]
        missing = [name for name, f in checked_fields if not f.found]
        low_conf = [name for name, f in found if f.confidence < LOW_CONFIDENCE_THRESHOLD]

        record.missing_fields = missing
        record.low_confidence_fields = low_conf
        if record.nik_valid is False:
            record.low_confidence_fields = list(set(low_conf + ["nik"]))

        record.overall_confidence = (
            sum(f.confidence for _, f in found) / len(found) if found else 0.0
        )
