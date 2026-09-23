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
from .validators import (
    clean_digit_string, normalize_rt_rw, normalize_whitespace,
    strip_label_punctuation, validate_nik,
)

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

        # Pass 3: Positional fallback for Nama. If Nama was not detected via label
        # (common in faded cards), the row immediately following NIK is strictly the Name.
        if not record.nama.found:
            nik_row_idx = None
            for idx, text in enumerate(texts):
                if record.nik.found and record.nik.value and record.nik.value in clean_digit_string(text):
                    nik_row_idx = idx
                    break
                elif "NIK" in text.upper():
                    nik_row_idx = idx
                    break
            if nik_row_idx is not None and nik_row_idx + 1 < len(texts):
                cand_text = texts[nik_row_idx + 1]
                if not scan_row_for_labels(cand_text, FIELD_LABELS):
                    clean_name = strip_label_punctuation(cand_text)
                    if clean_name and not any(ch.isdigit() for ch in clean_name):
                        record.nama.value = clean_name
                        record.nama.found = True
                        record.nama.confidence = row_confidence(rows[nik_row_idx + 1])
                        record.nama.source_row_text = cand_text

        # Pass 4: Value-driven fallback for Jenis Kelamin (e.g. lone "PEREMPUAN" or "LAKI-LAKI")
        if not record.jenis_kelamin.found:
            for idx, text in enumerate(texts):
                upper = text.upper()
                if "PEREMPUAN" in upper or "FEMALE" in upper:
                    record.jenis_kelamin.value = "PEREMPUAN"
                    record.jenis_kelamin.found = True
                    record.jenis_kelamin.confidence = row_confidence(rows[idx])
                    record.jenis_kelamin.source_row_text = text
                    break
                elif "LAKI" in upper or "MALE" in upper:
                    record.jenis_kelamin.value = "LAKI-LAKI"
                    record.jenis_kelamin.found = True
                    record.jenis_kelamin.confidence = row_confidence(rows[idx])
                    record.jenis_kelamin.source_row_text = text
                    break

        # Pass 5: Value-driven fallback for Agama (lone religion name on faded label)
        if not record.agama.found:
            RELIGIONS = ("ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "BUDHA", "KONGHUCU")
            for idx, text in enumerate(texts):
                upper = text.upper()
                for rel in RELIGIONS:
                    words = upper.split()
                    if rel in words:
                        record.agama.value = rel if rel != "BUDHA" else "BUDDHA"
                        record.agama.found = True
                        record.agama.confidence = row_confidence(rows[idx])
                        record.agama.source_row_text = text
                        break
                if record.agama.found:
                    break

        # Pass 6: Value-driven fallback for Status Perkawinan
        if not record.status_perkawinan.found:
            for idx, text in enumerate(texts):
                upper = text.upper()
                if "BELUM KAWIN" in upper or "BELUM KAWN" in upper or "UNMARRIED" in upper:
                    record.status_perkawinan.value = "BELUM KAWIN"
                    record.status_perkawinan.found = True
                    record.status_perkawinan.confidence = row_confidence(rows[idx])
                    record.status_perkawinan.source_row_text = text
                    break
                elif "KAWIN" in upper or "MARRIED" in upper:
                    if "BELUM" not in upper:
                        record.status_perkawinan.value = "KAWIN"
                        record.status_perkawinan.found = True
                        record.status_perkawinan.confidence = row_confidence(rows[idx])
                        record.status_perkawinan.source_row_text = text
                        break

        # Pass 7: Positional fallback for Alamat (row immediately preceding RT/RW if untagged)
        if not record.alamat.found and record.rt_rw.found:
            for idx, text in enumerate(texts):
                if record.rt_rw.value and record.rt_rw.value in text:
                    if idx > 0 and not scan_row_for_labels(texts[idx - 1], FIELD_LABELS):
                        cand = strip_label_punctuation(texts[idx - 1])
                        if cand and not any(cand.upper().startswith(kw) for kw in ("NIK", "PROVINSI", "KABUPATEN", "KOTA", "PEREMPUAN", "LAKI")):
                            record.alamat.value = cand
                            record.alamat.found = True
                            record.alamat.confidence = row_confidence(rows[idx - 1])
                            record.alamat.source_row_text = texts[idx - 1]
                    break

        # Pass 8: Positional fallback for Kecamatan (row between Kel/Desa and Agama if untagged)
        if not record.kecamatan.found and record.kel_desa.found:
            for idx, text in enumerate(texts):
                if record.kel_desa.value and record.kel_desa.value in text:
                    if idx + 1 < len(texts) and not scan_row_for_labels(texts[idx + 1], FIELD_LABELS):
                        cand = strip_label_punctuation(texts[idx + 1])
                        if cand and not any(rel in cand.upper() for rel in ("ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA")):
                            record.kecamatan.value = cand
                            record.kecamatan.found = True
                            record.kecamatan.confidence = row_confidence(rows[idx + 1])
                            record.kecamatan.source_row_text = texts[idx + 1]
                    break

        # Pass 9: Value-driven fallback for Kewarganegaraan
        if not record.kewarganegaraan.found:
            for idx, text in enumerate(texts):
                upper = text.upper().strip()
                words = upper.split()
                if "WNI" in words or "WNE" in words or upper in ("WNI", "WNE", "WN", "WNA") or "WN" in words:
                    record.kewarganegaraan.value = "WNI" if "WNA" not in upper else "WNA"
                    record.kewarganegaraan.found = True
                    record.kewarganegaraan.confidence = row_confidence(rows[idx])
                    record.kewarganegaraan.source_row_text = text
                    break

        # Pass 10: Value-driven fallback for Berlaku Hingga
        if not record.berlaku_hingga.found:
            import re
            date_pat = re.compile(r"(\d{2}[-\s/]\d{2}[-\s/]\d{4}|\d{4}[-\s/]\d{4})")
            for idx in range(len(texts) - 1, max(-1, len(texts) - 4), -1):
                upper = texts[idx].upper()
                if "SEUMUR HIDUP" in upper:
                    record.berlaku_hingga.value = "SEUMUR HIDUP"
                    record.berlaku_hingga.found = True
                    record.berlaku_hingga.confidence = row_confidence(rows[idx])
                    record.berlaku_hingga.source_row_text = texts[idx]
                    break
                m = date_pat.search(upper)
                if m:
                    record.berlaku_hingga.value = m.group(1).replace(" ", "-")
                    record.berlaku_hingga.found = True
                    record.berlaku_hingga.confidence = row_confidence(rows[idx])
                    record.berlaku_hingga.source_row_text = texts[idx]
                    break

        self._postprocess(record)
        self._score(record)
        return record

    @staticmethod
    def _extract_header_field(rows, texts, limit, keyword_or_tuple, target: FieldResult) -> None:
        keywords = (keyword_or_tuple,) if isinstance(keyword_or_tuple, str) else keyword_or_tuple
        for i in range(limit):
            upper = texts[i].upper()
            if any(kw in upper for kw in keywords):
                # Check if both PROVINSI and KOTA/KABUPATEN are merged into the same row
                if keyword_or_tuple == PROVINCE_KEYWORD:
                    for reg_kw in REGENCY_KEYWORDS:
                        if reg_kw in upper:
                            reg_pos = upper.find(reg_kw)
                            val = texts[i][:reg_pos].replace(PROVINCE_KEYWORD, "").strip()
                            target.value = normalize_whitespace(val) or None
                            target.found = bool(target.value)
                            target.confidence = row_confidence(rows[i])
                            target.source_row_text = texts[i]
                            return

                value = texts[i]
                for kw in keywords:
                    if kw in upper:
                        # If regency keyword is matched on a line with PROVINSI, take the part after reg_kw
                        pos = upper.find(kw)
                        value = texts[i][pos + len(kw):].strip()
                        break
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
            if "PEREMPUAN" in v or "FEMALE" in v or v.startswith("P") or v.startswith("F"):
                record.jenis_kelamin.value = "PEREMPUAN"
            elif "LAKI" in v or "MALE" in v or v.startswith("L") or v.startswith("M"):
                record.jenis_kelamin.value = "LAKI-LAKI"

        if record.status_perkawinan.found and record.status_perkawinan.value:
            v = record.status_perkawinan.value.upper()
            if "BELUM" in v or "SINGLE" in v or "UNMARRIED" in v:
                record.status_perkawinan.value = "BELUM KAWIN"
            elif "KAWIN" in v or "MARRIED" in v:
                record.status_perkawinan.value = "KAWIN"
            elif "CERAI HIDUP" in v or "DIVORCED" in v:
                record.status_perkawinan.value = "CERAI HIDUP"
            elif "CERAI MATI" in v or "WIDOWED" in v:
                record.status_perkawinan.value = "CERAI MATI"

        if record.gol_darah.found and record.gol_darah.value:
            v = record.gol_darah.value.upper().replace(":", "").replace(".", "").strip()
            if v in ("A", "B", "AB", "O"):
                record.gol_darah.value = v
            elif not v or v == "-":
                record.gol_darah.value = "-"

        if record.kewarganegaraan.found and record.kewarganegaraan.value:
            v = record.kewarganegaraan.value.upper()
            if "WNI" in v:
                record.kewarganegaraan.value = "WNI"

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
