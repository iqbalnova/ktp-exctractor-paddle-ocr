"""
KTPExtractor: end-to-end KTP field extraction.

Pipeline:
    image -> OCR (PP-OCRv5, CPU) -> text boxes -> rows (reading order)
          -> label:value matching per row -> field cleanup & validation
          -> KTPRecord
"""
from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional

from . import reference, vocabulary
from .consistency import decide, run_rules

from .fields import (
    FIELD_LABELS, PROVINCE_KEYWORD, REGENCY_KEYWORDS,
    scan_row_for_labels,
)
from .layout import group_into_rows, row_text, row_confidence
from .models import FieldResult, KTPRecord, TextBox, ValidationIssue
from .ocr_engine import PaddleOCREngine
from .validators import (
    clean_digit_string, normalize_rt_rw, normalize_whitespace,
    strip_label_punctuation, validate_nik, clean_pekerjaan,
    clean_kewarganegaraan, normalize_tempat_tgl_lahir,
)

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = float(os.getenv("KTP_LOW_CONFIDENCE_THRESHOLD", "0.75"))

# Minimum similarity for snapping a printed value onto the official
# vocabulary (reference/vocabulary.json).
VOCAB_SNAP_MIN_SCORE = 0.75

# Whole tokens that mark the start of the next KTP row rather than the
# continuation of a two-line Alamat.
_NON_ADDRESS_TOKENS = {"RT", "RW", "RT/RW", "KEL", "DESA", "KEL/DESA", "KECAMATAN", "AGAMA"}
_RT_RW_VALUE_RE = re.compile(r"^\d{1,3}\s*/\s*\d{1,3}$")

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

        # Pass 2b: Multi-line continuation for Alamat
        # If Alamat was found, any unlabeled row immediately following it
        # (before the RT/RW row) is the second line of the address (e.g. 'NO.40' or 'BLOK C2 NO 15').
        if record.alamat.found and record.alamat.value:
            for idx, text in enumerate(texts):
                if text == record.alamat.source_row_text or (record.alamat.source_row_text and text in record.alamat.source_row_text):
                    next_idx = idx + 1
                    if next_idx < len(texts):
                        if not scan_row_for_labels(texts[next_idx], FIELD_LABELS):
                            continuation = strip_label_punctuation(texts[next_idx])
                            tokens = set(continuation.upper().replace(":", " ").split())
                            if (continuation and not (tokens & _NON_ADDRESS_TOKENS)
                                    and not _RT_RW_VALUE_RE.match(continuation)):
                                record.alamat.value = f"{record.alamat.value} {continuation}"
                                record.alamat.source_row_text = f"{record.alamat.source_row_text} {texts[next_idx]}"
                                record.alamat.confidence = (record.alamat.confidence + row_confidence(rows[next_idx])) / 2
                    break

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
        self._vocab_fallback(record.jenis_kelamin, "jenis_kelamin", rows, texts)

        # Pass 5: Value-driven fallback for Agama (lone religion name on faded label)
        self._vocab_fallback(record.agama, "agama", rows, texts)

        # Pass 6: Value-driven fallback for Status Perkawinan
        self._vocab_fallback(record.status_perkawinan, "status_perkawinan", rows, texts)

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
                        if cand and vocabulary.find_in_text(cand, "agama") is None:
                            record.kecamatan.value = cand
                            record.kecamatan.found = True
                            record.kecamatan.confidence = row_confidence(rows[idx + 1])
                            record.kecamatan.source_row_text = texts[idx + 1]
                    break

        # Pass 9: Value-driven fallback for Kewarganegaraan
        self._vocab_fallback(record.kewarganegaraan, "kewarganegaraan", rows, texts)

        # Pass 10: Value-driven fallback for Berlaku Hingga
        if not record.berlaku_hingga.found:
            date_pat = re.compile(r"(\d{2}[-\s/]\d{2}[-\s/]\d{4}|\d{4}[-\s/]\d{4})")
            for idx in range(len(texts) - 1, max(-1, len(texts) - 4), -1):
                upper = texts[idx].upper()
                if vocabulary.find_in_text(upper, "berlaku_hingga"):
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
        record.issues.extend(run_rules(record))
        record.decision = decide(record)
        return record

    @staticmethod
    def _vocab_fallback(target: FieldResult, vocab_name: str, rows, texts) -> None:
        """Label-less fallback: take the first row that contains an official
        vocabulary value as a whole word (e.g. a lone "ISLAM" whose
        "Agama" label faded)."""
        if target.found:
            return
        for idx, text in enumerate(texts):
            canonical = vocabulary.find_in_text(text, vocab_name)
            if canonical:
                target.value = canonical
                target.found = True
                target.confidence = row_confidence(rows[idx])
                target.source_row_text = text
                return

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

        for name in ("jenis_kelamin", "status_perkawinan", "agama", "gol_darah"):
            KTPExtractor._snap_to_vocabulary(record, name)

        if record.provinsi.found and record.provinsi.value:
            record.provinsi.value = KTPExtractor._snap_province(record.provinsi.value)

        if record.tempat_tgl_lahir.found and record.tempat_tgl_lahir.value:
            record.tempat_tgl_lahir.value = normalize_tempat_tgl_lahir(record.tempat_tgl_lahir.value)

        if record.pekerjaan.found and record.pekerjaan.value:
            regency_hint = record.kabupaten_kota.value if record.kabupaten_kota.found else None
            record.pekerjaan.value = clean_pekerjaan(record.pekerjaan.value, regency=regency_hint)

        if record.kewarganegaraan.found and record.kewarganegaraan.value:
            record.kewarganegaraan.value = clean_kewarganegaraan(record.kewarganegaraan.value)

        if record.nik.found and record.nik.value:
            is_valid, cleaned, notes = validate_nik(record.nik.value)
            record.nik.value = cleaned
            record.nik_valid = is_valid
            record.nik_validation_notes = notes
            if not is_valid:
                logger.info("NIK failed validation: %s", notes)
                record.issues.append(ValidationIssue("nik", "NIK_INVALID", "error", "; ".join(notes)))
        else:
            record.nik_valid = False
            record.nik_validation_notes = ["NIK field was not detected at all."]
            record.issues.append(ValidationIssue("nik", "NIK_MISSING", "error", "NIK tidak terdeteksi."))

    @staticmethod
    def _snap_to_vocabulary(record: KTPRecord, name: str) -> None:
        target: FieldResult = getattr(record, name)
        if not (target.found and target.value):
            return
        canonical, score = vocabulary.snap(target.value, name, VOCAB_SNAP_MIN_SCORE)
        if canonical is None:
            record.issues.append(ValidationIssue(
                name, "VALUE_NOT_IN_VOCABULARY", "warning",
                f"Nilai '{target.value}' tidak dikenali sebagai nilai resmi "
                f"({', '.join(vocabulary.canonical_values(name))}).",
            ))
            target.confidence = min(target.confidence, LOW_CONFIDENCE_THRESHOLD - 0.01)
            return
        target.value = canonical
        target.confidence *= score

    @staticmethod
    def _snap_province(printed: str) -> str:
        """Correct small OCR typos in the province name using the official
        list; leave the value untouched if nothing is close enough."""
        upper = normalize_whitespace(printed.upper())
        best_name, best_score = None, 0.0
        for names in reference.provinces().values():
            for n in names:
                score = SequenceMatcher(None, upper, n).ratio()
                if score > best_score:
                    best_name, best_score = names[0], score
        return best_name if best_name and best_score >= 0.85 else printed

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
