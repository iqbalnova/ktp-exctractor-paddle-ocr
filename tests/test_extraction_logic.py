"""
Unit tests that exercise the extraction logic WITHOUT PaddleOCR installed,
by feeding synthetic TextBox lists directly to KTPExtractor.extract_from_boxes.
This mirrors what a real (imperfect) OCR pass on a KTP looks like: labels
and values as separate boxes on the same row, occasional typos, and mildly
inconsistent box heights.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ktp_extractor.extractor import KTPExtractor
from ktp_extractor.models import TextBox
from ktp_extractor.validators import validate_nik, normalize_rt_rw
from ktp_extractor.fields import match_label, FIELD_LABELS


def box(text, conf, x1, y1, x2, y2):
    return TextBox(text=text, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def make_synthetic_ktp_boxes():
    """Roughly mimics real PP-OCRv5 output for a KTP: label and value are
    separate detection boxes on the same row; y-coordinates aren't perfectly
    aligned (simulates slight photo tilt); NIK is OCR-mangled ('O' for '0').
    """
    return [
        box("PROVINSI JAWA BARAT", 0.98, 40, 10, 300, 30),
        box("KABUPATEN BOGOR", 0.97, 40, 32, 260, 52),
        box("N1K", 0.91, 40, 60, 90, 80),          # typo'd label
        box(": 3201O5O7098765O1", 0.85, 95, 61, 320, 81),  # OCR'd NIK w/ letter-O
        box("Nama", 0.99, 40, 90, 100, 110),
        box(": BUDI SANTOSO", 0.93, 105, 91, 320, 111),
        box("Tempat/Tgl Lahir", 0.88, 40, 118, 200, 138),
        box(": BOGOR, 05-07-1998", 0.90, 205, 119, 400, 139),
        box("Jenis Kelamin", 0.95, 40, 146, 180, 166),
        box(": LAKI-LAKI", 0.92, 185, 147, 300, 167),
        box("Gol. Darah", 0.80, 320, 146, 420, 166),
        box(": O", 0.70, 425, 147, 450, 167),
        box("Alamat", 0.94, 40, 174, 120, 194),
        box(": JL. MERDEKA NO. 17", 0.89, 125, 175, 380, 195),
        box("RT/RW", 0.90, 40, 202, 120, 222),
        box(": 003/004", 0.88, 125, 203, 250, 223),
        box("Kel/Desa", 0.93, 40, 230, 130, 250),
        box(": SUKASARI", 0.91, 135, 231, 300, 251),
        box("Kecamatan", 0.93, 40, 258, 150, 278),
        box(": BOGOR TIMUR", 0.90, 155, 259, 350, 279),
        box("Agama", 0.95, 40, 286, 110, 306),
        box(": ISLAM", 0.94, 115, 287, 250, 307),
        box("Kewarganegaraan", 0.92, 40, 370, 200, 390),
        box(": WNI", 0.90, 205, 371, 300, 391),
    ]


def test_end_to_end_extraction_with_synthetic_boxes():
    extractor = KTPExtractor.__new__(KTPExtractor)  # skip __init__ (no OCR engine needed)
    record = extractor.extract_from_boxes(make_synthetic_ktp_boxes())

    assert record.provinsi.value == "JAWA BARAT"
    assert record.kabupaten_kota.value == "BOGOR"
    assert record.nama.value == "BUDI SANTOSO"
    assert record.jenis_kelamin.value == "LAKI-LAKI"
    assert record.alamat.value == "JL. MERDEKA NO. 17"
    assert record.rt_rw.value == "003/004"
    assert record.kel_desa.value == "SUKASARI"
    assert record.kecamatan.value == "BOGOR TIMUR"
    assert record.kewarganegaraan.value == "WNI"
    assert record.tempat_tgl_lahir.value == "BOGOR, 05-07-1998"

    # Two fields printed side by side on the same physical row must both
    # be captured, not just the first one on the row.
    assert record.gol_darah.value == "O"

    # NIK label was OCR-typo'd to "N1K" -- fuzzy matching must still find it.
    assert record.nik.found is True
    # And the letter/digit confusion ('O' -> '0') must be repaired before validation.
    assert record.nik.value == "3201050709876501"
    assert record.nik_valid is True

    # Fields genuinely absent from this synthetic card should be reported
    # as missing, not silently defaulted or hallucinated.
    assert set(record.missing_fields) == {"berlaku_hingga", "status_perkawinan", "pekerjaan"}
    print("OK: end-to-end synthetic extraction")


def test_nik_validation_valid_male():
    is_valid, cleaned, notes = validate_nik("3201050709876501")
    assert is_valid is True
    assert cleaned == "3201050709876501"
    assert "male" in notes
    print("OK: valid male NIK")


def test_nik_validation_valid_female_day_offset():
    # Day 45 = 05 + 40 -> female, day 05.
    is_valid, cleaned, notes = validate_nik("3201054509876501")
    assert is_valid is True
    assert "female" in notes
    print("OK: valid female NIK (day+40 offset)")


def test_nik_validation_rejects_bad_month():
    # PP=32 RR=01 SS=05 DD=05 MM=13 (invalid) YY=87 SSSS=6501
    is_valid, cleaned, notes = validate_nik("3201050513876501")
    assert is_valid is False
    print("OK: invalid month correctly rejected")


def test_nik_validation_rejects_wrong_length():
    is_valid, cleaned, notes = validate_nik("12345")
    assert is_valid is False
    assert "16 digits" in notes[0]
    print("OK: wrong-length NIK correctly rejected")


def test_nik_digit_confusion_repair():
    # 'O' -> '0', 'l'/'I' -> '1', repaired before length/format checks run.
    is_valid, cleaned, notes = validate_nik("32OI0S0709876SOl")
    assert cleaned == "3201050709876501"
    assert is_valid is True
    print("OK: OCR digit-confusion repair")


def test_rt_rw_normalization_variants():
    assert normalize_rt_rw("003/004") == "003/004"
    assert normalize_rt_rw("3/4") == "003/004"
    assert normalize_rt_rw("003004") == "003/004"
    print("OK: RT/RW normalization variants")


def test_label_fuzzy_matching_tolerates_typo():
    nik_label = next(l for l in FIELD_LABELS if l.key == "nik")
    assert match_label("N1K : 123", nik_label) == "123"
    assert match_label("NIK 123", nik_label) == "123"
    assert match_label("Nama BUDI", nik_label) is None  # must not cross-match
    print("OK: fuzzy label matching (typo-tolerant, no false positives)")


if __name__ == "__main__":
    test_end_to_end_extraction_with_synthetic_boxes()
    test_nik_validation_valid_male()
    test_nik_validation_valid_female_day_offset()
    test_nik_validation_rejects_bad_month()
    test_nik_validation_rejects_wrong_length()
    test_nik_digit_confusion_repair()
    test_rt_rw_normalization_variants()
    test_label_fuzzy_matching_tolerates_typo()
    print("\nAll tests passed.")
