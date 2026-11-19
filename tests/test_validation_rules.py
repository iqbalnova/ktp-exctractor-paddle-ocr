"""
Regression tests for: labels inside values, vocabulary snapping,
NIK region checks, cross-field consistency rules and decision routing.
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ktp_extractor.consistency import rule_age_plausible
from ktp_extractor.extractor import KTPExtractor
from ktp_extractor.fields import FIELD_LABELS, scan_row_for_labels
from ktp_extractor.models import TextBox
from ktp_extractor.validators import parse_nik, validate_nik
from ktp_extractor.vocabulary import find_in_text, snap


def box(text, conf, x1, y1, x2, y2):
    return TextBox(text=text, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def _extract(rows):
    """rows: list of single-box row texts, laid out top to bottom."""
    extractor = KTPExtractor.__new__(KTPExtractor)
    boxes = [box(t, 0.98, 40, 10 + i * 30, 400, 30 + i * 30) for i, t in enumerate(rows)]
    return extractor.extract_from_boxes(boxes)


GOOD_CARD = [
    "PROVINSI JAWA BARAT",
    "KABUPATEN BOGOR",
    "NIK : 3201054507980001",
    "Nama : AGAM SAPUTRI",
    "Tempat/Tgl Lahir : BOGOR, 05-07-1998",
    "Jenis Kelamin : PEREMPUAN Gol. Darah : AB",
    "Alamat : JL. KECAPI STATUS NO. 4",
    "RT/RW : 003/004",
    "Kel/Desa : SUKASARI",
    "Kecamatan : BOGOR TIMUR",
    "Agama : ISLAM",
    "Status Perkawinan : BELUM KAWIN",
    "Pekerjaan : KARYAWAN SWASTA",
    "Kewarganegaraan : WNI",
    "Berlaku Hingga : SEUMUR HIDUP",
]


def test_label_words_inside_values_are_not_split():
    rec = _extract(GOOD_CARD)
    # "AGAM" is ~0.89 similar to the "Agama" label -- must stay in the name.
    assert rec.nama.value == "AGAM SAPUTRI", rec.nama.value
    # "KECAPI" / "STATUS" resemble "Kec" / "Status" aliases.
    assert rec.alamat.value == "JL. KECAPI STATUS NO. 4", rec.alamat.value
    assert rec.agama.value == "ISLAM"
    assert rec.status_perkawinan.value == "BELUM KAWIN"


def test_repeated_label_word_is_kept_in_value():
    segs = scan_row_for_labels("Nama Nama Asli Anda", FIELD_LABELS)
    assert len(segs) == 1 and segs[0][1] == "Nama Asli Anda", segs


def test_label_echo_is_treated_as_empty():
    segs = scan_row_for_labels("Agama Agama", FIELD_LABELS)
    assert segs[0][0].key == "agama" and segs[0][1] == ""


def test_inline_gol_darah_still_split():
    segs = scan_row_for_labels("Jenis Kelamin : LAKI-LAKI Gol. Darah : O", FIELD_LABELS)
    assert [(l.key, v) for l, v in segs] == [("jenis_kelamin", "LAKI-LAKI"), ("gol_darah", "O")]


def test_clean_card_is_accepted():
    rec = _extract(GOOD_CARD)
    assert rec.nik_valid is True
    assert rec.gol_darah.value == "AB"
    assert rec.issues == [], rec.issues
    assert rec.decision == "ACCEPT", (rec.decision, rec.low_confidence_fields, rec.missing_fields)


def test_birth_date_and_gender_mismatch_route_to_review():
    card = list(GOOD_CARD)
    card[4] = "Tempat/Tgl Lahir : BOGOR, 06-07-1998"   # NIK says 05-07-98
    card[5] = "Jenis Kelamin : LAKI-LAKI"              # NIK day 45 -> female
    rec = _extract(card)
    codes = {i.code for i in rec.issues}
    assert {"NIK_BIRTH_DATE_MISMATCH", "NIK_GENDER_MISMATCH"} <= codes, codes
    assert rec.decision == "REVIEW"


def test_province_mismatch_flagged():
    card = list(GOOD_CARD)
    card[0] = "PROVINSI JAWA TIMUR"   # NIK code 32 = JAWA BARAT
    rec = _extract(card)
    assert "NIK_PROVINCE_MISMATCH" in {i.code for i in rec.issues}


def test_province_typo_snapped_to_official_name():
    card = list(GOOD_CARD)
    card[0] = "PROVINSI JAWA BARAI"
    rec = _extract(card)
    assert rec.provinsi.value == "JAWA BARAT"


def test_invalid_nik_is_rejected():
    card = list(GOOD_CARD)
    card[2] = "NIK : 4529643389000738"   # province 45 does not exist, month 89
    rec = _extract(card)
    assert rec.nik_valid is False
    assert rec.decision == "REJECT"
    assert "NIK_INVALID" in {i.code for i in rec.issues}


def test_validate_nik_region_codes():
    assert validate_nik("9901050709876501")[0] is False   # province 99
    assert validate_nik("3200050709876501")[0] is False   # regency 00
    assert validate_nik("3201000709876501")[0] is False   # district 00
    assert validate_nik("9601050709876501")[0] is True    # Papua Barat Daya


def test_parse_nik_birth_date_century():
    info = parse_nik("3201054507050001")
    assert info.is_female and info.day == 5
    assert info.birth_date(today=date(2026, 1, 1)) == date(2005, 7, 5)
    assert parse_nik("3201050707800001").birth_date(today=date(2026, 1, 1)) == date(1980, 7, 7)


def test_age_rule_flags_future_birth_date():
    card = list(GOOD_CARD)
    card[4] = "Tempat/Tgl Lahir : BOGOR, 05-07-2098"
    rec = _extract(card)
    issues = rule_age_plausible(rec, today=date(2026, 1, 1))
    assert issues and issues[0].code == "BIRTH_DATE_IN_FUTURE"


def test_vocabulary_snap_and_whole_word_matching():
    assert snap("FEMALE", "jenis_kelamin")[0] == "PEREMPUAN"   # not "MALE"
    assert snap("MALE", "jenis_kelamin")[0] == "LAKI-LAKI"
    assert snap("PEREMPLAN", "jenis_kelamin")[0] == "PEREMPUAN"  # OCR typo
    assert snap("BELUM KAWIN", "status_perkawinan")[0] == "BELUM KAWIN"
    assert snap("CERAI MATI", "status_perkawinan")[0] == "CERAI MATI"
    assert snap("KATHOLIK", "agama")[0] == "KATOLIK"
    assert snap("0", "gol_darah")[0] == "O"
    assert snap("XYZ", "agama")[0] is None
    assert find_in_text("ALAMAT ISLAMIC CENTER", "agama") is None  # whole words only


def test_unknown_vocabulary_value_flagged_for_review():
    card = list(GOOD_CARD)
    card[10] = "Agama : QWERTY"
    rec = _extract(card)
    assert "VALUE_NOT_IN_VOCABULARY" in {i.code for i in rec.issues}
    assert "agama" in rec.low_confidence_fields
    assert rec.decision == "REVIEW"


def test_tempat_tgl_lahir_spaced_date_normalized():
    from ktp_extractor.validators import normalize_tempat_tgl_lahir
    assert normalize_tempat_tgl_lahir("Kota Lahir / 07 - 07 - 1997") == "Kota Lahir, 07-07-1997"
    assert normalize_tempat_tgl_lahir("BOGOR,30-10-2002") == "BOGOR, 30-10-2002"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK: {name}")
    print("\nAll tests passed.")
