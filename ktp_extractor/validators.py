"""
Validation and cleanup helpers.

NIK (Nomor Induk Kependudukan) has a well-known structure:
    PP RR SS DD MM YY SSSS
    PP   = province code
    RR   = regency/city code
    SS   = district (kecamatan) code
    DDMMYY = date of birth, with DD + 40 for female residents
    SSSS = serial number

There is no public official checksum digit, so validation here is
structural (length, digit-only, plausible date) rather than cryptographic.
It's a sanity check to catch OCR misreads, not a guarantee of authenticity.
"""
from __future__ import annotations

import calendar
import re
from typing import List, Tuple, Optional

# Common OCR character confusions seen on digit-heavy fields like NIK.
_DIGIT_CONFUSIONS = {
    "O": "0", "o": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
    "Q": "0",
}


def clean_digit_string(raw: str) -> str:
    """Strip everything but digits, first repairing common letter/digit OCR
    confusions so e.g. 'I234567B90I23456' -> '12345678901234456' -> digits.
    """
    repaired = "".join(_DIGIT_CONFUSIONS.get(ch, ch) for ch in raw)
    return re.sub(r"\D", "", repaired)


def validate_nik(raw_value: str) -> Tuple[bool, str, List[str]]:
    """Validate and clean a candidate NIK string.

    Returns (is_valid, cleaned_16_digit_string_or_original, notes).
    """
    notes: List[str] = []
    digits = clean_digit_string(raw_value)

    if len(digits) != 16:
        notes.append(f"Expected 16 digits after cleanup, got {len(digits)}.")
        return False, digits or raw_value, notes

    dd = int(digits[6:8])
    mm = int(digits[8:10])
    yy = int(digits[10:12])

    if not (1 <= mm <= 12):
        notes.append(f"Month segment '{digits[8:10]}' is not 01-12.")
        return False, digits, notes

    is_female = dd > 40
    day = dd - 40 if is_female else dd

    # Try both centuries; NIK doesn't encode which one.
    plausible_year_found = False
    for century_year in (1900 + yy, 2000 + yy):
        max_day = calendar.monthrange(century_year, mm)[1]
        if 1 <= day <= max_day:
            plausible_year_found = True
            break

    if not plausible_year_found:
        notes.append(f"Day segment '{day:02d}' is not valid for month {mm:02d}.")
        return False, digits, notes

    notes.append("female" if is_female else "male")
    return True, digits, notes


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_label_punctuation(text: str) -> str:
    """Remove a leading colon and surrounding junk left over after a label
    like 'Nama' has been matched and removed from the front of a row, e.g.
    ': BUDI SANTOSO' -> 'BUDI SANTOSO'.
    """
    return normalize_whitespace(re.sub(r"^[\s:;.\-]+", "", text))


def normalize_rt_rw(raw_value: str) -> str:
    """Normalize RT/RW readings like '00304', '003 004', '003/004' into a
    consistent 'RRR/WWW' form. Falls back to the cleaned digits if the
    format is ambiguous, rather than guessing.
    """
    digits = re.sub(r"\D", "", raw_value)
    if "/" in raw_value:
        parts = [p.strip() for p in raw_value.split("/") if p.strip()]
        if len(parts) == 2:
            rt = re.sub(r"\D", "", parts[0]).zfill(3)
            rw = re.sub(r"\D", "", parts[1]).zfill(3)
            return f"{rt}/{rw}"
    if len(digits) == 6:
        return f"{digits[:3]}/{digits[3:]}"
    if len(digits) == 5:
        # Ambiguous split; assume 3+2 since 2-digit RT/RW is rare but not
        # unheard of, and flag it as such via the raw digits fallback.
        return f"{digits[:3]}/{digits[3:]}"
    return raw_value.strip()


def clean_pekerjaan(pekerjaan: str, regency: Optional[str] = None) -> str:
    """Clean up trailing issuance stamp or regency/date text accidentally
    merged into the Pekerjaan field. Does not restrict or mutate legitimate job titles.

    Examples:
        'PELAJAR/MAHASISWA KOTA BOGOR' -> 'PELAJAR/MAHASISWA'
        'WIRASWASTA BANYUWANGI' -> 'WIRASWASTA'
        'PROGRAMMER 12-05-2021' -> 'PROGRAMMER'
    """
    text = normalize_whitespace(pekerjaan)

    # 1. Strip trailing date if present (e.g. 06-11-2020)
    text = re.sub(r"\s+\d{2}[-\s/]\d{2}[-\s/]\d{4}$", "", text).strip()

    # 2. Strip trailing regency name if known (e.g. BOGOR, BANYUWANGI)
    if regency:
        reg_clean = re.sub(r"^(KOTA|KABUPATEN)\s+", "", regency.strip(), flags=re.IGNORECASE).strip()
        if reg_clean and len(reg_clean) >= 3:
            pattern = rf"\s+(?:KOTA|KABUPATEN)?\s*{re.escape(reg_clean)}$"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 3. Strip generic trailing "KOTA ..." or "KABUPATEN ..." (e.g. "PELAJAR/MAHASISWA KOTA BOGOR")
    text = re.sub(r"\s+(?:KOTA|KABUPATEN)\s+[A-Z\s]+$", "", text, flags=re.IGNORECASE).strip()
    return text



def clean_kewarganegaraan(raw_value: str) -> str:
    """Normalize kewarganegaraan, stripping accidental issuance date."""
    upper = raw_value.upper()
    if "WNA" in upper:
        return "WNA"
    if "WNI" in upper:
        return "WNI"
    return normalize_whitespace(raw_value)


def normalize_tempat_tgl_lahir(raw: str) -> str:
    """Normalize Tempat/Tgl Lahir so the separator between birthplace and birthdate
    is consistently standardized as ', ' (comma space), fixing OCR misreads where
    the comma was scanned as a dot, colon, or space.

    Examples:
        'FUJIAN.25-03-1977'  -> 'FUJIAN, 25-03-1977'
        'BOGOR,30-10-2002'   -> 'BOGOR, 30-10-2002'
        'JAKARTA 15-08-1995' -> 'JAKARTA, 15-08-1995'
    """
    if not raw:
        return raw
    text = normalize_whitespace(raw)
    # Match: Place + separator + Date (DD-MM-YYYY or DD.MM.YYYY or DD/MM/YYYY)
    m = re.search(r"^(.*?)[,.:;\s]+(\d{2}[-\s/.]\d{2}[-\s/.]\d{4})$", text)
    if m:
        place = re.sub(r"^[,\s:;.\-]+", "", m.group(1)).strip()
        date_str = m.group(2).replace("/", "-").replace(".", "-").replace(" ", "-")
        if place:
            return f"{place}, {date_str}"
        return date_str
    return text


