"""
Cross-field consistency rules and the final routing decision.

The NIK encodes province, birth date and gender, which makes it the
strongest cross-check available on a KTP: when OCR misreads either the NIK
or the printed field, the two stop agreeing. Each rule is a small pure
function ``(record) -> list[ValidationIssue]`` registered in ``RULES``, so
adding a rule never requires touching extraction code.
"""
from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from typing import Callable, List, Optional

from . import reference
from .models import (
    DECISION_ACCEPT, DECISION_REJECT, DECISION_REVIEW, KTPRecord, ValidationIssue,
)
from .validators import parse_nik

_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")

# Fields without which the record is not useful for identity purposes.
CRITICAL_FIELDS = ("nik", "nama", "tempat_tgl_lahir")

MIN_AGE_YEARS = 17      # KTP is issued at 17 (or earlier if married) -> warning only
MAX_AGE_YEARS = 120

Rule = Callable[[KTPRecord], List[ValidationIssue]]


def _nik_info(record: KTPRecord):
    if not (record.nik_valid and record.nik.value):
        return None
    return parse_nik(record.nik.value)


def _printed_birth_date(record: KTPRecord) -> Optional[date]:
    if not (record.tempat_tgl_lahir.found and record.tempat_tgl_lahir.value):
        return None
    m = _DATE_RE.search(record.tempat_tgl_lahir.value.replace(" ", ""))
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _age_on(birth: date, today: date) -> int:
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def rule_nik_birth_date(record: KTPRecord) -> List[ValidationIssue]:
    info = _nik_info(record)
    if info is None or not record.tempat_tgl_lahir.found:
        return []
    printed = _printed_birth_date(record)
    if printed is None:
        return [ValidationIssue("tempat_tgl_lahir", "BIRTH_DATE_UNPARSEABLE", "warning",
                                "Tanggal lahir tidak dapat dibaca dengan format DD-MM-YYYY.")]
    if (printed.day, printed.month, printed.year % 100) != (info.day, info.month, info.year_2digit):
        return [ValidationIssue(
            "tempat_tgl_lahir", "NIK_BIRTH_DATE_MISMATCH", "warning",
            f"Tanggal lahir {printed:%d-%m-%Y} tidak cocok dengan NIK "
            f"({info.day:02d}-{info.month:02d}-xx{info.year_2digit:02d}).",
        )]
    return []


def rule_nik_gender(record: KTPRecord) -> List[ValidationIssue]:
    info = _nik_info(record)
    if info is None or not record.jenis_kelamin.found:
        return []
    expected = "PEREMPUAN" if info.is_female else "LAKI-LAKI"
    if record.jenis_kelamin.value != expected:
        return [ValidationIssue(
            "jenis_kelamin", "NIK_GENDER_MISMATCH", "warning",
            f"Jenis kelamin '{record.jenis_kelamin.value}' tidak cocok dengan NIK ({expected}).",
        )]
    return []


def rule_nik_province(record: KTPRecord) -> List[ValidationIssue]:
    info = _nik_info(record)
    if info is None or not (record.provinsi.found and record.provinsi.value):
        return []
    names = reference.provinces().get(info.province_code, [])
    printed = record.provinsi.value.upper()
    best = max((SequenceMatcher(None, printed, n).ratio() for n in names), default=0.0)
    if best < 0.8:
        return [ValidationIssue(
            "provinsi", "NIK_PROVINCE_MISMATCH", "warning",
            f"Provinsi '{record.provinsi.value}' tidak cocok dengan kode NIK "
            f"{info.province_code} ({names[0] if names else '?'}).",
        )]
    return []


def rule_age_plausible(record: KTPRecord, today: Optional[date] = None) -> List[ValidationIssue]:
    today = today or date.today()
    birth = _printed_birth_date(record)
    if birth is None:
        info = _nik_info(record)
        birth = info.birth_date(today) if info else None
    if birth is None:
        return []
    if birth > today:
        return [ValidationIssue("tempat_tgl_lahir", "BIRTH_DATE_IN_FUTURE", "error",
                                "Tanggal lahir berada di masa depan.")]
    age = _age_on(birth, today)
    if age > MAX_AGE_YEARS:
        return [ValidationIssue("tempat_tgl_lahir", "AGE_IMPLAUSIBLE", "warning",
                                f"Usia {age} tahun tidak wajar.")]
    if age < MIN_AGE_YEARS:
        return [ValidationIssue("tempat_tgl_lahir", "AGE_BELOW_KTP_MINIMUM", "warning",
                                f"Usia {age} tahun di bawah usia wajib KTP ({MIN_AGE_YEARS}).")]
    return []


RULES: List[Rule] = [
    rule_nik_birth_date,
    rule_nik_gender,
    rule_nik_province,
    rule_age_plausible,
]


def run_rules(record: KTPRecord) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for rule in RULES:
        issues.extend(rule(record))
    return issues


def decide(record: KTPRecord) -> str:
    """ACCEPT only when every critical field is present, the NIK is valid,
    nothing is low-confidence and no rule raised an issue."""
    if not record.nik.found or record.nik_valid is False:
        return DECISION_REJECT
    if any(i.severity == "error" for i in record.issues):
        return DECISION_REJECT
    missing_critical = any(f in record.missing_fields for f in CRITICAL_FIELDS)
    if record.issues or record.low_confidence_fields or missing_critical:
        return DECISION_REVIEW
    return DECISION_ACCEPT
