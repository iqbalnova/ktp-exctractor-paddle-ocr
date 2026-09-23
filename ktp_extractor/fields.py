"""
Field label catalogue and fuzzy label matching.

Real-world OCR on a phone-photographed KTP misreads labels constantly
("NIK" -> "N1K", "Kewarganegaraan" -> "Kewarganegaman"). Rather than
hand-patching each typo we've personally seen (which is what makes
regex-patch approaches brittle and never-finished), we match each row's
leading tokens against a label's canonical text using a similarity ratio,
so previously-unseen misreadings within a small edit distance still match.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .validators import strip_label_punctuation


@dataclass(frozen=True)
class FieldLabel:
    key: str                 # matches a KTPRecord attribute name
    canonical: str            # human-readable label as printed on a KTP
    aliases: Tuple[str, ...]  # alternate renderings actually seen in the wild
    # 0.80 is deliberately conservative: short labels like "Agama" (5
    # chars) and "Alamat" (6 chars) are similar enough to each other
    # (ratio ~0.73) that a looser threshold causes cross-field
    # misclassification -- verified empirically against every label pair
    # in this catalogue. Known specific OCR misreadings for short labels
    # should be added to their `aliases` tuple instead of loosening this.
    min_similarity: float = 0.80


# Order matters: rows are matched top-to-bottom against this list, and a row
# is only checked against labels that haven't already been consumed, so
# ordering roughly follows the physical layout of a KTP.
FIELD_LABELS: List[FieldLabel] = [
    FieldLabel("nik", "NIK", ("N I K", "NlK", "N1K")),
    FieldLabel("nama", "Nama", ()),
    FieldLabel("tempat_tgl_lahir", "Tempat/Tgl Lahir",
               ("Tempat Tgl Lahir", "Tempat/TglLahir", "TempatTgl Lahir", "Tempat/Tgi Lahir", "Tempat/Tol Lah",
                "Tompal/ToiLahir", "Tompal/Toi Lahir", "Tempat/Tgl Lah", "Tempat/TglLah")),
    FieldLabel("jenis_kelamin", "Jenis Kelamin", ("JenisKelamin", "Sex", "Jenis Kelamin / Sex", "leas kela", "leas")),
    FieldLabel("gol_darah", "Gol. Darah", ("GolDarah", "Gol Darah", "Blood Type"), min_similarity=0.85),
    FieldLabel("alamat", "Alamat", (), min_similarity=0.82),
    FieldLabel("rt_rw", "RT/RW", ("RTRW", "RT / RW", "RT-RW", "RTAW")),
    FieldLabel("kel_desa", "Kel/Desa", ("KelDesa", "Kel Desa", "Ke/Desa", "Kel/Kelurahan")),
    FieldLabel("kecamatan", "Kecamatan", ("uatan", "Kec.", "Kec")),
    FieldLabel("agama", "Agama", ("Religion", "Agama / Religion")),
    FieldLabel("status_perkawinan", "Status Perkawinan",
               ("StatusPerkawinan", "Status Pedawnan", "Status", "Marital Status", "Status Perkawinan / Marital Status")),
    FieldLabel("pekerjaan", "Pekerjaan", ("Occupation", "Pekerian", "Pekorjaan", "Pekerpaan", "Pekerjaan / Occupation")),
    FieldLabel("kewarganegaraan", "Kewarganegaraan",
               ("Nationality", "Kewargnegaraan", "Kevarganegaraan", "Kawarganegaraan", "Ke aan WN", "Ke aan", "Kewarganegaraan / Nationality")),
    FieldLabel("berlaku_hingga", "Berlaku Hingga", ("BerlakuHingga", "Berlaku Sampai", "Berlaku Hegga", "Expiry Date")),
]

# These two never appear with a colon-style label; they're printed as a bare
# running header ("PROVINSI JAWA BARAT") above the NIK row, so they're
# matched by keyword-in-row rather than label-then-value.
PROVINCE_KEYWORD = "PROVINSI"
REGENCY_KEYWORDS = ("KABUPATEN", "KOTA")


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _label_match_length(words: List[str], start: int, label: FieldLabel) -> Optional[int]:
    """If a rendering of ``label`` begins at ``words[start]``, return how
    many words it consumes. Otherwise return None.

    The candidate prefix length is anchored to each candidate string's OWN
    word count (±1, to tolerate OCR merging/splitting a word), rather than
    trying arbitrary lengths.
    """
    candidates = (label.canonical, *label.aliases)

    # Pass 1: exact match (case-insensitive)
    for candidate in candidates:
        base_len = len(candidate.split())
        for word_count in (base_len, base_len - 1, base_len + 1):
            if word_count <= 0 or start + word_count > len(words):
                continue
            prefix = " ".join(words[start:start + word_count])
            if prefix.lower() == candidate.lower():
                return word_count

    # Pass 2: fuzzy match with guard against over-consuming unrelated leading words
    for candidate in candidates:
        base_len = len(candidate.split())
        for word_count in (base_len, base_len - 1, base_len + 1):
            if word_count <= 0 or start + word_count > len(words):
                continue
            prefix = " ".join(words[start:start + word_count])
            req_sim = max(label.min_similarity, 0.85) if word_count > base_len else label.min_similarity
            if _similarity(prefix, candidate) >= req_sim:
                return word_count
    return None


def match_label(row_text: str, label: FieldLabel) -> Optional[str]:
    """Convenience wrapper: if ``row_text`` starts with ``label``, return
    everything after it as the value. Used for the header rows (province /
    regency) and in tests; field extraction itself uses
    ``scan_row_for_labels`` below, which additionally supports more than one
    label:value pair on the same physical row (e.g. "Jenis Kelamin" and
    "Gol. Darah" are printed side by side on a real KTP).
    """
    words = row_text.split()
    consumed = _label_match_length(words, 0, label)
    if consumed is None:
        return None
    return strip_label_punctuation(" ".join(words[consumed:]))


def find_label_row_index(rows_text: List[str], label: FieldLabel, start_from: int = 0) -> Optional[int]:
    for i in range(start_from, len(rows_text)):
        if match_label(rows_text[i], label) is not None:
            return i
    return None


def scan_row_for_labels(row_text: str, labels: List[FieldLabel]) -> List[Tuple[FieldLabel, str]]:
    """Find every label:value segment on a single physical row, in left-to-
    right order. Handles the common case of two short fields printed side
    by side (e.g. "Jenis Kelamin : LAKI-LAKI   Gol. Darah : O").
    """
    words = row_text.split()
    matches: List[Tuple[int, FieldLabel, int]] = []  # (start_idx, label, consumed)

    i = 0
    while i < len(words):
        found_here = None
        for label in labels:
            consumed = _label_match_length(words, i, label)
            if consumed is not None:
                found_here = (label, consumed)
                break
        if found_here:
            label, consumed = found_here
            matches.append((i, label, consumed))
            i += consumed
        else:
            i += 1

    segments: List[Tuple[FieldLabel, str]] = []
    for idx, (start, label, consumed) in enumerate(matches):
        value_start = start + consumed
        value_end = matches[idx + 1][0] if idx + 1 < len(matches) else len(words)
        value = strip_label_punctuation(" ".join(words[value_start:value_end]))
        segments.append((label, value))
    return segments
