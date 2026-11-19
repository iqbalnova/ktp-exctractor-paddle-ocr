"""
Field label catalogue and fuzzy label matching.

Real-world OCR on a phone-photographed KTP misreads labels constantly
("NIK" -> "N1K", "Kewarganegaraan" -> "Kewarganegaman"). Rather than
hand-patching each typo, each row's leading tokens are matched against a
label's canonical text using a similarity ratio, so previously-unseen
misreadings within a small edit distance still match.

The catalogue itself (labels, aliases, thresholds) is data, loaded from
``reference/labels.json``.

Labels are only recognised at the START of a row, plus labels explicitly
marked ``inline`` (printed side by side on the same physical row, e.g.
"Gol. Darah" next to "Jenis Kelamin"). Scanning every word of a row for
labels would split values that happen to contain a label-like word -- e.g.
the name "AGAM" (~0.89 similar to "Agama") or an address containing "KEC".
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from . import reference
from .validators import strip_label_punctuation


@dataclass(frozen=True)
class FieldLabel:
    key: str                 # matches a KTPRecord attribute name
    canonical: str            # human-readable label as printed on a KTP
    aliases: Tuple[str, ...]  # alternate renderings actually seen in the wild
    # 0.80 is deliberately conservative: short labels like "Agama" (5
    # chars) and "Alamat" (6 chars) are similar enough to each other
    # (ratio ~0.73) that a looser threshold causes cross-field
    # misclassification. Known specific OCR misreadings for short labels
    # should be added to their `aliases` instead of loosening this.
    min_similarity: float = 0.80
    # True if this label may appear after another field's value on the
    # same physical row.
    inline: bool = False

    @property
    def candidates(self) -> Tuple[str, ...]:
        # Longest rendering first, so "Jenis Kelamin / Sex" is consumed
        # whole instead of leaving "/ Sex" behind as part of the value.
        return tuple(sorted((self.canonical, *self.aliases), key=lambda c: -len(c.split())))


def load_field_labels() -> List[FieldLabel]:
    return [
        FieldLabel(
            key=d["key"],
            canonical=d["canonical"],
            aliases=tuple(d.get("aliases", ())),
            min_similarity=float(d.get("min_similarity", 0.80)),
            inline=bool(d.get("inline", False)),
        )
        for d in reference.label_definitions()
    ]


# Order matters: follows the physical layout of a KTP.
FIELD_LABELS: List[FieldLabel] = load_field_labels()

# These two never appear with a colon-style label; they're printed as a bare
# running header ("PROVINSI JAWA BARAT") above the NIK row, so they're
# matched by keyword-in-row rather than label-then-value.
PROVINCE_KEYWORD = "PROVINSI"
REGENCY_KEYWORDS = ("KABUPATEN", "KOTA")

# Tokens that carry no information before a label (OCR noise at card edge).
_LEADING_NOISE = {":", "|", "-", ".", ",", ";", "/", "\\"}


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _label_match_length(words: List[str], start: int, label: FieldLabel) -> Optional[int]:
    """If a rendering of ``label`` begins at ``words[start]``, return how
    many words it consumes. Otherwise return None.

    The candidate prefix length is anchored to each candidate string's OWN
    word count (±1, to tolerate OCR merging/splitting a word), rather than
    trying arbitrary lengths.
    """
    candidates = label.candidates

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


def _first_content_index(words: List[str]) -> int:
    i = 0
    while i < len(words) and words[i] in _LEADING_NOISE:
        i += 1
    return i


def match_label(row_text: str, label: FieldLabel) -> Optional[str]:
    """If ``row_text`` starts with ``label``, return everything after it as
    the value. Field extraction itself uses ``scan_row_for_labels``."""
    words = row_text.split()
    start = _first_content_index(words)
    consumed = _label_match_length(words, start, label)
    if consumed is None:
        return None
    return strip_label_punctuation(" ".join(words[start + consumed:]))


def find_label_row_index(rows_text: List[str], label: FieldLabel, start_from: int = 0) -> Optional[int]:
    for i in range(start_from, len(rows_text)):
        if match_label(rows_text[i], label) is not None:
            return i
    return None


def _is_label_echo(value: str, label: FieldLabel) -> bool:
    """True if the 'value' is just the label printed again (blank template
    cards, or OCR duplicating a box), e.g. row "Agama Agama"."""
    v = strip_label_punctuation(value).lower()
    return bool(v) and any(_similarity(v, c) >= 0.9 for c in label.candidates)


def scan_row_for_labels(row_text: str, labels: List[FieldLabel]) -> List[Tuple[FieldLabel, str]]:
    """Find the label:value segment(s) on a single physical row.

    The row must START with a label. After that, only ``inline`` labels
    (that haven't matched yet on this row) may start a new segment, which
    handles "Jenis Kelamin : LAKI-LAKI   Gol. Darah : O" without letting
    words inside ordinary values be mistaken for labels.
    """
    words = row_text.split()
    start = _first_content_index(words)
    if start >= len(words):
        return []

    first = None
    for label in labels:
        consumed = _label_match_length(words, start, label)
        if consumed is not None:
            first = (start, label, consumed)
            break
    if first is None:
        return []

    matches: List[Tuple[int, FieldLabel, int]] = [first]
    used_keys = {first[1].key}
    inline_labels = [l for l in labels if l.inline]

    i = start + first[2]
    while i < len(words) and inline_labels:
        found_here = None
        for label in inline_labels:
            if label.key in used_keys:
                continue
            consumed = _label_match_length(words, i, label)
            if consumed is not None:
                found_here = (i, label, consumed)
                break
        if found_here:
            matches.append(found_here)
            used_keys.add(found_here[1].key)
            i += found_here[2]
        else:
            i += 1

    segments: List[Tuple[FieldLabel, str]] = []
    for idx, (seg_start, label, consumed) in enumerate(matches):
        value_start = seg_start + consumed
        value_end = matches[idx + 1][0] if idx + 1 < len(matches) else len(words)
        value = strip_label_punctuation(" ".join(words[value_start:value_end]))
        if _is_label_echo(value, label):
            value = ""
        segments.append((label, value))
    return segments
