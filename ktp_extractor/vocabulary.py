"""
Snap noisy OCR values onto an official vocabulary.

Replaces ad-hoc ``if "KAWIN" in upper`` checks with one data-driven routine:

1. exact match against canonical values / synonyms        -> score 1.0
2. whole-word phrase containment (longest phrase first,     -> score 0.95
   so "BELUM KAWIN" wins over "KAWIN", and "FEMALE" is never
   mistaken for "MALE")
3. fuzzy similarity against every phrase                    -> score = ratio

The returned score is meant to be multiplied into the field confidence, so
a value that only matched fuzzily is visibly less certain than an exact one.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from . import reference

_NON_WORD = re.compile(r"[^A-Z0-9+\-]+")


def _norm(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.upper()).split())


@lru_cache(maxsize=None)
def _phrases(vocab_name: str) -> Tuple[Tuple[str, str], ...]:
    """(normalized phrase, canonical) pairs, longest phrase first."""
    vocab = reference.vocabularies()[vocab_name]
    pairs: Dict[str, str] = {}
    for canonical in vocab.get("canonical", []):
        pairs[_norm(canonical)] = canonical
    for synonym, canonical in vocab.get("synonyms", {}).items():
        pairs[_norm(synonym)] = canonical
    pairs.pop("", None)
    return tuple(sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0])))


def canonical_values(vocab_name: str) -> List[str]:
    return list(reference.vocabularies()[vocab_name].get("canonical", []))


def _contains_phrase(haystack: str, phrase: str) -> bool:
    return re.search(rf"(?<![A-Z0-9]){re.escape(phrase)}(?![A-Z0-9])", haystack) is not None


def find_in_text(text: str, vocab_name: str) -> Optional[str]:
    """Return the canonical value whose phrase appears as whole words in
    ``text`` (no fuzzy matching -- used for label-less fallbacks where a
    false positive is worse than a miss)."""
    haystack = _norm(text)
    if not haystack:
        return None
    for phrase, canonical in _phrases(vocab_name):
        if len(phrase) < 2 and haystack != phrase:
            continue  # single letters (blood type) only count as a whole value
        if _contains_phrase(haystack, phrase):
            return canonical
    return None


def snap(value: str, vocab_name: str, min_score: float = 0.75) -> Tuple[Optional[str], float]:
    """Map ``value`` to a canonical vocabulary entry.

    Returns ``(canonical, score)``; ``canonical`` is None when nothing
    reaches ``min_score``.
    """
    normalized = _norm(value or "")
    if not normalized:
        return None, 0.0

    phrases = _phrases(vocab_name)
    for phrase, canonical in phrases:
        if normalized == phrase:
            return canonical, 1.0

    found = find_in_text(normalized, vocab_name)
    if found is not None:
        return found, 0.95

    best: Tuple[Optional[str], float] = (None, 0.0)
    for phrase, canonical in phrases:
        if len(phrase) < 3:
            continue  # fuzzy-matching 1-2 char codes is meaningless
        ratio = SequenceMatcher(None, normalized, phrase).ratio()
        if ratio > best[1]:
            best = (canonical, ratio)
    if best[1] >= min_score:
        return best
    return None, best[1]
