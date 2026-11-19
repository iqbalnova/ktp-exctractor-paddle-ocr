"""
Reference data loader (labels, vocabularies, province codes).

All domain knowledge that used to be hard-coded in Python now lives in the
JSON files next to this module, so it can be reviewed, versioned and
extended without touching extraction logic. Set ``KTP_REFERENCE_DIR`` to a
directory containing replacement files to override any of them at runtime
(missing files fall back to the bundled defaults).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

_BUNDLED_DIR = Path(__file__).resolve().parent


def _load_json(name: str) -> Any:
    override_dir = os.getenv("KTP_REFERENCE_DIR")
    if override_dir:
        candidate = Path(override_dir) / name
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as fh:
                return json.load(fh)
    with (_BUNDLED_DIR / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def _strip_comments(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=None)
def label_definitions() -> List[Dict[str, Any]]:
    return list(_load_json("labels.json")["labels"])


@lru_cache(maxsize=None)
def vocabularies() -> Dict[str, Dict[str, Any]]:
    return _strip_comments(_load_json("vocabulary.json"))


@lru_cache(maxsize=None)
def provinces() -> Dict[str, List[str]]:
    """Map of 2-digit province code -> [printed name, *aliases]."""
    return _strip_comments(_load_json("provinces.json"))


def clear_cache() -> None:
    """Reload reference files on next access (useful in tests / hot reload)."""
    label_definitions.cache_clear()
    vocabularies.cache_clear()
    provinces.cache_clear()
