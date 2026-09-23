"""
Data models for KTP (Indonesian ID card) OCR extraction results.

These are plain dataclasses with zero external dependencies, so they can be
imported and unit-tested without PaddleOCR (or any OCR engine) installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class TextBox:
    """A single recognized text region from the OCR engine, normalized to a
    simple axis-aligned box regardless of the engine's native polygon format.
    """
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 1.0)


@dataclass
class FieldResult:
    """The extracted value for a single KTP field, with provenance so callers
    can decide whether to trust it or flag it for manual review.
    """
    value: Optional[str] = None
    confidence: float = 0.0
    found: bool = False
    source_row_text: Optional[str] = None


@dataclass
class KTPRecord:
    """Structured representation of everything we attempt to extract from a
    single KTP image.
    """
    provinsi: FieldResult = field(default_factory=FieldResult)
    kabupaten_kota: FieldResult = field(default_factory=FieldResult)
    nik: FieldResult = field(default_factory=FieldResult)
    nama: FieldResult = field(default_factory=FieldResult)
    tempat_tgl_lahir: FieldResult = field(default_factory=FieldResult)
    jenis_kelamin: FieldResult = field(default_factory=FieldResult)
    gol_darah: FieldResult = field(default_factory=FieldResult)
    alamat: FieldResult = field(default_factory=FieldResult)
    rt_rw: FieldResult = field(default_factory=FieldResult)
    kel_desa: FieldResult = field(default_factory=FieldResult)
    kecamatan: FieldResult = field(default_factory=FieldResult)
    agama: FieldResult = field(default_factory=FieldResult)
    status_perkawinan: FieldResult = field(default_factory=FieldResult)
    pekerjaan: FieldResult = field(default_factory=FieldResult)
    kewarganegaraan: FieldResult = field(default_factory=FieldResult)
    berlaku_hingga: FieldResult = field(default_factory=FieldResult)

    # Metadata about the extraction run itself, not about the person.
    nik_valid: Optional[bool] = None
    nik_validation_notes: list = field(default_factory=list)
    low_confidence_fields: list = field(default_factory=list)
    missing_fields: list = field(default_factory=list)
    overall_confidence: float = 0.0
    raw_text_lines: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


class KTPExtractionError(Exception):
    """Raised when the OCR engine fails or returns no usable text at all.
    Field-level misses are NOT raised as errors -- they show up as
    ``missing_fields`` / ``low_confidence_fields`` on the KTPRecord instead,
    since a partially-readable KTP is still a useful result for the caller.
    """
