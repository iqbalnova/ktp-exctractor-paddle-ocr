from .extractor import KTPExtractor
from .models import KTPRecord, FieldResult, TextBox, KTPExtractionError
from .ocr_engine import PaddleOCREngine

__all__ = [
    "KTPExtractor",
    "KTPRecord",
    "FieldResult",
    "TextBox",
    "KTPExtractionError",
    "PaddleOCREngine",
]

__version__ = "1.0.0"
