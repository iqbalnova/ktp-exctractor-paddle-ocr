"""
Thin wrapper around PaddleOCR's PP-OCRv5 pipeline, configured for CPU-only
inference. Isolated in its own module so ``layout.py``, ``fields.py`` and
``validators.py`` stay importable (and unit-testable) without PaddleOCR
installed at all.
"""
from __future__ import annotations

import logging
from typing import List

from .models import TextBox, KTPExtractionError

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """Runs PP-OCRv5 text detection + recognition on CPU and normalizes the
    result into a flat list of ``TextBox``.
    """

    def __init__(
        self,
        lang: str = "id",
        device: str = "cpu",
        cpu_threads: int = 2,
        enable_mkldnn: bool = False,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        min_confidence: float = 0.0,
    ) -> None:
        """
        Args:
            lang: PaddleOCR language model. 'id' (Indonesian) is preferred;
                falls back to 'en' if unavailable in the installed model zoo.
            device: 'cpu' keeps this fully GPU-free, per requirement.
            cpu_threads: kept modest by default -- CPU inference for PP-OCR's
                mobile models is memory-bound more than compute-bound, and
                some PaddleOCR 3.x releases have had CPU memory regressions
                (see PaddlePaddle/PaddleOCR#17955) that get worse with more
                threads on constrained containers.
            enable_mkldnn: Intel's math library, meaningfully faster on x86.
                Set to False if you hit "ConvertPirAttribute2RuntimeAttribute"
                or similar backend errors on an older/unusual CPU.
            use_doc_orientation_classify: corrects whole-image rotation
                (e.g. a KTP photographed sideways).
            use_doc_unwarping: corrects page curl/perspective warp. Off by
                default -- KTP is a small rigid card, not a curled document
                page, and this step adds CPU time for little benefit here.
            use_textline_orientation: corrects individual rotated text
                lines, useful when the card itself is tilted in the photo.
            min_confidence: recognition results below this score are dropped
                before they ever reach field matching.
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - exercised only
            # without the optional dependency installed.
            raise KTPExtractionError(
                "paddleocr is not installed. Install with: "
                "pip install paddlepaddle paddleocr"
            ) from exc

        self.min_confidence = min_confidence
        self._lang = lang

        common_kwargs = dict(
            device=device,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
        )
        try:
            self._ocr = PaddleOCR(lang=lang, ocr_version="PP-OCRv5", **common_kwargs)
        except Exception as exc:
            logger.warning(
                "Failed to load PP-OCRv5 with lang=%r (%s); falling back to "
                "lang='en'. Indonesian-specific character sets may be less "
                "accurate.", lang, exc,
            )
            self._lang = "en"
            self._ocr = PaddleOCR(lang="en", ocr_version="PP-OCRv5", **common_kwargs)

    def run(self, image_path: str) -> List[TextBox]:
        """Run OCR on a single image file and return normalized text boxes."""
        import cv2

        scale = 1.0
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            max_side = 1200
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                target_input = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                target_input = img
        else:
            target_input = image_path

        try:
            results = self._ocr.predict(input=target_input)
        except Exception as exc:
            raise KTPExtractionError(f"OCR inference failed: {exc}") from exc

        boxes: List[TextBox] = []
        for page in results:
            res = page if isinstance(page, dict) else getattr(page, "json", {}).get("res", {})
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", res.get("rec_boxes"))

            if polys is None or len(texts) == 0:
                continue

            for text, score, poly in zip(texts, scores, polys):
                if score < self.min_confidence or not str(text).strip():
                    continue
                x1, y1, x2, y2 = _poly_to_box(poly)
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                boxes.append(TextBox(text=str(text), confidence=float(score),
                                      x1=x1, y1=y1, x2=x2, y2=y2))

        if not boxes:
            raise KTPExtractionError(
                "OCR produced no usable text at all -- check image quality, "
                "focus, and that the KTP fills a reasonable portion of the frame."
            )
        return boxes


def _poly_to_box(poly) -> tuple:
    """Normalize either a 4-point polygon ([[x,y],...]) or a flat
    [x1,y1,x2,y2] box (both appear across PaddleOCR versions/configs) into
    (x1, y1, x2, y2).
    """
    flat = [float(v) for point in poly for v in (point if hasattr(point, "__iter__") else [point])]
    if len(flat) == 4:
        return tuple(flat)
    xs = flat[0::2]
    ys = flat[1::2]
    return min(xs), min(ys), max(xs), max(ys)
