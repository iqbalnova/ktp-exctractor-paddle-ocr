"""
Layout reconstruction for OCR output.

PaddleOCR returns text boxes in detection order, which is *roughly* top to
bottom but not reliably left-to-right within a row, and says nothing about
which boxes belong on the same visual line. KTP is a fixed-layout form
(label, then value, on the same horizontal row), so recovering rows is what
lets us do "label: value" matching instead of guessing from array indices.
"""
from __future__ import annotations

from typing import List
from .models import TextBox


def group_into_rows(boxes: List[TextBox], y_overlap_ratio: float = 0.5) -> List[List[TextBox]]:
    """Group text boxes into left-to-right reading-order rows.

    Two boxes are considered part of the same row if their vertical spans
    overlap by at least ``y_overlap_ratio`` of the shorter box's height.
    This tolerates the few-degree tilt that's common in phone photos of a
    KTP, without needing a full deskew step.
    """
    if not boxes:
        return []

    # Sort by vertical position first so rows are built top-to-bottom.
    remaining = sorted(boxes, key=lambda b: b.cy)
    rows: List[List[TextBox]] = []

    for box in remaining:
        placed = False
        for row in rows:
            if _overlaps_row(box, row, y_overlap_ratio):
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    # Order rows top-to-bottom, and boxes within each row left-to-right.
    rows.sort(key=lambda row: min(b.y1 for b in row))
    for row in rows:
        row.sort(key=lambda b: b.x1)

    return rows


def _overlaps_row(box: TextBox, row: List[TextBox], y_overlap_ratio: float) -> bool:
    row_y1 = min(b.y1 for b in row)
    row_y2 = max(b.y2 for b in row)
    overlap = min(box.y2, row_y2) - max(box.y1, row_y1)
    if overlap <= 0:
        return False
    shorter_height = min(box.height, row_y2 - row_y1 or box.height)
    return overlap / shorter_height >= y_overlap_ratio


def row_text(row: List[TextBox]) -> str:
    return " ".join(b.text for b in row).strip()


def row_confidence(row: List[TextBox]) -> float:
    if not row:
        return 0.0
    return sum(b.confidence for b in row) / len(row)
