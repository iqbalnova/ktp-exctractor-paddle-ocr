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


def group_into_rows(boxes: List[TextBox], y_threshold_ratio: float = 0.42) -> List[List[TextBox]]:
    """Group text boxes into left-to-right reading-order rows.

    Boxes are grouped based on vertical center (cy) proximity relative to
    the median font height of the candidate row. This prevents the classic
    'snowball' chaining bug where an expanding min-y to max-y bounding box
    greedily swallows lower lines on tilted or multi-column layouts.
    """
    if not boxes:
        return []

    # Sort boxes top to bottom by vertical center
    sorted_boxes = sorted(boxes, key=lambda b: b.cy)
    rows: List[List[TextBox]] = []

    for box in sorted_boxes:
        best_row = None
        min_dist = float("inf")
        for r in rows:
            # Check horizontal collision: two boxes cannot belong to the same text line
            # if they are vertically stacked over the same horizontal X span.
            has_x_collision = False
            for existing in r:
                x_overlap = min(box.x2, existing.x2) - max(box.x1, existing.x1)
                shorter_w = min(box.x2 - box.x1, existing.x2 - existing.x1)
                if shorter_w > 0 and x_overlap / shorter_w > 0.35:
                    has_x_collision = True
                    break
            if has_x_collision:
                continue

            mean_cy = sum(b.cy for b in r) / len(r)
            # Use median box height in the row as reference font scale
            sorted_h = sorted(b.height for b in r)
            median_h = sorted_h[len(sorted_h) // 2]
            dist = abs(box.cy - mean_cy)
            if dist <= median_h * y_threshold_ratio and dist < min_dist:
                best_row = r
                min_dist = dist
        if best_row is not None:
            best_row.append(box)
        else:
            rows.append([box])

    # Order rows top-to-bottom, and boxes within each row left-to-right.
    rows.sort(key=lambda row: min(b.y1 for b in row))
    for row in rows:
        row.sort(key=lambda b: b.x1)

    return rows


def _overlaps_row(box: TextBox, row: List[TextBox], y_overlap_ratio: float = 0.5) -> bool:
    """Legacy helper preserved for backwards compatibility."""
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
