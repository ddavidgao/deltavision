"""
Extended cropping utilities.
Core extract_crops lives in diff.py — this module adds context-aware cropping.
"""

from PIL import Image
from typing import List, Tuple


def crop_with_context(
    frame: Image.Image,
    bbox: Tuple[int, int, int, int],
    context_factor: float = 1.5,
) -> Image.Image:
    """
    Crop a region with extra context around it.
    context_factor=1.5 means 50% extra padding on each side.
    """
    w, h = frame.size
    bx, by, bw, bh = bbox

    # Expand by context_factor
    cx = int(bw * (context_factor - 1) / 2)
    cy = int(bh * (context_factor - 1) / 2)

    x1 = max(0, bx - cx)
    y1 = max(0, by - cy)
    x2 = min(w, bx + bw + cx)
    y2 = min(h, by + bh + cy)

    return frame.crop((x1, y1, x2, y2))


def merge_overlapping_bboxes(
    bboxes: List[Tuple[int, int, int, int]], overlap_threshold: float = 0.3
) -> List[Tuple[int, int, int, int]]:
    """
    Merge bounding boxes that overlap significantly.
    Prevents sending redundant crops to the model.
    """
    if not bboxes:
        return []

    merged = list(bboxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i in range(len(merged)):
            if i in used:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                if _iou(current, merged[j]) > overlap_threshold:
                    current = _union(current, merged[j])
                    used.add(j)
                    changed = True
            new_merged.append(current)
            used.add(i)

        merged = new_merged

    return merged


def _iou(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    """Intersection over union of two (x, y, w, h) bboxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _union(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Bounding box union."""
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)
