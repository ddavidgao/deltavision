"""
Core frame differencing. No LLM. No external APIs.
Uses OpenCV for all computation.
"""

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DiffResult:
    diff_ratio: float                     # fraction of pixels that changed
    diff_mask: np.ndarray                 # binary mask of changes
    diff_image: Image.Image               # visual diff heatmap (for model input)
    changed_bboxes: List[Tuple[int, int, int, int]]  # (x, y, w, h) bounding boxes
    largest_change_area: float            # area of largest region as fraction of screen
    action_had_effect: bool               # True if diff_ratio > MIN_EFFECT_THRESHOLD


def compute_diff(t0: Image.Image, t1: Image.Image, config) -> DiffResult:
    """
    Pixel-level difference between two frames.

    Pipeline:
    1. Grayscale conversion
    2. Absolute difference
    3. Gaussian blur (noise reduction)
    4. Binary threshold
    5. Morphological dilation (merge nearby regions)
    6. Contour detection → bounding boxes
    7. Filter by minimum area
    """
    arr0 = np.array(t0.convert("L"))
    arr1 = np.array(t1.convert("L"))

    diff = cv2.absdiff(arr0, arr1)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, config.DIFF_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.DILATE_KERNEL_SIZE, config.DILATE_KERNEL_SIZE)
    )
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_pixels = arr0.size
    changed_pixels = int(np.count_nonzero(thresh))
    diff_ratio = changed_pixels / total_pixels

    bboxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= config.MIN_CONTOUR_AREA:
            bboxes.append((x, y, w, h))

    # Largest regions first — model sees the most important changes first
    bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    # Build visual diff heatmap for model
    diff_colored = cv2.applyColorMap(blurred, cv2.COLORMAP_HOT)
    diff_pil = Image.fromarray(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB))

    largest = max((b[2] * b[3] for b in bboxes), default=0)

    return DiffResult(
        diff_ratio=diff_ratio,
        diff_mask=thresh,
        diff_image=diff_pil,
        changed_bboxes=bboxes[: config.MAX_REGIONS],
        largest_change_area=largest / total_pixels,
        action_had_effect=diff_ratio > config.MIN_EFFECT_THRESHOLD,
    )


def extract_crops(
    t0: Image.Image,
    t1: Image.Image,
    bboxes: List[Tuple[int, int, int, int]],
    padding: int = 10,
) -> List[dict]:
    """
    Extract before/after crops for each changed bounding box.
    Returns list of dicts with crop_before, crop_after, bbox, change_magnitude.
    """
    crops = []
    w, h = t0.size
    for (bx, by, bw, bh) in bboxes:
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(w, bx + bw + padding)
        y2 = min(h, by + bh + padding)
        crops.append(
            {
                "bbox": (bx, by, bw, bh),
                "crop_before": t0.crop((x1, y1, x2, y2)),
                "crop_after": t1.crop((x1, y1, x2, y2)),
                "change_magnitude": (bw * bh) / (w * h),
            }
        )
    return crops
