"""
Core frame differencing. No LLM. No external APIs.
Uses OpenCV for all computation.
"""

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np
from PIL import Image

# --- Token cost model (must match dv_playwright_mcp.py) ---
# A region's cost to send to the model is base + per-tile, where each tile
# covers up to 512×512 pixels of the cropped region. Used by the
# merge-bboxes-for-min-cost optimizer below.
_BBOX_BASE_TOKENS = 85
_BBOX_PER_TILE_TOKENS = 170
_TILE_PX = 512


def _bbox_token_cost(w: int, h: int) -> int:
    tw = max(1, ceil(w / _TILE_PX))
    th = max(1, ceil(h / _TILE_PX))
    return _BBOX_BASE_TOKENS + tw * th * _BBOX_PER_TILE_TOKENS


def _bbox_union(a: tuple[int, int, int, int],
                b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x = min(ax, bx)
    y = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x, y, x2 - x, y2 - y)


def merge_bboxes_for_min_cost(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Greedy rectangle-merge optimizer.

    Given a list of (x, y, w, h) bounding boxes covering changed pixels,
    return a (possibly smaller) list whose total token cost is minimized.

    Pairs are merged when cost(A∪B) < cost(A) + cost(B). Continues until no
    pair-merge would save tokens. The empty-input case returns []; a single
    bbox is returned unchanged.

    Why this exists: without merging, a step with one big sidebar change plus
    5 tiny scattered UI tweaks emits 6 separate crops totaling > 1365 tokens,
    triggering the proxy's full-frame fallback. With merging, the noise
    collapses into the big region and we send 1-2 cheap crops instead.

    Empirically validated on the SF Maps→Sheets run_20 trace: lifted simulated
    savings from 37.2% to 56.2% with no other changes (see
    benchmarks/ablation/simulate_greedy_merge.py).

    O(n³) worst case — only run for small n (≤ 20). Doesn't try harder
    optimizations (k-means clustering, set cover) because the input space is
    tiny by construction (post-dilation contours).
    """
    if len(bboxes) <= 1:
        return list(bboxes)
    boxes = [tuple(b) for b in bboxes]
    while len(boxes) > 1:
        best_save = 0
        best = None
        for i in range(len(boxes)):
            ci = _bbox_token_cost(boxes[i][2], boxes[i][3])
            for j in range(i + 1, len(boxes)):
                cj = _bbox_token_cost(boxes[j][2], boxes[j][3])
                u = _bbox_union(boxes[i], boxes[j])
                cu = _bbox_token_cost(u[2], u[3])
                save = ci + cj - cu
                if save > best_save:
                    best_save = save
                    best = (i, j, u)
        if best is None:
            break
        i, j, u = best
        # Remove originals (j > i, so pop j first) and append the merged box.
        boxes = [b for k, b in enumerate(boxes) if k != i and k != j] + [u]
    return boxes


@dataclass
class DiffResult:
    diff_ratio: float                     # fraction of pixels that changed
    diff_mask: np.ndarray                 # binary mask of changes
    diff_image: Image.Image               # visual diff heatmap (for model input)
    changed_bboxes: list[tuple[int, int, int, int]]  # (x, y, w, h) bounding boxes
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

    bboxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= config.MIN_CONTOUR_AREA:
            bboxes.append((x, y, w, h))

    # Greedy rectangle-merge optimizer.
    # Cost model says base+per-tile per region — a step with N small scattered
    # changes pays N × base before any pixel cost, which can exceed full-frame
    # cost on legitimately small diffs. Merging adjacent/overlapping bboxes
    # whenever the union is cheaper than the pair eliminates this fragmentation
    # blow-up. Disabled via config.BBOX_MERGE_ENABLED=False for ablation.
    if getattr(config, "BBOX_MERGE_ENABLED", True):
        bboxes = merge_bboxes_for_min_cost(bboxes)

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
    bboxes: list[tuple[int, int, int, int]],
    padding: int = 10,
) -> list[dict]:
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
