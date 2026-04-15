"""
Deterministic transition classification.
Decides whether a step is DELTA or NEW_PAGE.
Zero LLM involvement — all decisions are threshold-based CV.

4-layer cascade, ordered cheapest to most expensive:
  1. URL change     (free)
  2. Diff ratio     (fast numpy, already computed)
  3. Perceptual hash (fast PIL)
  4. Anchor template (cv2 template matching)
"""

from enum import Enum
from dataclasses import dataclass
from PIL import Image
import cv2
import numpy as np

from .phash import compute_phash, hamming_distance
from .diff import compute_diff, DiffResult


class TransitionType(Enum):
    DELTA = "delta"
    NEW_PAGE = "new_page"


@dataclass
class ClassificationResult:
    transition: TransitionType
    trigger: str  # "url_change" | "diff_ratio" | "phash" | "anchor_loss" | "none"
    diff_ratio: float
    phash_distance: int
    anchor_score: float


def classify_transition(
    t0: Image.Image,
    t1: Image.Image,
    url_before: str,
    url_after: str,
    anchor_template: Image.Image,
    config,
    diff_result: DiffResult | None = None,
    last_action_type: str | None = None,
) -> ClassificationResult:
    """
    4-layer classification cascade. First match wins.

    Scroll-aware: if the last action was a scroll, the visual diff is expected
    to be large (viewport shifted) but it's NOT a new page. Skip Layers 2-3
    (diff_ratio and pHash) since scrolling invalidates both.
    Layer 4 (anchor match) is also skipped since the anchor may have scrolled
    out of view.
    """
    is_scroll = last_action_type == "scroll"

    # Layer 1: URL change — free, most reliable for traditional nav
    if url_before != url_after:
        return ClassificationResult(
            transition=TransitionType.NEW_PAGE,
            trigger="url_change",
            diff_ratio=0.0,
            phash_distance=0,
            anchor_score=1.0,
        )

    # Compute diff if not already done
    if diff_result is None:
        diff_result = compute_diff(t0, t1, config)

    # Scroll bypass: scrolling shifts the viewport but does NOT change page state.
    # Layers 2-4 would false-positive because the visual content moved. Return
    # DELTA immediately — the agent loop will re-anchor after scroll.
    if is_scroll:
        return ClassificationResult(
            transition=TransitionType.DELTA,
            trigger="scroll_bypass",
            diff_ratio=diff_result.diff_ratio,
            phash_distance=0,
            anchor_score=1.0,
        )

    # Layer 2: Diff ratio — covers SPA nav, full content replacement, reloads
    if diff_result.diff_ratio > config.NEW_PAGE_DIFF_THRESHOLD:
        return ClassificationResult(
            transition=TransitionType.NEW_PAGE,
            trigger="diff_ratio",
            diff_ratio=diff_result.diff_ratio,
            phash_distance=0,
            anchor_score=1.0,
        )

    # Layer 3: Perceptual hash distance
    # Animation guard: when diff_ratio is low (page looks ~the same), require
    # a higher pHash distance to trigger. Animated elements (spinners, fades)
    # produce small diff but elevated pHash — real SPA nav has BOTH high.
    phash_t0 = compute_phash(t0)
    phash_t1 = compute_phash(t1)
    distance = hamming_distance(phash_t0, phash_t1)

    phash_threshold = config.PHASH_DISTANCE_THRESHOLD
    low_diff_floor = getattr(config, 'PHASH_LOW_DIFF_FLOOR', 0.15)
    animation_margin = getattr(config, 'PHASH_ANIMATION_MARGIN', 5)
    if diff_result.diff_ratio < low_diff_floor:
        phash_threshold += animation_margin

    if distance > phash_threshold:
        return ClassificationResult(
            transition=TransitionType.NEW_PAGE,
            trigger="phash",
            diff_ratio=diff_result.diff_ratio,
            phash_distance=distance,
            anchor_score=1.0,
        )

    # Layer 4: Anchor template match — catches SPA nav where URL stays same
    anchor_score = match_anchor(t1, anchor_template, config)

    if anchor_score < config.ANCHOR_MATCH_THRESHOLD:
        return ClassificationResult(
            transition=TransitionType.NEW_PAGE,
            trigger="anchor_loss",
            diff_ratio=diff_result.diff_ratio,
            phash_distance=distance,
            anchor_score=anchor_score,
        )

    # All checks passed — DELTA
    return ClassificationResult(
        transition=TransitionType.DELTA,
        trigger="none",
        diff_ratio=diff_result.diff_ratio,
        phash_distance=distance,
        anchor_score=anchor_score,
    )


def match_anchor(frame: Image.Image, template: Image.Image, config) -> float:
    """
    OpenCV template matching against a stable anchor crop.
    Returns normalized match score [0, 1]. Higher = better match.

    Falls back to MSE comparison when template has near-zero variance
    (TM_CCOEFF_NORMED is undefined for constant-value templates).
    """
    frame_gray = np.array(frame.convert("L"))
    tmpl_gray = np.array(template.convert("L"))

    # Template can't be larger than frame
    if (
        tmpl_gray.shape[0] > frame_gray.shape[0]
        or tmpl_gray.shape[1] > frame_gray.shape[1]
    ):
        return 0.0

    # Low-variance fallback: TM_CCOEFF_NORMED divides by std dev,
    # producing garbage for uniform/near-uniform templates.
    # Use normalized MSE instead.
    if tmpl_gray.std() < 2.0:
        th, tw = tmpl_gray.shape
        # Extract best-match region via TM_SQDIFF (works for constant templates)
        result = cv2.matchTemplate(frame_gray, tmpl_gray, cv2.TM_SQDIFF)
        _, min_val, _, _ = cv2.minMaxLoc(result)
        # Normalize: 0 = perfect match, 255^2 = worst. Invert to [0, 1].
        max_possible = 255.0 * 255.0 * th * tw
        score = 1.0 - (min_val / max_possible) if max_possible > 0 else 1.0
        return score

    result = cv2.matchTemplate(frame_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def extract_anchor(full_frame: Image.Image, config) -> Image.Image:
    """
    Extract anchor template from a full frame.
    Default: top strip (navigation bar area).
    """
    w, h = full_frame.size
    if config.ANCHOR_BBOX:
        return full_frame.crop(config.ANCHOR_BBOX)
    anchor_h = int(h * config.ANCHOR_HEIGHT_FRACTION)
    return full_frame.crop((0, 0, w, anchor_h))
