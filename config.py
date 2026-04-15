"""
All tunable parameters in one place.
Benchmark-specific overrides go in benchmarks/*/task.py
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DeltaVisionConfig:

    # -- Transition Classification Thresholds --

    # Fraction of pixels that must change to classify as NEW_PAGE
    NEW_PAGE_DIFF_THRESHOLD: float = 0.75

    # Hamming distance between pHashes to classify as NEW_PAGE
    # Max possible: 64 (8x8 hash). Same page ~<10, new page ~>25
    PHASH_DISTANCE_THRESHOLD: int = 20

    # Template match score below which anchor is considered "lost"
    ANCHOR_MATCH_THRESHOLD: float = 0.6

    # Fraction of screen height to use as anchor crop (top nav area)
    ANCHOR_HEIGHT_FRACTION: float = 0.08

    # Override with specific (x1, y1, x2, y2) bbox for anchor
    ANCHOR_BBOX: Optional[Tuple[int, int, int, int]] = None

    # -- Diff Engine Parameters --

    # Pixel brightness change to consider "changed" (0-255)
    DIFF_PIXEL_THRESHOLD: int = 15

    # Morphological dilation kernel size for merging nearby changed regions
    DILATE_KERNEL_SIZE: int = 20

    # Minimum contour area (pixels) to include as a changed region
    MIN_CONTOUR_AREA: int = 200

    # Minimum diff_ratio to consider action "had effect"
    MIN_EFFECT_THRESHOLD: float = 0.005

    # Maximum changed regions to send to model (sorted by size, largest first)
    MAX_REGIONS: int = 6

    # Padding around each bbox crop in pixels
    CROP_PADDING: int = 15

    # -- Agent Loop Parameters --

    MAX_STEPS: int = 50

    # ms to wait after action before capturing t1
    POST_ACTION_WAIT_MS: int = 800

    # Consecutive no-effect steps before forcing full frame refresh
    MAX_NO_EFFECT_RETRIES: int = 3

    # -- Browser --

    BROWSER_WIDTH: int = 1280
    BROWSER_HEIGHT: int = 900
    HEADLESS: bool = False

    # -- OCR / Text Delta (Level 1 optimization) --

    # If a changed region is smaller than this fraction of screen, try OCR first
    OCR_REGION_MAX_FRACTION: float = 0.05

    # Minimum OCR confidence to trust the text extraction
    OCR_MIN_CONFIDENCE: float = 0.7

    # -- Model --

    # Claude API model ID
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # Local model name (HuggingFace ID)
    LOCAL_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Local model quantization: None, "4bit", "8bit"
    LOCAL_QUANTIZATION: Optional[str] = None


# -- Presets --

MCGRAWHILL_CONFIG = DeltaVisionConfig(
    NEW_PAGE_DIFF_THRESHOLD=0.60,
    POST_ACTION_WAIT_MS=1200,
    PHASH_DISTANCE_THRESHOLD=18,
    ANCHOR_HEIGHT_FRACTION=0.06,
    MIN_CONTOUR_AREA=100,
    MAX_REGIONS=8,
)
