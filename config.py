"""
All tunable parameters in one place.
Benchmark-specific overrides go in benchmarks/*/task.py

Every field has a validator in __post_init__. This catches fat-finger config
errors at construction time rather than letting them cause weird CV behavior
much later — e.g. a negative PHASH_DISTANCE_THRESHOLD would make every frame
classify as NEW_PAGE, wasting tokens silently.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


class ConfigError(ValueError):
    """Raised when a DeltaVisionConfig is constructed with invalid values."""


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

    # Minimum diff_ratio required for pHash to trigger NEW_PAGE.
    # Animated pages have low diff (<15%) but elevated pHash due to
    # subtle motion. Require pHash to exceed threshold by PHASH_ANIMATION_MARGIN
    # when diff is below this floor.
    PHASH_LOW_DIFF_FLOOR: float = 0.15
    PHASH_ANIMATION_MARGIN: int = 5  # extra pHash distance needed when diff < floor

    # -- Ablation --

    # Force full-frame observations (disable delta gating). For controlled comparison.
    FORCE_FULL_FRAME: bool = False

    # -- Model --

    # Claude API model ID
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # Local model name (HuggingFace ID)
    LOCAL_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Local model quantization: None, "4bit", "8bit"
    LOCAL_QUANTIZATION: Optional[str] = None

    def __post_init__(self):
        # Fractions must live in [0, 1].
        for name in ("NEW_PAGE_DIFF_THRESHOLD", "ANCHOR_MATCH_THRESHOLD",
                     "ANCHOR_HEIGHT_FRACTION", "MIN_EFFECT_THRESHOLD",
                     "OCR_REGION_MAX_FRACTION", "OCR_MIN_CONFIDENCE",
                     "PHASH_LOW_DIFF_FLOOR"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ConfigError(f"{name} must be in [0, 1], got {v}")

        # pHash Hamming distance is on 8x8 hash = 64 bits.
        if not (0 <= self.PHASH_DISTANCE_THRESHOLD <= 64):
            raise ConfigError(
                f"PHASH_DISTANCE_THRESHOLD must be in [0, 64], got {self.PHASH_DISTANCE_THRESHOLD}"
            )
        if self.PHASH_ANIMATION_MARGIN < 0:
            raise ConfigError(
                f"PHASH_ANIMATION_MARGIN must be >= 0, got {self.PHASH_ANIMATION_MARGIN}"
            )

        # Pixel / count params must be positive integers.
        for name in ("DIFF_PIXEL_THRESHOLD", "DILATE_KERNEL_SIZE",
                     "MIN_CONTOUR_AREA", "MAX_REGIONS", "CROP_PADDING",
                     "MAX_STEPS", "POST_ACTION_WAIT_MS",
                     "MAX_NO_EFFECT_RETRIES", "BROWSER_WIDTH", "BROWSER_HEIGHT"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0:
                raise ConfigError(f"{name} must be a non-negative int, got {v!r}")

        if self.DIFF_PIXEL_THRESHOLD > 255:
            raise ConfigError(
                f"DIFF_PIXEL_THRESHOLD is a 0-255 brightness delta, got {self.DIFF_PIXEL_THRESHOLD}"
            )

        if self.MAX_STEPS < 1:
            raise ConfigError(f"MAX_STEPS must be >= 1, got {self.MAX_STEPS}")

        if self.MAX_REGIONS < 1:
            raise ConfigError(f"MAX_REGIONS must be >= 1, got {self.MAX_REGIONS}")

        # Quantization must be one of the known values.
        if self.LOCAL_QUANTIZATION not in (None, "4bit", "8bit"):
            raise ConfigError(
                f"LOCAL_QUANTIZATION must be None, '4bit', or '8bit', "
                f"got {self.LOCAL_QUANTIZATION!r}"
            )

        # Anchor bbox coherence: if given, must be (x1, y1, x2, y2) with x2>x1, y2>y1.
        if self.ANCHOR_BBOX is not None:
            if len(self.ANCHOR_BBOX) != 4:
                raise ConfigError(f"ANCHOR_BBOX must have 4 elements, got {self.ANCHOR_BBOX}")
            x1, y1, x2, y2 = self.ANCHOR_BBOX
            if x2 <= x1 or y2 <= y1:
                raise ConfigError(
                    f"ANCHOR_BBOX must satisfy x2>x1 and y2>y1, got {self.ANCHOR_BBOX}"
                )


# -- Presets --

MCGRAWHILL_CONFIG = DeltaVisionConfig(
    NEW_PAGE_DIFF_THRESHOLD=0.60,
    POST_ACTION_WAIT_MS=1200,
    PHASH_DISTANCE_THRESHOLD=18,
    ANCHOR_HEIGHT_FRACTION=0.06,
    MIN_CONTOUR_AREA=100,
    MAX_REGIONS=8,
)
