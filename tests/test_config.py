"""
Config validation tests. Every __post_init__ branch needs coverage —
a bad config silently propagates into CV behavior that's hard to debug.
"""

import pytest

from config import DeltaVisionConfig, ConfigError, MCGRAWHILL_CONFIG


class TestDefaults:
    def test_defaults_construct(self):
        # Should not raise.
        cfg = DeltaVisionConfig()
        assert cfg.MAX_STEPS == 50
        assert cfg.PHASH_DISTANCE_THRESHOLD == 20

    def test_mcgrawhill_preset_valid(self):
        # The preset must validate against __post_init__.
        assert MCGRAWHILL_CONFIG.NEW_PAGE_DIFF_THRESHOLD == 0.60
        assert MCGRAWHILL_CONFIG.PHASH_DISTANCE_THRESHOLD == 18


class TestFractions:
    @pytest.mark.parametrize("field", [
        "NEW_PAGE_DIFF_THRESHOLD",
        "ANCHOR_MATCH_THRESHOLD",
        "ANCHOR_HEIGHT_FRACTION",
        "MIN_EFFECT_THRESHOLD",
        "OCR_REGION_MAX_FRACTION",
        "OCR_MIN_CONFIDENCE",
        "PHASH_LOW_DIFF_FLOOR",
    ])
    def test_fraction_above_one_rejected(self, field):
        with pytest.raises(ConfigError, match=field):
            DeltaVisionConfig(**{field: 1.5})

    @pytest.mark.parametrize("field", [
        "NEW_PAGE_DIFF_THRESHOLD",
        "ANCHOR_MATCH_THRESHOLD",
    ])
    def test_fraction_negative_rejected(self, field):
        with pytest.raises(ConfigError, match=field):
            DeltaVisionConfig(**{field: -0.1})

    def test_fraction_at_boundary_accepted(self):
        # 0 and 1 inclusive
        DeltaVisionConfig(NEW_PAGE_DIFF_THRESHOLD=0.0)
        DeltaVisionConfig(NEW_PAGE_DIFF_THRESHOLD=1.0)


class TestPHashThresholds:
    def test_phash_over_64_rejected(self):
        with pytest.raises(ConfigError, match="PHASH_DISTANCE_THRESHOLD"):
            DeltaVisionConfig(PHASH_DISTANCE_THRESHOLD=65)

    def test_phash_negative_rejected(self):
        with pytest.raises(ConfigError, match="PHASH_DISTANCE_THRESHOLD"):
            DeltaVisionConfig(PHASH_DISTANCE_THRESHOLD=-1)

    def test_phash_at_64_accepted(self):
        DeltaVisionConfig(PHASH_DISTANCE_THRESHOLD=64)

    def test_animation_margin_negative_rejected(self):
        with pytest.raises(ConfigError, match="PHASH_ANIMATION_MARGIN"):
            DeltaVisionConfig(PHASH_ANIMATION_MARGIN=-5)


class TestPositiveInts:
    @pytest.mark.parametrize("field", [
        "DIFF_PIXEL_THRESHOLD", "DILATE_KERNEL_SIZE", "MIN_CONTOUR_AREA",
        "MAX_REGIONS", "CROP_PADDING", "MAX_STEPS",
        "POST_ACTION_WAIT_MS", "MAX_NO_EFFECT_RETRIES",
        "BROWSER_WIDTH", "BROWSER_HEIGHT",
    ])
    def test_negative_rejected(self, field):
        with pytest.raises(ConfigError, match=field):
            DeltaVisionConfig(**{field: -1})

    @pytest.mark.parametrize("field", [
        "MIN_CONTOUR_AREA", "POST_ACTION_WAIT_MS",
    ])
    def test_float_rejected(self, field):
        with pytest.raises(ConfigError, match=field):
            DeltaVisionConfig(**{field: 100.5})

    def test_diff_pixel_over_255_rejected(self):
        with pytest.raises(ConfigError, match="DIFF_PIXEL_THRESHOLD"):
            DeltaVisionConfig(DIFF_PIXEL_THRESHOLD=256)

    def test_max_steps_zero_rejected(self):
        with pytest.raises(ConfigError, match="MAX_STEPS"):
            DeltaVisionConfig(MAX_STEPS=0)

    def test_max_regions_zero_rejected(self):
        with pytest.raises(ConfigError, match="MAX_REGIONS"):
            DeltaVisionConfig(MAX_REGIONS=0)


class TestQuantization:
    @pytest.mark.parametrize("v", [None, "4bit", "8bit"])
    def test_valid_quantization(self, v):
        cfg = DeltaVisionConfig(LOCAL_QUANTIZATION=v)
        assert cfg.LOCAL_QUANTIZATION == v

    @pytest.mark.parametrize("v", ["16bit", "2bit", "int4", "false", ""])
    def test_invalid_quantization_rejected(self, v):
        with pytest.raises(ConfigError, match="LOCAL_QUANTIZATION"):
            DeltaVisionConfig(LOCAL_QUANTIZATION=v)


class TestAnchorBBox:
    def test_bbox_none_allowed(self):
        # Default anchor strategy is top-strip; no bbox needed.
        DeltaVisionConfig(ANCHOR_BBOX=None)

    def test_bbox_valid(self):
        DeltaVisionConfig(ANCHOR_BBOX=(0, 0, 100, 50))

    def test_bbox_wrong_length_rejected(self):
        with pytest.raises(ConfigError, match="ANCHOR_BBOX"):
            DeltaVisionConfig(ANCHOR_BBOX=(0, 0, 100))

    def test_bbox_inverted_x_rejected(self):
        with pytest.raises(ConfigError, match="ANCHOR_BBOX"):
            DeltaVisionConfig(ANCHOR_BBOX=(100, 0, 50, 50))

    def test_bbox_inverted_y_rejected(self):
        with pytest.raises(ConfigError, match="ANCHOR_BBOX"):
            DeltaVisionConfig(ANCHOR_BBOX=(0, 100, 100, 50))

    def test_bbox_degenerate_rejected(self):
        """Zero-width/height bbox would crash cv2.matchTemplate."""
        with pytest.raises(ConfigError, match="ANCHOR_BBOX"):
            DeltaVisionConfig(ANCHOR_BBOX=(10, 10, 10, 20))
