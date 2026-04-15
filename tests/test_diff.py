"""
Tests for the frame differencing engine.
Uses synthetic images — no browser needed.
"""

import numpy as np
from PIL import Image
import pytest

from vision.diff import compute_diff, extract_crops
from config import DeltaVisionConfig


@pytest.fixture
def config():
    return DeltaVisionConfig()


def make_solid(color: int, size=(1280, 900)) -> Image.Image:
    """Create a solid grayscale image."""
    arr = np.full((size[1], size[0]), color, dtype=np.uint8)
    return Image.fromarray(arr)


def make_with_rect(
    bg: int, rect_color: int, rect: tuple, size=(1280, 900)
) -> Image.Image:
    """Create image with a colored rectangle."""
    arr = np.full((size[1], size[0]), bg, dtype=np.uint8)
    x, y, w, h = rect
    arr[y : y + h, x : x + w] = rect_color
    return Image.fromarray(arr)


class TestComputeDiff:
    def test_identical_frames(self, config):
        img = make_solid(128)
        result = compute_diff(img, img, config)
        assert result.diff_ratio == 0.0
        assert not result.action_had_effect
        assert len(result.changed_bboxes) == 0

    def test_completely_different(self, config):
        t0 = make_solid(0)
        t1 = make_solid(255)
        result = compute_diff(t0, t1, config)
        assert result.diff_ratio > 0.9
        assert result.action_had_effect

    def test_small_change_detected(self, config):
        """A 50x50 rect on 1280x900 is ~0.2% of pixels — below effect threshold
        but bboxes should still be found for the model to reason about."""
        t0 = make_solid(128)
        t1 = make_with_rect(128, 255, (100, 100, 50, 50))
        result = compute_diff(t0, t1, config)
        assert len(result.changed_bboxes) >= 1
        bbox = result.changed_bboxes[0]
        assert bbox[0] <= 110 and bbox[1] <= 110

    def test_large_change_has_effect(self, config):
        """A region large enough to cross the MIN_EFFECT_THRESHOLD."""
        t0 = make_solid(128)
        t1 = make_with_rect(128, 255, (100, 100, 200, 200))
        result = compute_diff(t0, t1, config)
        assert result.action_had_effect
        assert result.diff_ratio > config.MIN_EFFECT_THRESHOLD

    def test_subthreshold_noise_filtered(self, config):
        """Pixel changes below DIFF_PIXEL_THRESHOLD should be invisible."""
        t0 = make_solid(128)
        arr1 = np.full((900, 1280), 128, dtype=np.uint8)
        # Change a pixel by only 5 — below the threshold of 15
        arr1[50, 50] = 133
        t1 = Image.fromarray(arr1)
        result = compute_diff(t0, t1, config)
        assert result.diff_ratio == 0.0
        assert len(result.changed_bboxes) == 0

    def test_max_regions_cap(self, config):
        config.MAX_REGIONS = 2
        t0 = make_solid(128)
        # Create multiple distinct changed regions
        arr1 = np.full((900, 1280), 128, dtype=np.uint8)
        for i in range(5):
            x = i * 200 + 50
            arr1[100:150, x : x + 50] = 255
        t1 = Image.fromarray(arr1)
        result = compute_diff(t0, t1, config)
        assert len(result.changed_bboxes) <= 2


class TestExtractCrops:
    def test_basic_crop(self):
        t0 = make_solid(128)
        t1 = make_with_rect(128, 255, (100, 100, 50, 50))
        bboxes = [(100, 100, 50, 50)]
        crops = extract_crops(t0, t1, bboxes, padding=10)
        assert len(crops) == 1
        assert crops[0]["bbox"] == (100, 100, 50, 50)
        assert crops[0]["crop_before"].size[0] == 70  # 50 + 2*10 padding
        assert crops[0]["crop_after"].size[0] == 70

    def test_crop_clamped_to_image_bounds(self):
        t0 = make_solid(128, size=(200, 200))
        t1 = make_solid(128, size=(200, 200))
        bboxes = [(0, 0, 30, 30)]  # corner — padding would go negative
        crops = extract_crops(t0, t1, bboxes, padding=20)
        assert len(crops) == 1
        # Should clamp to (0,0) not (-20,-20)
        assert crops[0]["crop_before"].size[0] == 50  # 30 + 20 (only right padding)
