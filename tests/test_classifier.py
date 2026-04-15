"""
Tests for the transition classifier.
Each layer of the cascade tested in isolation.
"""

import numpy as np
from PIL import Image
import pytest

from vision.classifier import (
    classify_transition,
    extract_anchor,
    match_anchor,
    TransitionType,
)
from config import DeltaVisionConfig


@pytest.fixture
def config():
    return DeltaVisionConfig()


def make_solid(color: int, size=(1280, 900)) -> Image.Image:
    arr = np.full((size[1], size[0]), color, dtype=np.uint8)
    return Image.fromarray(arr)


def make_with_header(header_color: int, body_color: int, size=(1280, 900)) -> Image.Image:
    """Image with a distinct header strip (anchor region)."""
    arr = np.full((size[1], size[0]), body_color, dtype=np.uint8)
    header_h = int(size[1] * 0.08)
    arr[:header_h, :] = header_color
    return Image.fromarray(arr)


class TestURLChange:
    def test_different_urls_triggers_new_page(self, config):
        img = make_solid(128)
        anchor = extract_anchor(img, config)
        result = classify_transition(
            img, img, "https://a.com", "https://b.com", anchor, config
        )
        assert result.transition == TransitionType.NEW_PAGE
        assert result.trigger == "url_change"

    def test_same_url_passes(self, config):
        img = make_solid(128)
        anchor = extract_anchor(img, config)
        result = classify_transition(
            img, img, "https://a.com", "https://a.com", anchor, config
        )
        assert result.transition == TransitionType.DELTA


class TestDiffRatio:
    def test_high_diff_triggers_new_page(self, config):
        t0 = make_solid(0)
        t1 = make_solid(255)
        anchor = extract_anchor(t0, config)
        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config
        )
        assert result.transition == TransitionType.NEW_PAGE
        assert result.trigger == "diff_ratio"


class TestAnchorMatch:
    def test_anchor_loss_triggers_new_page(self, config):
        """Header present in t0 but gone in t1 → anchor loss."""
        t0 = make_with_header(250, 10)
        t1 = make_with_header(10, 10)  # header now same as body = "lost"
        anchor = extract_anchor(t0, config)

        # Force diff_ratio below threshold so we reach anchor check
        config.NEW_PAGE_DIFF_THRESHOLD = 0.99
        config.PHASH_DISTANCE_THRESHOLD = 60

        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config
        )
        assert result.transition == TransitionType.NEW_PAGE
        assert result.trigger == "anchor_loss"

    def test_anchor_present_passes(self, config):
        """Same header in both frames → anchor match."""
        t0 = make_with_header(200, 50)
        t1 = make_with_header(200, 80)  # body changed but header same
        anchor = extract_anchor(t0, config)

        config.NEW_PAGE_DIFF_THRESHOLD = 0.99
        config.PHASH_DISTANCE_THRESHOLD = 60

        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config
        )
        assert result.transition == TransitionType.DELTA


class TestExtractAnchor:
    def test_default_top_strip(self, config):
        img = make_solid(128, size=(1280, 900))
        anchor = extract_anchor(img, config)
        expected_h = int(900 * config.ANCHOR_HEIGHT_FRACTION)
        assert anchor.size == (1280, expected_h)

    def test_custom_bbox(self, config):
        config.ANCHOR_BBOX = (10, 10, 100, 50)
        img = make_solid(128, size=(1280, 900))
        anchor = extract_anchor(img, config)
        assert anchor.size == (90, 40)  # (100-10, 50-10)
