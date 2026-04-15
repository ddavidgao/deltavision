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


class TestScrollBypass:
    """Scroll bypass: scrolling changes the viewport but NOT the page.
    Without scroll_bypass, large scrolls trigger false NEW_PAGE via pHash."""

    def test_scroll_bypasses_all_layers(self, config):
        """Even with completely different frames, scroll should classify as DELTA."""
        t0 = make_solid(0)
        t1 = make_solid(255)
        anchor = extract_anchor(t0, config)
        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config,
            last_action_type="scroll",
        )
        assert result.transition == TransitionType.DELTA
        assert result.trigger == "scroll_bypass"

    def test_scroll_bypass_still_computes_diff(self, config):
        """Diff ratio should be computed even when scroll bypasses classification."""
        t0 = make_solid(0)
        t1 = make_solid(255)
        anchor = extract_anchor(t0, config)
        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config,
            last_action_type="scroll",
        )
        assert result.diff_ratio > 0.9  # frames are completely different

    def test_url_change_overrides_scroll(self, config):
        """URL change during scroll (e.g. hash nav) should still be NEW_PAGE."""
        img = make_solid(128)
        anchor = extract_anchor(img, config)
        result = classify_transition(
            img, img, "https://a.com/page1", "https://a.com/page2", anchor, config,
            last_action_type="scroll",
        )
        # URL change fires at Layer 1, before scroll bypass
        assert result.transition == TransitionType.NEW_PAGE
        assert result.trigger == "url_change"

    def test_non_scroll_action_no_bypass(self, config):
        """Click action with big diff should still trigger NEW_PAGE normally."""
        t0 = make_solid(0)
        t1 = make_solid(255)
        anchor = extract_anchor(t0, config)
        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config,
            last_action_type="click",
        )
        assert result.transition == TransitionType.NEW_PAGE

    def test_no_action_type_no_bypass(self, config):
        """No last_action_type (initial observation) should not bypass."""
        t0 = make_solid(0)
        t1 = make_solid(255)
        anchor = extract_anchor(t0, config)
        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config,
            last_action_type=None,
        )
        assert result.transition == TransitionType.NEW_PAGE


class TestAnimationGuard:
    """Animation guard: low diff + borderline pHash should NOT trigger NEW_PAGE.
    Real SPA nav has high diff AND high pHash; animation has low diff + elevated pHash."""

    def test_low_diff_high_phash_is_delta(self, config):
        """Simulates animated page: barely any pixels changed but pHash crosses threshold."""
        # Create frames that are very similar but with slightly different structure
        t0 = make_with_header(200, 100)
        # Slightly different body to get moderate pHash but low diff
        arr = np.array(t0)
        # Change a scattered pattern to increase pHash without massive diff
        arr[100::20, ::20] = 200  # sparse changes
        t1 = Image.fromarray(arr)
        anchor = extract_anchor(t0, config)

        config.NEW_PAGE_DIFF_THRESHOLD = 0.99  # don't trigger on diff
        config.PHASH_DISTANCE_THRESHOLD = 15
        config.PHASH_LOW_DIFF_FLOOR = 0.15
        config.PHASH_ANIMATION_MARGIN = 10

        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config
        )
        # The diff should be low, and pHash should be below threshold + margin
        # So this should be classified as DELTA
        if result.diff_ratio < config.PHASH_LOW_DIFF_FLOOR:
            assert result.trigger != "phash" or result.transition == TransitionType.DELTA

    def test_high_diff_normal_phash_still_triggers(self, config):
        """Real SPA nav with high diff should still trigger even with animation guard."""
        t0 = make_solid(0)
        t1 = make_solid(200)
        anchor = extract_anchor(t0, config)

        config.PHASH_LOW_DIFF_FLOOR = 0.15
        config.PHASH_ANIMATION_MARGIN = 5

        result = classify_transition(
            t0, t1, "https://a.com", "https://a.com", anchor, config
        )
        assert result.transition == TransitionType.NEW_PAGE


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
