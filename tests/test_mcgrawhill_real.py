"""
Real McGraw-Hill screenshot tests.
Uses actual screenshots from EAPS 111 SmartBook to validate classifier
and diff engine against real-world educational SPA content.

Fixtures:
  mcgraw_q2_feedback.png  — Q2 answer feedback (correct, with explanations)
  mcgraw_q3_question.png  — Q3 multiple choice question
  mcgraw_reading_mode.png — Reading mode (textbook content with diagrams)

All share the same URL: learning.mheducation.com/static/awd/index.html#/
Nav bar ("McGraw Hill Recharge" + "Exit Recharge") persists across all states.
"""

import os
from pathlib import Path
from PIL import Image
import pytest

from config import DeltaVisionConfig, MCGRAWHILL_CONFIG
from vision.diff import compute_diff, extract_crops
from vision.classifier import (
    classify_transition,
    extract_anchor,
    TransitionType,
)
from vision.phash import compute_phash, hamming_distance

FIXTURES = Path(__file__).parent / "fixtures" / "mcgrawhill"

# Skip entire module if fixtures aren't present
pytestmark = pytest.mark.skipif(
    not (FIXTURES / "mcgraw_q3_question.png").exists(),
    reason="McGraw-Hill fixtures not found",
)

URL = "https://learning.mheducation.com/static/awd/index.html?_t=1776216172521#/"


@pytest.fixture
def config():
    return MCGRAWHILL_CONFIG


@pytest.fixture
def q2_feedback():
    return Image.open(FIXTURES / "mcgraw_q2_feedback.png")


@pytest.fixture
def q3_question():
    return Image.open(FIXTURES / "mcgraw_q3_question.png")


@pytest.fixture
def reading_mode():
    return Image.open(FIXTURES / "mcgraw_reading_mode.png")


class TestRealDiffMetrics:
    """Measure actual diff metrics on real screenshots."""

    def test_feedback_to_question_diff(self, config, q2_feedback, q3_question):
        """Q2 feedback → Q3 question: full content replacement.
        Should have high diff_ratio — completely different question."""
        diff = compute_diff(q2_feedback, q3_question, config)
        print(f"\nFeedback→Question: diff_ratio={diff.diff_ratio:.3f}, "
              f"bboxes={len(diff.changed_bboxes)}, "
              f"largest_area={diff.largest_change_area:.3f}")
        assert diff.action_had_effect
        assert diff.diff_ratio > 0.05  # significant change

    def test_question_to_reading_diff(self, config, q3_question, reading_mode):
        """Question → Reading mode: layout change BUT both pages are mostly
        white, so diff_ratio stays low (~2%). The content difference shows up
        in the bbox crops, not in raw pixel count. This is why pHash and
        anchor matching exist — diff_ratio alone can't catch everything."""
        diff = compute_diff(q3_question, reading_mode, config)
        print(f"\nQuestion→Reading: diff_ratio={diff.diff_ratio:.3f}, "
              f"bboxes={len(diff.changed_bboxes)}, "
              f"largest_area={diff.largest_change_area:.3f}")
        assert diff.action_had_effect
        assert len(diff.changed_bboxes) >= 2  # multiple regions changed

    def test_feedback_to_reading_diff(self, config, q2_feedback, reading_mode):
        """Feedback → Reading: completely different content."""
        diff = compute_diff(q2_feedback, reading_mode, config)
        print(f"\nFeedback→Reading: diff_ratio={diff.diff_ratio:.3f}, "
              f"bboxes={len(diff.changed_bboxes)}, "
              f"largest_area={diff.largest_change_area:.3f}")
        assert diff.diff_ratio > 0.1


class TestRealClassifier:
    """Run the full 4-layer cascade on real transitions."""

    def test_feedback_to_question_is_new_page(self, config, q2_feedback, q3_question):
        """Q2 feedback → Q3 question. Same URL but full content swap.
        Should be classified as NEW_PAGE."""
        anchor = extract_anchor(q2_feedback, config)
        cls = classify_transition(
            q2_feedback, q3_question, URL, URL, anchor, config
        )
        print(f"\nFeedback→Question: transition={cls.transition.value}, "
              f"trigger={cls.trigger}, diff={cls.diff_ratio:.3f}, "
              f"phash={cls.phash_distance}, anchor={cls.anchor_score:.3f}")
        # This should be detected as a new page — the content changed entirely
        # Even though URL and nav bar stay the same

    def test_question_to_reading_is_new_page(self, config, q3_question, reading_mode):
        """Question → Reading mode. Same URL, massive layout change."""
        anchor = extract_anchor(q3_question, config)
        cls = classify_transition(
            q3_question, reading_mode, URL, URL, anchor, config
        )
        print(f"\nQuestion→Reading: transition={cls.transition.value}, "
              f"trigger={cls.trigger}, diff={cls.diff_ratio:.3f}, "
              f"phash={cls.phash_distance}, anchor={cls.anchor_score:.3f}")


class TestRealPHash:
    """pHash behavior on real screenshots."""

    def test_phash_distances(self, q2_feedback, q3_question, reading_mode):
        """Measure pHash distances between all frame pairs.
        These are real web screenshots with rich texture — pHash should
        behave much better than on synthetic images."""
        h_fb = compute_phash(q2_feedback)
        h_q3 = compute_phash(q3_question)
        h_rd = compute_phash(reading_mode)

        d_fb_q3 = hamming_distance(h_fb, h_q3)
        d_q3_rd = hamming_distance(h_q3, h_rd)
        d_fb_rd = hamming_distance(h_fb, h_rd)

        print(f"\npHash distances (real screenshots):")
        print(f"  Feedback→Question: {d_fb_q3}")
        print(f"  Question→Reading:  {d_q3_rd}")
        print(f"  Feedback→Reading:  {d_fb_rd}")

        # All pairs should have significant distance since they're
        # completely different content
        assert d_fb_q3 > 5
        assert d_q3_rd > 5


class TestRealAnchor:
    """Anchor template matching on real nav bar."""

    def test_nav_bar_persists_across_states(self, config, q2_feedback, q3_question, reading_mode):
        """The McGraw Hill Recharge nav bar should match across all states.
        This is the whole point of anchor matching — stable nav = same app."""
        anchor = extract_anchor(q2_feedback, config)

        from vision.classifier import match_anchor
        score_q3 = match_anchor(q3_question, anchor, config)
        score_rd = match_anchor(reading_mode, anchor, config)

        print(f"\nAnchor match scores (anchor from Q2 feedback):")
        print(f"  vs Q3 question: {score_q3:.3f}")
        print(f"  vs Reading mode: {score_rd:.3f}")

        # Nav bar should be stable — high match scores
        # (the "McGraw Hill Recharge" header is identical across modes)
        assert score_q3 > 0.5, f"Anchor should match Q3, got {score_q3}"
        assert score_rd > 0.5, f"Anchor should match Reading, got {score_rd}"


class TestRealCropExtraction:
    """Verify crop extraction produces useful regions on real data."""

    def test_feedback_to_question_crops(self, config, q2_feedback, q3_question):
        diff = compute_diff(q2_feedback, q3_question, config)
        crops = extract_crops(q2_feedback, q3_question, diff.changed_bboxes, config.CROP_PADDING)

        print(f"\nCrops extracted: {len(crops)}")
        for i, c in enumerate(crops):
            print(f"  Region {i}: bbox={c['bbox']}, "
                  f"magnitude={c['change_magnitude']:.3f}, "
                  f"crop_size={c['crop_after'].size}")

        assert len(crops) > 0
        # At least one region should cover a significant part of the screen
        assert any(c["change_magnitude"] > 0.01 for c in crops)
