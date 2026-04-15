"""
Integration tests — mock model + mock browser, real vision pipeline.
Tests the full observation flow without network or API calls.
"""

import asyncio
import numpy as np
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, PropertyMock
import pytest

from config import DeltaVisionConfig
from vision.diff import compute_diff, extract_crops
from vision.classifier import classify_transition, extract_anchor, TransitionType
from observation.builder import build_observation
from observation.types import FullFrameObservation, DeltaObservation
from agent.state import AgentState
from agent.actions import Action, ActionType, parse_action
from model.base import ModelResponse


@pytest.fixture
def config():
    return DeltaVisionConfig()


# -- Observation builder tests --

class TestObservationBuilder:
    def test_build_full_frame(self):
        img = Image.new("RGB", (100, 100), color="red")
        obs = build_observation(
            obs_type="full_frame",
            task="test task",
            step=0,
            last_action=None,
            frame=img,
            url="https://example.com",
            trigger_reason="initial",
        )
        assert isinstance(obs, FullFrameObservation)
        assert obs.obs_type == "full_frame"
        assert obs.url == "https://example.com"
        assert obs.frame is img

    def test_build_delta(self, config):
        t0 = Image.new("L", (200, 200), color=128)
        arr1 = np.full((200, 200), 128, dtype=np.uint8)
        arr1[50:100, 50:100] = 255
        t1 = Image.fromarray(arr1)
        diff_result = compute_diff(t0, t1, config)
        crops = extract_crops(t0, t1, diff_result.changed_bboxes, config.CROP_PADDING)

        obs = build_observation(
            obs_type="delta",
            task="test",
            step=3,
            last_action=Action(type=ActionType.CLICK, x=75, y=75),
            diff_result=diff_result,
            crops=crops,
            action_had_effect=diff_result.action_had_effect,
            no_change_count=0,
        )
        assert isinstance(obs, DeltaObservation)
        assert obs.obs_type == "delta"
        assert obs.action_had_effect
        assert len(obs.crops) > 0

    def test_build_delta_with_text_deltas(self):
        obs = build_observation(
            obs_type="delta",
            task="test",
            step=5,
            last_action=Action(type=ActionType.CLICK, x=100, y=200),
            text_deltas=[{"bbox": (100, 200, 80, 20), "before": "unavailable", "after": "available"}],
            action_had_effect=True,
            no_change_count=0,
        )
        assert isinstance(obs, DeltaObservation)
        assert len(obs.text_deltas) == 1
        assert obs.text_deltas[0]["before"] == "unavailable"


# -- Action parsing tests --

class TestActionParsing:
    def test_parse_click(self):
        a = parse_action({"type": "click", "x": 100, "y": 200})
        assert a.type == ActionType.CLICK
        assert a.x == 100
        assert a.y == 200

    def test_parse_type(self):
        a = parse_action({"type": "type", "text": "hello"})
        assert a.type == ActionType.TYPE
        assert a.text == "hello"

    def test_parse_scroll(self):
        a = parse_action({"type": "scroll", "direction": "down", "amount": 300})
        assert a.type == ActionType.SCROLL

    def test_parse_key(self):
        a = parse_action({"type": "key", "key": "Enter"})
        assert a.type == ActionType.KEY
        assert a.key == "Enter"

    def test_parse_done(self):
        a = parse_action({"type": "done"})
        assert a.type == ActionType.DONE

    def test_parse_none(self):
        assert parse_action(None) is None

    def test_parse_invalid(self):
        assert parse_action({"type": "invalid_action"}) is None

    def test_action_str_repr(self):
        assert "click(100, 200)" == str(Action(type=ActionType.CLICK, x=100, y=200))
        assert "type('hi')" == str(Action(type=ActionType.TYPE, text="hi"))
        assert "done" == str(Action(type=ActionType.DONE))


# -- Agent state tests --

class TestAgentState:
    def test_initial_state(self):
        s = AgentState(task="test")
        assert s.step == 0
        assert not s.done
        assert s.delta_ratio == 0.0

    def test_no_change_streak(self):
        s = AgentState(task="test")
        s.increment_no_change_streak()
        s.increment_no_change_streak()
        assert s.no_change_streak == 2
        s.reset_no_change_streak()
        assert s.no_change_streak == 0

    def test_delta_ratio(self):
        s = AgentState(task="test")
        from vision.classifier import ClassificationResult
        delta = ClassificationResult(TransitionType.DELTA, "none", 0.01, 0, 0.9)
        new_page = ClassificationResult(TransitionType.NEW_PAGE, "url_change", 0.0, 0, 1.0)

        s.log_transition(delta, Action(type=ActionType.CLICK, x=1, y=1), 0)
        s.log_transition(delta, Action(type=ActionType.CLICK, x=1, y=1), 1)
        s.log_transition(new_page, Action(type=ActionType.CLICK, x=1, y=1), 2)

        assert s.delta_ratio == pytest.approx(2 / 3)


# -- Full pipeline simulation (mock model, real vision) --

class TestFullPipeline:
    """Simulate a 3-step agent run with real diff/classification, mock model."""

    def test_simulated_run(self, config):
        """
        Simulate: initial full frame → click (delta with effect) → click (no effect) → done
        """
        state = AgentState(task="test task")

        # Step 0: gradient + noise — real screenshots have rich texture that
        # stabilizes pHash against small localized changes. Pure gradients are
        # too smooth (DCT is very clean, so a small patch flips many coefficients).
        rng = np.random.RandomState(42)
        grad = np.tile(np.linspace(0, 255, 1280), (900, 1))
        noise = rng.normal(0, 20, (900, 1280))
        frame0_arr = np.clip(grad + noise, 0, 255).astype(np.uint8)
        frame0 = Image.fromarray(frame0_arr)
        obs0 = build_observation(
            obs_type="full_frame", task="test task", step=0,
            last_action=None, frame=frame0, url="https://test.com",
            trigger_reason="initial",
        )
        assert obs0.obs_type == "full_frame"
        state.add_observation(obs0)
        anchor = extract_anchor(frame0, config)

        # Skip pHash layer — DCT-based pHash is too sensitive for synthetic
        # images (a 50x100 patch flips 30+ bits on smooth images).
        # Real screenshots are tested in test_live_capture.py where pHash works.
        config.PHASH_DISTANCE_THRESHOLD = 64  # max possible, effectively disabled

        # Step 1: click caused a change (button highlight appeared).
        # Moderate brightness shift — simulates real button hover/focus, not a
        # synthetic 50→200 jump that would blow up the pHash.
        frame1_arr = frame0_arr.copy()
        frame1_arr[350:470, 550:750] = np.clip(
            frame1_arr[350:470, 550:750].astype(np.int16) + 80, 0, 255
        ).astype(np.uint8)
        frame1 = Image.fromarray(frame1_arr)

        diff1 = compute_diff(frame0, frame1, config)
        cls1 = classify_transition(
            frame0, frame1,
            "https://test.com", "https://test.com",
            anchor, config, diff1,
        )
        assert cls1.transition == TransitionType.DELTA
        assert diff1.action_had_effect

        crops1 = extract_crops(frame0.convert("L"), frame1, diff1.changed_bboxes, config.CROP_PADDING)
        obs1 = build_observation(
            obs_type="delta", task="test task", step=1,
            last_action=Action(type=ActionType.CLICK, x=650, y=425),
            diff_result=diff1, crops=crops1,
            action_had_effect=True, no_change_count=0,
        )
        assert obs1.obs_type == "delta"
        assert len(obs1.crops) >= 1
        state.add_observation(obs1)
        state.step = 1

        # Step 2: click on same spot — no change
        # Step 2: click on same spot — no change
        diff2 = compute_diff(frame1, frame1, config)
        assert not diff2.action_had_effect
        assert diff2.diff_ratio == 0.0

        state.log_transition(cls1, Action(type=ActionType.CLICK, x=650, y=425), 0)
        assert state.delta_ratio == 1.0  # 100% delta so far
