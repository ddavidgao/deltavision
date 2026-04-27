"""
Tests for DeltaVisionObserver — the public middleware API.

Verifies:
  1. Observer lifecycle (t0 bootstrap, re-anchoring on NEW_PAGE, reset)
  2. Screenshot input flexibility (PIL / bytes / b64 / data URL)
  3. Classification pass-through (DELTA vs NEW_PAGE routing)
  4. Every format adapter produces a schema matching the target API
  5. Edge cases: empty crops, scroll action, no effect

These tests pin the integration contracts for Anthropic / OpenAI / Browser
Use / Skyvern / Stagehand. If a test here breaks, a CU bot's integration
breaks.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from observer import DeltaVisionObserver, DVObservation, _load_screenshot

# ============================================================= helpers

def solid(color=(128, 128, 128), size=(400, 300)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def striped(variant: int, size=(400, 300)) -> Image.Image:
    """Dramatically different per variant — triggers NEW_PAGE."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = ((x + variant * 30) % 255, (y + variant * 50) % 255, 128)
    return Image.fromarray(arr)


def small_delta(base: Image.Image, seed: int = 1) -> Image.Image:
    """Produce a frame that differs from `base` in a tiny contiguous region
    only — stays well inside the DELTA classification boundary (<<1% pixels
    changed, pHash distance tiny, anchor preserved).

    This is what a real SPA click/type produces: a few hundred pixels change.
    """
    arr = np.asarray(base).copy()
    # modify a very small bottom-center region (avoids the top-strip anchor)
    rng = np.random.default_rng(seed)
    h, w = arr.shape[:2]
    # Tiny ~0.2% area patch — well under NEW_PAGE threshold (0.75) AND pHash
    # threshold (20). 32x32 on a 1280x900 = 0.09% pixels.
    bh = bw = min(32, h // 16, w // 16)
    # place toward bottom so the anchor (top 8%) stays intact
    y0 = h - bh - 8
    x0 = (w - bw) // 2
    arr[y0:y0 + bh, x0:x0 + bw] = rng.integers(0, 255, size=(bh, bw, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_b64(img: Image.Image) -> str:
    return base64.b64encode(png_bytes(img)).decode()


# ============================================================= input parsing

class TestLoadScreenshot:
    def test_pil_passthrough(self):
        img = solid()
        out = _load_screenshot(img)
        assert out is img or out.size == img.size

    def test_rgb_coerced(self):
        img = solid().convert("RGBA")
        out = _load_screenshot(img)
        assert out.mode == "RGB"

    def test_bytes(self):
        img = solid()
        out = _load_screenshot(png_bytes(img))
        assert out.size == img.size
        assert out.mode == "RGB"

    def test_base64_string(self):
        img = solid()
        out = _load_screenshot(png_b64(img))
        assert out.size == img.size

    def test_data_url(self):
        img = solid()
        data_url = f"data:image/png;base64,{png_b64(img)}"
        out = _load_screenshot(data_url)
        assert out.size == img.size

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _load_screenshot(12345)


# ============================================================= lifecycle

class TestLifecycle:
    def test_first_call_is_full_frame(self):
        obs = DeltaVisionObserver()
        result = obs.observe(solid())
        assert result.obs_type == "full_frame"
        assert result.trigger == "initial"
        assert result.frame is not None
        assert result.thumbnail is None
        assert result.crops == []

    def test_second_call_with_no_change_is_delta(self):
        obs = DeltaVisionObserver()
        img = solid()
        obs.observe(img)
        result = obs.observe(img)  # identical frame
        assert result.obs_type == "delta"
        assert result.diff_ratio == 0.0

    def test_new_page_triggers_reanchor(self):
        obs = DeltaVisionObserver()
        obs.observe(solid((0, 0, 0)))          # initial black
        _ = obs.observe(solid((255, 255, 255)))  # big change
        # After a NEW_PAGE, t0 is re-anchored — a third identical frame is
        # measured against the NEW t0, not the original
        result = obs.observe(solid((255, 255, 255)))
        assert result.diff_ratio == 0.0

    def test_reset_clears_state(self):
        obs = DeltaVisionObserver()
        obs.observe(solid((0, 0, 0)))
        obs.observe(solid((255, 255, 255)))  # NEW_PAGE
        assert obs.step > 0
        obs.reset()
        assert obs.step == 0
        # After reset, first call should behave like a fresh observer
        result = obs.observe(solid())
        assert result.trigger == "initial"

    def test_step_counter_advances(self):
        obs = DeltaVisionObserver()
        obs.observe(striped(0))
        obs.observe(striped(1))
        obs.observe(striped(2))
        assert obs.step == 2


class TestUrlPassthrough:
    def test_url_change_triggers_new_page(self):
        obs = DeltaVisionObserver()
        obs.observe(solid(), url="https://a.com")
        result = obs.observe(solid(), url="https://b.com")
        # URL change → Layer 1 → NEW_PAGE even when pixels identical
        assert result.obs_type == "full_frame"
        assert result.trigger in ("url_change", "new_url")

    def test_none_url_does_not_crash(self):
        obs = DeltaVisionObserver()
        obs.observe(solid(), url=None)
        result = obs.observe(solid(), url=None)
        assert result.obs_type == "delta"


class TestInputFlexibility:
    def test_accepts_pil(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        assert r.obs_type == "full_frame"

    def test_accepts_bytes(self):
        obs = DeltaVisionObserver()
        r = obs.observe(png_bytes(solid()))
        assert r.obs_type == "full_frame"

    def test_accepts_b64(self):
        obs = DeltaVisionObserver()
        r = obs.observe(png_b64(solid()))
        assert r.obs_type == "full_frame"

    def test_accepts_data_url(self):
        obs = DeltaVisionObserver()
        r = obs.observe(f"data:image/png;base64,{png_b64(solid())}")
        assert r.obs_type == "full_frame"


# ============================================================= adapters

class TestAnthropicAdapter:
    def _setup_delta(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        return obs.observe(small_delta(base))

    def test_full_frame_has_one_image_block(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        blocks = r.to_anthropic_tool_result_content()
        imgs = [b for b in blocks if b.get("type") == "image"]
        assert len(imgs) == 1
        src = imgs[0]["source"]
        assert src["type"] == "base64"
        assert src["media_type"] == "image/png"
        assert isinstance(src["data"], str) and len(src["data"]) > 100

    def test_delta_has_thumbnail_and_crops(self):
        r = self._setup_delta()
        blocks = r.to_anthropic_tool_result_content()
        imgs = [b for b in blocks if b.get("type") == "image"]
        # At least thumbnail; possibly crops
        assert len(imgs) >= 1
        # Every image block is schema-valid
        for b in imgs:
            assert b["source"]["type"] == "base64"
            assert b["source"]["media_type"] == "image/png"

    def test_delta_has_text_header(self):
        r = self._setup_delta()
        blocks = r.to_anthropic_tool_result_content()
        texts = [b for b in blocks if b.get("type") == "text"]
        assert any("DELTA" in t["text"] for t in texts)


class TestOpenAICUAAdapter:
    def test_full_frame_shape(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        out = r.to_openai_computer_call_output(call_id="call_abc")
        assert out["type"] == "computer_call_output"
        assert out["call_id"] == "call_abc"
        assert out["output"]["type"] == "computer_screenshot"
        assert out["output"]["image_url"].startswith("data:image/png;base64,")

    def test_delta_shape(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        r = obs.observe(small_delta(base))
        out = r.to_openai_computer_call_output(call_id="call_xyz")
        # CUA expects a single screenshot — our adapter composites into one
        assert out["type"] == "computer_call_output"
        assert out["call_id"] == "call_xyz"
        assert out["output"]["type"] == "computer_screenshot"
        data_url = out["output"]["image_url"]
        assert data_url.startswith("data:image/png;base64,")
        # Decode and verify it's a real PNG
        b64 = data_url.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert img.width > 0 and img.height > 0


class TestOpenAIVisionAdapter:
    def test_full_frame(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        parts = r.to_openai_vision_content()
        assert any(p.get("type") == "image_url" for p in parts)
        urls = [p for p in parts if p.get("type") == "image_url"]
        assert urls[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_delta_has_multiple_parts(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        r = obs.observe(small_delta(base))
        assert r.obs_type == "delta"
        parts = r.to_openai_vision_content()
        # header text + thumbnail image + possibly crops
        assert len(parts) >= 2


class TestBrowserUseAdapter:
    def test_returns_base64_png_string(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        b64 = r.to_browser_use_screenshot_b64()
        assert isinstance(b64, str)
        # Must decode as a valid PNG
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert img.format == "PNG"

    def test_delta_produces_composited_single_image(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        r = obs.observe(small_delta(base))
        assert r.obs_type == "delta"
        b64 = r.to_browser_use_screenshot_b64()
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # Composite is at least as wide as the 320-thumbnail
        assert img.width >= 320
        assert img.height >= 225


class TestSkyvernAdapter:
    def test_full_frame_returns_single_png(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        shots = r.to_skyvern_screenshots_list()
        assert len(shots) == 1
        assert isinstance(shots[0], bytes)
        # PNG magic bytes
        assert shots[0][:8] == b"\x89PNG\r\n\x1a\n"

    def test_delta_returns_thumbnail_plus_crops(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        r = obs.observe(small_delta(base))
        assert r.obs_type == "delta"
        shots = r.to_skyvern_screenshots_list()
        # At least the thumbnail
        assert len(shots) >= 1
        for s in shots:
            assert s[:8] == b"\x89PNG\r\n\x1a\n"


class TestStagehandAdapter:
    def test_full_frame_part_shape(self):
        obs = DeltaVisionObserver()
        r = obs.observe(solid())
        parts = r.to_stagehand_middleware_parts()
        media = [p for p in parts if p.get("type") == "media"]
        assert len(media) == 1
        assert media[0]["mediaType"] == "image/png"
        assert isinstance(media[0]["data"], str)

    def test_delta_has_header_text_and_media(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        r = obs.observe(small_delta(base))
        assert r.obs_type == "delta"
        parts = r.to_stagehand_middleware_parts()
        text_parts = [p for p in parts if p.get("type") == "text"]
        media_parts = [p for p in parts if p.get("type") == "media"]
        assert len(text_parts) >= 1
        assert len(media_parts) >= 1


class TestRawAdapter:
    def test_contains_all_fields(self):
        obs = DeltaVisionObserver()
        base = striped(0)
        obs.observe(base)
        r = obs.observe(small_delta(base))
        raw = r.to_raw()
        for key in ("obs_type", "step", "trigger", "diff_ratio",
                    "phash_distance", "anchor_score", "action_had_effect",
                    "frame", "thumbnail", "crops", "crop_bboxes",
                    "url", "last_action", "estimated_image_tokens"):
            assert key in raw


# ============================================================= token estimation

class TestTokenEstimation:
    def test_delta_cheaper_than_full_frame(self):
        # Use a textured base so small_delta doesn't perturb pHash too much —
        # a solid-color base is pathological for pHash (any perturbation
        # flips its hash). Real screenshots are always textured.
        obs = DeltaVisionObserver()
        base = striped(0, size=(1280, 900))
        ff = obs.observe(base)
        delta = obs.observe(small_delta(base))
        assert delta.obs_type == "delta", \
            f"expected delta, got {delta.obs_type} with trigger={delta.trigger}"
        assert ff.estimated_image_tokens() > delta.estimated_image_tokens()

    def test_full_frame_tokens_scale_with_resolution(self):
        obs1 = DeltaVisionObserver()
        r1 = obs1.observe(solid(size=(640, 480)))

        obs2 = DeltaVisionObserver()
        r2 = obs2.observe(solid(size=(1920, 1080)))

        assert r2.estimated_image_tokens() > r1.estimated_image_tokens()


# ============================================================= cost split (v1.0.7-dev)

class TestCostSplit:
    """The v1.0.7-dev cost-split exposes two distinct token numbers on
    every observation:

        model_facing_tokens()  — what the adapter ships to the model
        dv_internal_tokens()   — what DV consumed internally (always full frame)

    The savings claim is `1 - model_facing / dv_internal`, summed across
    a trace. Before this split, both numbers lived under one ambiguous
    `estimated_image_tokens()` call, which let critics ask "are you
    counting the screenshots DV throws away?" — the new methods make
    the answer explicit.
    """

    def test_full_frame_observation_has_equal_costs(self):
        """On a full-frame observation, the model gets the same image DV
        consumed. Both costs equal the cost of that frame."""
        obs = DeltaVisionObserver()
        ff = obs.observe(striped(0, size=(1280, 800)))
        assert ff.is_new_page()
        assert ff.model_facing_tokens() == ff.dv_internal_tokens(), (
            f"full-frame: model-facing ({ff.model_facing_tokens()}) "
            f"should equal dv-internal ({ff.dv_internal_tokens()}) — "
            f"DV shipped exactly what it consumed"
        )

    def test_delta_model_facing_smaller_than_internal(self):
        """The whole point of DV: on a delta observation, model_facing <
        dv_internal. This is the savings primitive."""
        obs = DeltaVisionObserver()
        base = striped(0, size=(1280, 900))
        obs.observe(base)
        delta = obs.observe(small_delta(base))
        assert delta.obs_type == "delta"
        assert delta.model_facing_tokens() < delta.dv_internal_tokens(), (
            f"delta step should ship fewer tokens to the model "
            f"({delta.model_facing_tokens()}) than DV consumed "
            f"({delta.dv_internal_tokens()}) — that's the savings"
        )

    def test_dv_internal_tokens_does_not_depend_on_obs_type(self):
        """DV consumes a full frame whether it ends up shipping one or
        not. dv_internal should equal the FF cost of the same frame
        regardless of obs_type."""
        obs = DeltaVisionObserver()
        base = striped(0, size=(1280, 900))
        ff = obs.observe(base)
        delta = obs.observe(small_delta(base))
        # Both observations consumed a 1280×900 frame.
        assert ff.dv_internal_tokens() == delta.dv_internal_tokens(), (
            f"dv_internal should be size-only (consumed frame size), "
            f"not obs_type-dependent: ff={ff.dv_internal_tokens()}, "
            f"delta={delta.dv_internal_tokens()}"
        )

    def test_estimated_image_tokens_is_alias_for_model_facing(self):
        """Back-compat: existing user code calls estimated_image_tokens().
        It must keep working AND must return the same number as the
        new model_facing_tokens() method until v1.1.0 removes it."""
        obs = DeltaVisionObserver()
        base = striped(0, size=(1280, 900))
        ff = obs.observe(base)
        delta = obs.observe(small_delta(base))
        for o in (ff, delta):
            assert o.estimated_image_tokens() == o.model_facing_tokens(), (
                f"deprecated alias must equal model_facing_tokens() "
                f"(obs_type={o.obs_type})"
            )

    def test_to_raw_emits_both_cost_keys_and_legacy_alias(self):
        """The to_raw() dict (used by JSON serialization) must include
        both new keys AND the deprecated alias for one release."""
        obs = DeltaVisionObserver()
        ff = obs.observe(striped(0, size=(1280, 800)))
        raw = ff.to_raw()
        # New explicit keys
        assert "model_facing_tokens" in raw
        assert "dv_internal_tokens" in raw
        # Back-compat key
        assert "estimated_image_tokens" in raw
        assert raw["estimated_image_tokens"] == raw["model_facing_tokens"]

    def test_consumed_frame_size_populated_on_both_obs_types(self):
        """consumed_frame_size is the data dv_internal_tokens() prefers.
        Observer always populates it on both full_frame AND delta observations
        so the cost-split is always available without falling back to other
        resolution paths."""
        obs = DeltaVisionObserver()
        base = striped(0, size=(1280, 900))
        ff = obs.observe(base)
        delta = obs.observe(small_delta(base))
        assert ff.consumed_frame_size == (1280, 900)
        assert delta.consumed_frame_size == (1280, 900)

    # BUG-0008 hardening: dv_internal_tokens() must produce a useful answer
    # OR raise — silent zero would let savings claims silently inflate.

    def test_full_frame_obs_without_consumed_frame_size_falls_back_to_frame(self):
        """A hand-built full_frame DVObservation that omits consumed_frame_size
        should still work — `frame` is right there, so dv_internal_tokens()
        can derive the size. This keeps mocks/tests/custom integrations from
        breaking just because the cost-split was added."""
        from PIL import Image as _Image
        frame = _Image.new("RGB", (1280, 800), (200, 200, 200))
        obs = DVObservation(
            obs_type="full_frame", step=0,
            trigger="initial", diff_ratio=0.0,
            phash_distance=0, anchor_score=1.0,
            action_had_effect=False,
            frame=frame,
            # consumed_frame_size deliberately omitted
        )
        assert obs.consumed_frame_size is None
        # Should derive from frame.size and equal model_facing on a full frame
        assert obs.dv_internal_tokens() > 0
        assert obs.dv_internal_tokens() == obs.model_facing_tokens(), (
            "full-frame fallback path should match the cost of `frame`"
        )

    def test_delta_obs_without_consumed_frame_size_raises(self):
        """A hand-built delta DVObservation that omits consumed_frame_size
        has no way to recover what DV consumed (delta observations don't
        carry the full frame). Must raise ValueError — silent zero would
        let cost-split callers compute bogus 100%/inf savings."""
        import pytest as _pytest
        from PIL import Image as _Image
        thumb = _Image.new("RGB", (320, 200), (255, 255, 255))
        obs = DVObservation(
            obs_type="delta", step=1,
            trigger="diff_ratio", diff_ratio=0.05,
            phash_distance=2, anchor_score=0.95,
            action_had_effect=True,
            frame=None,
            thumbnail=thumb,
            crops=[],
            crop_bboxes=[],
            # consumed_frame_size deliberately omitted
        )
        with _pytest.raises(ValueError, match="consumed_frame_size"):
            obs.dv_internal_tokens()


# ============================================================= scroll handling

class TestScrollReanchor:
    def test_scroll_action_re_anchors(self):
        """After a scroll, the t0 anchor should move to the new frame so
        subsequent diffs don't include the scroll shift."""
        obs = DeltaVisionObserver()
        base = striped(0)
        f2 = small_delta(base, seed=1)
        obs.observe(base)
        obs.observe(f2, last_action="scroll(down, 500px)")
        # Next observe of the same f2 should be near-zero diff since anchor moved
        result = obs.observe(f2)
        assert result.diff_ratio < 0.05
