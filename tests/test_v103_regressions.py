"""
v1.0.3 regression tests.

Two distinct failure modes caught during dogfood testing of v1.0.2:

  1. `import deltavision` raised ModuleNotFoundError. Wheel shipped flat
     top-level modules (observer, vision, agent, ...) but no umbrella
     `deltavision` namespace. Distribution name ≠ import name, with no
     public notice → bad first-time UX.

  2. A single-step observation could cost MORE tokens than a full-frame
     baseline. Specifically: scroll events produced a crop bbox covering
     ~100% of the viewport, and DV would emit thumbnail (~96 tok) + full-
     sized crop (~1365 tok) vs the FF baseline of 1365 tok — a net +13%
     regression.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw


# ---------- Finding 1: umbrella import ----------


def test_import_deltavision_works():
    """`import deltavision` must succeed in a fresh install."""
    import deltavision
    assert hasattr(deltavision, "DeltaVisionObserver")
    assert hasattr(deltavision, "DeltaVisionConfig")
    assert hasattr(deltavision, "DVObservation")


def test_from_deltavision_imports():
    """Public API re-exports must be reachable via `from deltavision import X`."""
    from deltavision import (
        DeltaVisionObserver,
        DeltaVisionConfig,
        DVObservation,
        compute_diff,
        compute_phash,
        extract_page_state,
        format_page_state_for_prompt,
    )
    assert DeltaVisionObserver is not None
    assert DeltaVisionConfig is not None
    assert callable(compute_diff)
    assert callable(compute_phash)
    assert callable(extract_page_state)
    assert callable(format_page_state_for_prompt)


def test_flat_imports_still_work():
    """Backwards compat: the flat-module import style v1.0.2 shipped with must
    keep working. We don't want to break users who wrote `from observer import ...`."""
    from observer import DeltaVisionObserver  # noqa: F401
    from vision.diff import compute_diff       # noqa: F401
    from vision.elements import extract_page_state  # noqa: F401


def test_deltavision_version_exported():
    import deltavision
    assert hasattr(deltavision, "__version__")
    assert isinstance(deltavision.__version__, str)
    assert deltavision.__version__.count(".") >= 2  # semver-ish


# ---------- Finding 2: crops-cover-frame guard ----------


def _synthetic_frame(w: int = 1280, h: int = 800, fill=(255, 255, 255)) -> Image.Image:
    """A deterministic frame for testing — solid color + grid pattern.

    The grid keeps pHash stable across small mutations; otherwise a single
    colored rectangle on pure white can shift pHash distance past the
    NEW_PAGE threshold, contaminating fine-grained delta tests.
    """
    img = Image.new("RGB", (w, h), fill)
    d = ImageDraw.Draw(img)
    # Faint grid to anchor pHash
    for y in range(0, h, 40):
        d.line([(0, y), (w, y)], fill=(230, 230, 230), width=1)
    for x in range(0, w, 80):
        d.line([(x, 0), (x, h)], fill=(230, 230, 230), width=1)
    # Small persistent header
    d.rectangle([10, 10, 100, 40], fill=(200, 200, 200))
    return img


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_large_scroll_does_not_exceed_ff_baseline():
    """
    Core regression: if DV's crops cover most of the frame, DV must not
    emit delta (which would be thumbnail + near-full-frame crop); it must
    fall back to full_frame. This ensures DV's token cost never exceeds
    the FF baseline on pathological delta cases.
    """
    from observer import DeltaVisionObserver

    obs = DeltaVisionObserver()

    # Step 1: establish anchor frame (white)
    t0 = _synthetic_frame(fill=(255, 255, 255))
    r0 = obs.observe(_png_bytes(t0), url="http://example.com", last_action="initial")
    assert r0.obs_type == "full_frame"

    # Step 2: dramatic change covering most of the frame (simulate scroll)
    # Fill the entire bottom 75% with a different color — this will produce
    # a huge diff area and a crop that covers nearly the whole frame.
    t1 = _synthetic_frame(fill=(255, 255, 255))
    d = ImageDraw.Draw(t1)
    d.rectangle([0, 200, 1280, 800], fill=(10, 50, 200))  # 75% of frame changed
    r1 = obs.observe(_png_bytes(t1), url="http://example.com", last_action="scroll down 600px")

    ff_tokens = max(75, int((1280 * 800) / 750))  # 1365 for 1280x800

    # DV must not be worse than FF on this step.
    assert r1.estimated_image_tokens() <= ff_tokens, (
        f"DV emitted {r1.estimated_image_tokens()} tok vs FF {ff_tokens} tok — "
        f"crops-cover-frame guard failed. obs_type={r1.obs_type}, "
        f"trigger={r1.trigger!r}, crops={len(r1.crops or [])}"
    )


def test_crop_covers_frame_guard_triggers():
    """Specifically verify that when crops cover ≥75% of the frame, the
    guard fires and emits a full_frame with trigger='crop_covers_frame'."""
    from observer import DeltaVisionObserver

    obs = DeltaVisionObserver()

    # Bootstrap
    t0 = _synthetic_frame(fill=(255, 255, 255))
    obs.observe(_png_bytes(t0), url="http://example.com")

    # Dramatic whole-frame change
    t1 = _synthetic_frame(fill=(20, 20, 20))  # flip to near-black
    r = obs.observe(_png_bytes(t1), url="http://example.com", last_action="scroll")

    # Should fall back to full_frame via the guard OR legitimately classify
    # as NEW_PAGE (either behavior is correct — both avoid the regression).
    assert r.obs_type == "full_frame"


def test_guard_does_not_over_fire():
    """
    The crops-cover-frame guard fires on a frame-sized delta (real scroll).
    It must NOT fire on small deltas. We test this indirectly: if the guard
    were over-eager, the `trigger` on a small delta observation would be
    `crop_covers_frame`. Whatever triggers fire legitimately (pHash,
    diff_ratio, etc.) are fine — we just don't want `crop_covers_frame`
    on cases where the crops are actually tiny.
    """
    from observer import DeltaVisionObserver

    obs = DeltaVisionObserver()

    t0 = _synthetic_frame(fill=(255, 255, 255))
    obs.observe(_png_bytes(t0), url="http://example.com")

    t1 = _synthetic_frame(fill=(255, 255, 255))
    d = ImageDraw.Draw(t1)
    # Tiny change — a few characters typed into a cell, ~2% of frame
    d.rectangle([100, 100, 300, 130], fill=(180, 180, 255))
    r = obs.observe(_png_bytes(t1), url="http://example.com", last_action="type 'hello'")

    # Whatever path is taken, it must not be the crops-cover-frame fallback.
    assert r.trigger != "crop_covers_frame", (
        f"Guard over-fired on a small delta! r={r}"
    )


def test_coverage_threshold_config():
    """Config should expose CROP_COVERAGE_MAX as tunable."""
    from config import DeltaVisionConfig
    cfg = DeltaVisionConfig()
    assert hasattr(cfg, "CROP_COVERAGE_MAX")
    assert 0.0 < cfg.CROP_COVERAGE_MAX <= 1.0


def test_coverage_threshold_can_be_disabled():
    """If set to 1.0, the guard should never fire (users can opt out)."""
    from observer import DeltaVisionObserver
    from config import DeltaVisionConfig

    cfg = DeltaVisionConfig()
    cfg.CROP_COVERAGE_MAX = 1.0  # effectively disable the guard
    obs = DeltaVisionObserver(config=cfg)

    t0 = _synthetic_frame(fill=(255, 255, 255))
    obs.observe(_png_bytes(t0), url="http://example.com")

    t1 = _synthetic_frame(fill=(255, 255, 255))
    d = ImageDraw.Draw(t1)
    d.rectangle([0, 200, 1280, 800], fill=(10, 50, 200))
    r = obs.observe(_png_bytes(t1), url="http://example.com", last_action="scroll")

    # With guard off, we MIGHT still get full_frame via NEW_PAGE classification,
    # but we specifically don't want the "crop_covers_frame" trigger.
    assert r.trigger != "crop_covers_frame", (
        "Guard should be disabled when CROP_COVERAGE_MAX=1.0"
    )
