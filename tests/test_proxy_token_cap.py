"""
Regression test for the token-cap guard in dv_playwright_mcp.DVState.

The proxy used to emit `dv_tokens > FULL_FRAME_TOKENS` on frames fragmented into
many small crops — the crop-base overhead (85 tokens × N crops) plus tile tokens
could exceed the 1365 cost of a full frame, making DV strictly worse than FF on
that step. The guard introduced 2026-04-24 caps dv_cost at FULL_FRAME_TOKENS by
falling back to a full-frame response when crops would cost more.

Invariant this test enforces: for ANY frame pair the proxy processes, the
reported dv_cost must never exceed FULL_FRAME_TOKENS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The proxy module imports the optional `mcp` SDK at the top level — it's a runtime
# dependency for the live-MCP server, not a test dependency. Skip the whole module
# in environments where mcp isn't installed (CI runs the offline subset only).
pytest.importorskip("mcp", reason="mcp SDK not installed; proxy tests require it")

import dv_playwright_mcp as proxy  # noqa: E402  (importorskip gate above)


def _white(w=1280, h=800):
    return Image.new("RGB", (w, h), (255, 255, 255))


def _checkered_scatter(w=1280, h=800, n_regions=20, region_size=40):
    """Image with many scattered dark regions — maximizes crop count."""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    px = img.load()
    # Deterministic scatter pattern — spread regions across the frame
    import math
    for i in range(n_regions):
        cx = int((w - region_size) * ((i * 0.618) % 1.0))  # golden-ratio scatter
        cy = int((h - region_size) * ((i * 0.382) % 1.0))
        for dx in range(region_size):
            for dy in range(region_size):
                px[cx + dx, cy + dy] = (0, 0, 0)
    return img


def test_token_cap_on_fragmented_diff():
    """
    When t1 has many small scattered changes vs t0, the proxy MUST cap DV cost
    at FULL_FRAME_TOKENS rather than billing more than a full frame.
    """
    # Fresh state — skip the initial-frame branch by seeding t0 directly
    state = proxy.DVState()
    t0 = _white()
    # Seed via process_screenshot so classifier's internal state is coherent
    state.process_screenshot(t0)

    assert state.dv_tokens == proxy.FULL_FRAME_TOKENS, \
        "initial frame should cost exactly one full frame"

    # Second frame: many scattered regions → fragmented diff → many crops
    t1 = _checkered_scatter(n_regions=20, region_size=40)
    start_dv = state.dv_tokens
    content, transition, trigger = state.process_screenshot(t1)
    step_cost = state.dv_tokens - start_dv

    assert step_cost <= proxy.FULL_FRAME_TOKENS, (
        f"fragmented-diff step cost {step_cost} tokens, which exceeds "
        f"FULL_FRAME_TOKENS ({proxy.FULL_FRAME_TOKENS}). The token-cap guard in "
        f"DVState.process_screenshot must fall back to a full frame when crops "
        f"would cost more."
    )


def test_identical_frames_cheap():
    """Identical frames should yield the cache-pointer fast exit (85 tokens)."""
    state = proxy.DVState()
    t0 = _white()
    state.process_screenshot(t0)
    start_dv = state.dv_tokens
    state.process_screenshot(t0.copy())
    step_cost = state.dv_tokens - start_dv

    # Should be minimal — either CROP_BASE_TOKENS (85) or similar small value
    assert step_cost <= proxy.CROP_BASE_TOKENS * 2, (
        f"identical-frame step cost {step_cost} tokens, expected near "
        f"CROP_BASE_TOKENS ({proxy.CROP_BASE_TOKENS})"
    )


def test_full_frame_mode_no_cap_logic():
    """FF ablation mode should always emit FULL_FRAME_TOKENS regardless of diff."""
    original = proxy.FORCE_FULL_FRAME
    proxy.FORCE_FULL_FRAME = True
    try:
        state = proxy.DVState()
        t0 = _white()
        state.process_screenshot(t0)
        assert state.dv_tokens == proxy.FULL_FRAME_TOKENS

        start_dv = state.dv_tokens
        t1 = _checkered_scatter()
        state.process_screenshot(t1)
        assert state.dv_tokens - start_dv == proxy.FULL_FRAME_TOKENS, \
            "FF mode should bill a full frame on every step"
    finally:
        proxy.FORCE_FULL_FRAME = original


def _small_localized_change(seed):
    """A small localized change at a deterministic location — produces a single
    cheap-crop delta when diffed against `_white()`. Different `seed` values
    produce visually different small dots so consecutive frames keep counting
    as deltas (not as 'no visible change')."""
    img = Image.new("RGB", (1280, 800), (255, 255, 255))
    px = img.load()
    # Small 30x30 black square at a position keyed off seed
    cx = 100 + (seed * 50) % 800
    cy = 100 + (seed * 30) % 600
    for dx in range(30):
        for dy in range(30):
            px[cx + dx, cy + dy] = (0, 0, 0)
    return img


def test_periodic_refresh_fires_after_n_deltas():
    """
    After DELTA_REFRESH_EVERY consecutive small deltas, the proxy must force
    a full-frame response so the agent re-anchors. This protects against the
    spatial-context-loss failure mode where the agent gets lost in dialog
    interactions because it's only seeing tiny crops.

    Sequence of states observed during this test:
      call 0: white → first_localized_change   -> NEW_PAGE (big pHash jump)
      call 1: localized → next_localized       -> DELTA, streak=1
      call 2:                                   -> DELTA, streak=2
      ...
      call N: streak=N
      call N+1: refresh fires (streak >= N at entry)
    """
    state = proxy.DVState()
    state.process_screenshot(_white())  # initial full frame
    # The first localized change vs blank white triggers NEW_PAGE (big pHash).
    # That's not a "delta" we care about for this test — call it the seed.
    state.process_screenshot(_small_localized_change(0))

    n = proxy.DELTA_REFRESH_EVERY
    triggers = []
    costs = []
    # From here on, each subsequent localized change is small enough that
    # the diff is a delta rather than a new_page.
    for i in range(1, n + 3):
        _, _, trigger = state.process_screenshot(_small_localized_change(i))
        triggers.append(trigger)
        costs.append(state.dv_tokens)

    # The N+1'th delta call is when streak reaches N, so the (N+1)th call
    # in our loop (index n in 0-indexed) is where refresh fires.
    refresh_indices = [i for i, t in enumerate(triggers) if t == "periodic_refresh"]
    assert len(refresh_indices) >= 1, (
        f"Expected periodic_refresh in {len(triggers)} delta calls, got {triggers}"
    )
    # The first refresh fires when delta_streak first reaches >= n.
    # streak transitions: after delta call k -> streak=k. The refresh check is
    # on entry to a call: if streak >= n already, refresh. So first refresh is
    # call (n+1) in absolute, which is index (n) in our 0-indexed loop.
    assert refresh_indices[0] == n, (
        f"Expected first refresh at delta-call index {n} (0-indexed), got {refresh_indices[0]}"
    )


def test_periodic_refresh_resets_streak():
    """After firing a refresh, the streak resets and the very next call is a crop."""
    state = proxy.DVState()
    state.process_screenshot(_white())
    state.process_screenshot(_small_localized_change(0))  # seed (likely NEW_PAGE)

    n = proxy.DELTA_REFRESH_EVERY
    triggers = []
    # Feed n+1 small-delta calls; the last one should be the refresh
    for i in range(1, n + 2):
        _, _, trigger = state.process_screenshot(_small_localized_change(i))
        triggers.append(trigger)

    assert triggers[-1] == "periodic_refresh", (
        f"Expected last call to be periodic_refresh, got {triggers}"
    )
    assert state.delta_streak == 0, "delta_streak should reset after refresh"

    # Next delta call should NOT be a refresh — streak just reset
    _, _, trigger = state.process_screenshot(_small_localized_change(99))
    assert trigger != "periodic_refresh", (
        "delta_streak just reset — should be 1 now, far below threshold"
    )
