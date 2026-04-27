"""
Targeted test for the dv_playwright_mcp v1-trace retrofit.

The proxy now dual-writes:
    - dv_runs/dv_proxy_run_<id>.jsonl    (legacy, untouched)
    - dv_runs/dv_trace_v1_<id>.jsonl     (v1 BenchmarkTrace, validates clean)

We don't run the full MCP server (no Playwright session needed) — just
construct a DVState directly, feed it 3 hand-built frames, and assert:
    * both files exist
    * legacy file has 3 step lines (one per call)
    * v1 file parses cleanly via parse_trace
    * verify-trace passes (`validate_trace(...).ok == True`)
    * the v1 trace's payload_image_sha256 actually matches what
      canonical_image_manifest produces from the recorded payload_images
      (this is the credibility unlock — the verifier already enforces this,
      but exercising it end-to-end through real proxy code is the
      point of this test)
"""
from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("mcp", reason="proxy module imports the optional `mcp` SDK")


def _white(w=1280, h=800):
    return Image.new("RGB", (w, h), (255, 255, 255))


def _bordered(w=1280, h=800, *, x=200, y=200, bw=100, bh=80, color=(0, 0, 0)):
    """Frame identical to _white() except a small filled rect — produces a
    localized DELTA when diffed against _white()."""
    img = _white(w, h)
    px = img.load()
    for dx in range(bw):
        for dy in range(bh):
            px[x + dx, y + dy] = color
    return img


def test_dual_write_proxy_emits_valid_v1_trace(tmp_path, monkeypatch):
    """End-to-end: drive process_screenshot 3× and verify both legacy +
    v1 trace files are produced, and the v1 trace passes the validator."""
    # Redirect both log files to a tmp_path BEFORE importing the proxy so
    # the module-level paths bind to this test's directory.
    monkeypatch.setenv("DV_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DV_TASK_ID", "proxy-retrofit-test")
    monkeypatch.setenv(
        "DV_TASK_DESCRIPTION",
        "drive 3 frames through proxy and verify dual-write",
    )
    monkeypatch.setenv("DV_TRIAL_GROUP_ID", "test-group")
    monkeypatch.setenv("DV_MODEL", "claude-sonnet-4")

    # Force a fresh import so RUN_ID, LOG_FILE, TRACE_FILE bind to this
    # test's tmp_path. Must come after env setup.
    import importlib
    import sys
    if "dv_playwright_mcp" in sys.modules:
        del sys.modules["dv_playwright_mcp"]
    proxy = importlib.import_module("dv_playwright_mcp")

    # Sanity: log files exist as paths and live in tmp_path
    assert proxy.LOG_FILE.parent == tmp_path
    assert proxy.TRACE_FILE.parent == tmp_path
    assert proxy.TRACE_FILE.name.startswith("dv_trace_v1_")

    # Drive 3 screenshots:
    #   1. white      → initial (full_frame)
    #   2. bordered   → delta (small black rect appears)
    #   3. white      → delta (rect disappears)
    state = proxy.DVState()
    f1 = _white()
    f2 = _bordered()
    f3 = _white()

    state.process_screenshot(f1)
    state.process_screenshot(f2)
    state.process_screenshot(f3)

    # Force the writer to flush its summary line. In production this happens
    # via atexit; in tests we close manually.
    proxy._close_trace_writer()

    # Both files should exist now
    assert proxy.LOG_FILE.exists(), "legacy log file missing"
    assert proxy.TRACE_FILE.exists(), "v1 trace file missing"

    # Legacy: 3 step records
    legacy_lines = [
        line for line in proxy.LOG_FILE.read_text().splitlines() if line.strip()
    ]
    assert len(legacy_lines) == 3, (
        f"expected 3 legacy log lines, got {len(legacy_lines)}: {legacy_lines}"
    )

    # v1 trace: parse + validate
    from results.trace import parse_trace, validate_trace
    trace = parse_trace(proxy.TRACE_FILE)
    report = validate_trace(trace, check_paths=False)
    assert report.ok, f"v1 trace failed validation: {report.errors}"
    assert trace.summary is not None, "writer should emit summary on clean close"
    assert report.n_steps == 3
    # First step is initial (full_frame), so total dv_internal = 3 × 1365.
    assert report.total_dv_internal_tokens == 3 * proxy.FULL_FRAME_TOKENS

    # Header captures the task identity from env
    assert trace.header["task_id"] == "proxy-retrofit-test"
    assert trace.header["trial_group_id"] == "test-group"
    assert trace.header["observation_mode"] == "dv"
    assert trace.header["model"] == "claude-sonnet-4"

    # Each step's payload_image_sha256 must equal the canonical-manifest
    # hash of its payload_images. The validator already enforces this, but
    # we double-check to confirm the proxy is producing internally-consistent
    # records (not just records the validator happens to accept).
    from results.trace import canonical_image_manifest
    for s in trace.steps:
        recomputed, _ = canonical_image_manifest(s["payload_images"])
        assert s["payload_image_sha256"] == recomputed, (
            f"step {s['step_idx']}: payload_image_sha256 inconsistent with "
            f"its recorded payload_images manifest"
        )


def test_soft_nudge_records_zero_image_tokens(tmp_path, monkeypatch):
    """The soft no-change nudge sends a text-only payload. The v1 trace's
    cost-split semantics demand model_facing_tokens=0 on that step (no
    images → no image tokens), even though the LEGACY log records 85
    tokens for "minimum proxy overhead." Without this fix the trace would
    inflate the savings denominator with phantom image tokens.

    Also asserts the trigger is the explicit `no_change_soft_nudge`, not
    the upstream classifier trigger.
    """
    monkeypatch.setenv("DV_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DV_TASK_ID", "soft-nudge-test")
    # NO_CHANGE_HARD_NUDGE defaults to 2; we want the soft branch (streak=1)
    # to fire, so we send 2 frames: first triggers initial, second triggers
    # the soft nudge (since it's identical to the first).
    monkeypatch.setenv("DV_NO_CHANGE_HARD_NUDGE", "5")  # raise so soft fires

    import importlib
    import sys
    if "dv_playwright_mcp" in sys.modules:
        del sys.modules["dv_playwright_mcp"]
    proxy = importlib.import_module("dv_playwright_mcp")

    state = proxy.DVState()
    f1 = _white()
    state.process_screenshot(f1)
    state.process_screenshot(f1.copy())  # identical → no-change → soft nudge
    proxy._close_trace_writer()

    from results.trace import canonical_image_manifest, parse_trace, validate_trace
    trace = parse_trace(proxy.TRACE_FILE)
    report = validate_trace(trace, check_paths=False)
    assert report.ok, f"trace failed validation: {report.errors}"

    # Step 0 = initial (full_frame). Step 1 should be the soft nudge.
    assert len(trace.steps) == 2
    soft = trace.steps[1]
    assert soft["trigger"] == "no_change_soft_nudge", (
        f"trigger should be 'no_change_soft_nudge', got {soft['trigger']!r}"
    )
    assert soft["model_facing_tokens"] == 0, (
        f"text-only soft-nudge step should report 0 model-facing image tokens, "
        f"got {soft['model_facing_tokens']}"
    )
    assert soft["payload_images"] == [], (
        f"soft-nudge has zero image blocks; payload_images should be empty, "
        f"got {soft['payload_images']}"
    )
    # Manifest hash should be the hash of an empty list (stable canonical).
    expected_hash, _ = canonical_image_manifest([])
    assert soft["payload_image_sha256"] == expected_hash


def test_token_cap_fallback_emits_explicit_trigger(tmp_path, monkeypatch):
    """When the summed crop cost exceeds FULL_FRAME_TOKENS and the proxy
    falls back to a full-frame send, the v1 trace MUST record trigger=
    'crop_token_cap_fallback' — NOT the upstream classifier trigger.
    Otherwise post-run analysis can't distinguish a genuine classifier-
    driven full frame from a cost-driven fallback.

    Implementation note: the fallback path is genuinely hard to reach on
    synthetic input because (a) high-contrast scattered patterns trip pHash
    NEW_PAGE classification, and (b) the diff engine's contour-finding
    pre-merges adjacent regions before the optimizer even runs. We exercise
    the fallback by mocking two seams:
        * classify_transition → force DELTA (avoids the NEW_PAGE branch)
        * _estimate_crop_tokens → return FULL_FRAME_TOKENS+1 (forces fallback)
    This isolates the test to the cost-fallback logic in process_screenshot,
    which is the actual subject under test.
    """
    monkeypatch.setenv("DV_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DV_TASK_ID", "token-cap-test")

    import importlib
    import sys
    if "dv_playwright_mcp" in sys.modules:
        del sys.modules["dv_playwright_mcp"]
    proxy = importlib.import_module("dv_playwright_mcp")

    from vision.classifier import ClassificationResult, TransitionType
    real_classify = proxy.classify_transition

    def _force_delta(*args, **kwargs):
        real = real_classify(*args, **kwargs)
        return ClassificationResult(
            transition=TransitionType.DELTA,
            trigger=real.trigger,
            diff_ratio=real.diff_ratio,
            phash_distance=real.phash_distance,
            anchor_score=real.anchor_score,
        )
    monkeypatch.setattr(proxy, "classify_transition", _force_delta)

    # Force fragmented-cost path: report crops as more expensive than a
    # full frame. The ACTUAL cost computation is correct in production
    # (and well-covered by test_proxy_token_cap.py); here we just need
    # the fallback branch to fire deterministically.
    monkeypatch.setattr(
        proxy, "_estimate_crop_tokens",
        lambda crops: proxy.FULL_FRAME_TOKENS + 1,
    )

    def _localized(w=1280, h=800):
        # Small localized change → produces a real (non-empty) crop list,
        # which is what we need to enter the `if crops:` branch.
        img = _white(w, h)
        px = img.load()
        for dx in range(50):
            for dy in range(50):
                px[100 + dx, 100 + dy] = (0, 0, 0)
        return img

    state = proxy.DVState()
    state.process_screenshot(_white())   # initial
    state.process_screenshot(_localized())  # delta → fallback (mocked cost)
    proxy._close_trace_writer()

    from results.trace import parse_trace, validate_trace
    trace = parse_trace(proxy.TRACE_FILE)
    report = validate_trace(trace, check_paths=False)
    assert report.ok, f"trace failed validation: {report.errors}"

    fallback = trace.steps[1]
    assert fallback["obs_type"] == "full_frame"
    assert fallback["trigger"] == "crop_token_cap_fallback", (
        f"fallback trigger should be 'crop_token_cap_fallback', "
        f"got {fallback['trigger']!r} — analysis can't distinguish this "
        f"from a genuine classifier full frame without the explicit label"
    )
    assert fallback["model_facing_tokens"] == proxy.FULL_FRAME_TOKENS


def test_ff_mode_emits_observation_mode_ff(tmp_path, monkeypatch):
    """When DV_FORCE_FULL_FRAME=1, the trace's header.observation_mode must
    be 'ff' so paired-trial tooling can match it to its DV sibling."""
    monkeypatch.setenv("DV_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DV_FORCE_FULL_FRAME", "1")
    monkeypatch.setenv("DV_TASK_ID", "ff-test")

    import importlib
    import sys
    if "dv_playwright_mcp" in sys.modules:
        del sys.modules["dv_playwright_mcp"]
    proxy = importlib.import_module("dv_playwright_mcp")

    assert proxy.FORCE_FULL_FRAME is True
    # Filename also gets the _ff suffix so the file alone tells you which
    # ablation it is, matching the legacy convention.
    assert proxy.TRACE_FILE.name.endswith("_ff.jsonl")

    state = proxy.DVState()
    state.process_screenshot(_white())
    state.process_screenshot(_bordered())
    proxy._close_trace_writer()

    from results.trace import parse_trace, validate_trace
    trace = parse_trace(proxy.TRACE_FILE)
    report = validate_trace(trace, check_paths=False)
    assert report.ok, f"FF v1 trace failed validation: {report.errors}"
    assert trace.header["observation_mode"] == "ff"

    # Every step in FF mode should be obs_type=full_frame with
    # model_facing == dv_internal == 1365 (no compression).
    for s in trace.steps:
        assert s["obs_type"] == "full_frame", (
            f"FF mode step {s['step_idx']} should be full_frame, "
            f"got {s['obs_type']}"
        )
        assert s["model_facing_tokens"] == proxy.FULL_FRAME_TOKENS
        assert s["dv_internal_tokens"] == proxy.FULL_FRAME_TOKENS
