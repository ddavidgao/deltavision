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
