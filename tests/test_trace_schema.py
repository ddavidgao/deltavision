"""
Tests for results/trace.py — the BenchmarkTrace schema.

Coverage:
    1. Round-trip: writer → file → parser → validator (no errors).
    2. Hash helpers: canonical_image_manifest deterministic across input
       orders that should be identical, AND distinct across inputs that
       should differ.
    3. Validator catches every invariant violation we care about.
    4. Optional file-on-disk hash check (frame_path / payload_path) works
       and fails on mismatch.
    5. JSONL structural errors: missing header, double header, step before
       header, unknown _kind.
"""
from __future__ import annotations

import json

import pytest

from results.trace import (
    SCHEMA_VERSION,
    ParsedTrace,
    TraceHeader,
    TraceStep,
    TraceSummary,
    TraceWriter,
    bytes_sha256,
    canonical_image_manifest,
    config_hash,
    file_sha256,
    parse_trace,
    payload_text_sha256,
    validate_trace,
)

# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

class TestHashHelpers:
    def test_bytes_sha256_known_vector(self):
        # Known vector: sha256("") = e3b0c44...
        assert bytes_sha256(b"").startswith("e3b0c442")

    def test_payload_text_sha256_utf8(self):
        h_ascii = payload_text_sha256("hello")
        h_unicode = payload_text_sha256("héllo")
        assert h_ascii != h_unicode  # different bytes

    def test_canonical_image_manifest_stable(self):
        imgs = [
            {"sha256": "a" * 64, "media_type": "image/png", "bytes": 100},
            {"sha256": "b" * 64, "media_type": "image/png", "bytes": 200},
        ]
        h1, m1 = canonical_image_manifest(imgs)
        h2, m2 = canonical_image_manifest(imgs)
        assert h1 == h2
        assert m1 == m2

    def test_canonical_image_manifest_strips_extra_keys(self):
        # The canonical form keeps only sha256/media_type/bytes — extra
        # input keys (e.g. crop coordinates) shouldn't change the hash.
        imgs_a = [{"sha256": "a" * 64, "media_type": "image/png", "bytes": 100}]
        imgs_b = [
            {
                "sha256": "a" * 64, "media_type": "image/png", "bytes": 100,
                "crop_box": [0, 0, 10, 10],  # extra metadata
            }
        ]
        h_a, _ = canonical_image_manifest(imgs_a)
        h_b, _ = canonical_image_manifest(imgs_b)
        assert h_a == h_b

    def test_canonical_image_manifest_distinguishes_distinct(self):
        h_a, _ = canonical_image_manifest([
            {"sha256": "a" * 64, "media_type": "image/png", "bytes": 100}
        ])
        h_b, _ = canonical_image_manifest([
            {"sha256": "b" * 64, "media_type": "image/png", "bytes": 100}
        ])
        assert h_a != h_b

    def test_file_sha256_matches_bytes_sha256(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world")
        assert file_sha256(f) == bytes_sha256(b"hello world")


# ---------------------------------------------------------------------------
# Round-trip: write → parse → validate
# ---------------------------------------------------------------------------

def _mk_header(**overrides) -> TraceHeader:
    base = dict(
        trace_id="test-trace-001",
        trial_group_id="test-group-A",
        observation_mode="dv",
        dv_version="1.0.6",
        dv_config_hash="abc123",
        task_id="test-task",
        task_description="unit test trace",
        model="claude-sonnet-4",
    )
    base.update(overrides)
    return TraceHeader(**base)


def _mk_step(idx: int, *, obs_type="delta", **overrides) -> TraceStep:
    base = dict(
        step_idx=idx,
        ts="2026-04-26T00:00:00Z",
        url="https://example.com",
        obs_type=obs_type,
        trigger="diff_below_threshold",
        dv_internal_tokens=1365,
        model_facing_tokens=425 if obs_type == "delta" else 1365,
        frame_sha256="0" * 64,
        payload_image_sha256="1" * 64,
        payload_text_sha256="2" * 64,
        crop_bboxes_px=[[10, 20, 100, 50]] if obs_type == "delta" else [],
    )
    base.update(overrides)
    return TraceStep(**base)


class TestRoundTrip:
    def test_minimal_trace_round_trips(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        with TraceWriter(path, _mk_header()) as w:
            w.write_step(_mk_step(0, obs_type="full_frame"))
            w.write_step(_mk_step(1, obs_type="delta"))
            w.write_step(_mk_step(2, obs_type="delta"))

        trace = parse_trace(path)
        assert trace.header["trace_id"] == "test-trace-001"
        assert trace.header["schema_version"] == SCHEMA_VERSION
        assert len(trace.steps) == 3
        assert trace.summary is not None
        assert trace.summary["n_steps"] == 3

        report = validate_trace(trace, check_paths=False)
        assert report.ok, f"errors: {report.errors}"
        assert report.n_steps == 3
        assert report.total_dv_internal_tokens == 1365 * 3
        assert report.total_model_facing_tokens == 1365 + 425 + 425

    def test_summary_recomputed_savings_matches_writer(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        with TraceWriter(path, _mk_header()) as w:
            w.write_step(_mk_step(0, obs_type="full_frame"))
            w.write_step(_mk_step(1, obs_type="delta"))

        trace = parse_trace(path)
        report = validate_trace(trace, check_paths=False)
        assert report.ok

        # Writer's savings_pct should match what the validator recomputes
        expected = (1 - (1365 + 425) / (1365 * 2)) * 100
        assert abs(trace.summary["savings_pct_total"] - expected) < 0.01
        assert abs(report.savings_pct_total - expected) < 0.01

    def test_writer_skips_summary_on_exception(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        with pytest.raises(RuntimeError):
            with TraceWriter(path, _mk_header()) as w:
                w.write_step(_mk_step(0))
                raise RuntimeError("simulated crash")

        trace = parse_trace(path)
        # Header + 1 step survived to disk, but no summary line — the
        # verifier should treat this as "did not complete cleanly."
        assert len(trace.steps) == 1
        assert trace.summary is None


# ---------------------------------------------------------------------------
# Validator: catches each invariant violation
# ---------------------------------------------------------------------------

def _build_raw_trace(
    header: dict, steps: list[dict], summary: dict | None,
    tmp_path,
) -> ParsedTrace:
    """Write a trace bypassing the dataclass writer so we can construct
    invalid traces and confirm the validator catches them."""
    path = tmp_path / "trace.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"_kind": "header", **header}) + "\n")
        for s in steps:
            f.write(json.dumps({"_kind": "step", **s}) + "\n")
        if summary is not None:
            f.write(json.dumps({"_kind": "summary", **summary}) + "\n")
    return parse_trace(path)


def _good_header_dict() -> dict:
    h = _mk_header()
    return {k: v for k, v in h.__dict__.items()}


def _good_step_dict(idx: int, **overrides) -> dict:
    s = _mk_step(idx, **overrides)
    return {k: v for k, v in s.__dict__.items() if v is not None}


class TestValidator:
    def test_missing_required_header_keys(self, tmp_path):
        h = _good_header_dict()
        del h["task_id"]
        trace = _build_raw_trace(h, [_good_step_dict(0)], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("task_id" in e for e in report.errors)

    def test_invalid_observation_mode(self, tmp_path):
        h = _good_header_dict()
        h["observation_mode"] = "mixed"  # the option we explicitly rejected
        trace = _build_raw_trace(h, [_good_step_dict(0)], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("observation_mode" in e for e in report.errors)

    def test_wrong_schema_version(self, tmp_path):
        h = _good_header_dict()
        h["schema_version"] = "999"
        trace = _build_raw_trace(h, [_good_step_dict(0)], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("schema_version" in e for e in report.errors)

    def test_model_facing_exceeds_dv_internal(self, tmp_path):
        # DV cannot send more than it consumed. Hard error.
        s = _good_step_dict(0)
        s["dv_internal_tokens"] = 100
        s["model_facing_tokens"] = 200
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("model_facing" in e and "dv_internal" in e
                   for e in report.errors)

    def test_full_frame_with_compression_warns(self, tmp_path):
        # full_frame steps should bill what they consumed (no compression).
        # If they don't, that's suspicious enough to warn but not error.
        s = _good_step_dict(0, obs_type="full_frame")
        s["dv_internal_tokens"] = 1365
        s["model_facing_tokens"] = 800   # weird
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        # Not an error (some custom adapter could legitimately do this) but
        # should warn.
        assert any("full_frame" in w for w in report.warnings)

    def test_duplicate_step_idx(self, tmp_path):
        s1 = _good_step_dict(0)
        s2 = _good_step_dict(0)  # duplicate
        trace = _build_raw_trace(_good_header_dict(), [s1, s2], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("duplicate step_idx" in e for e in report.errors)

    def test_malformed_bbox(self, tmp_path):
        s = _good_step_dict(0)
        s["crop_bboxes_px"] = [[10, 20, 100]]  # missing h
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("crop_bboxes_px" in e for e in report.errors)

    def test_summary_total_mismatch(self, tmp_path):
        s = _good_step_dict(0, obs_type="delta")
        s["dv_internal_tokens"] = 1365
        s["model_facing_tokens"] = 425
        bad_summary = {
            "ended_at": "2026-04-26T00:01:00Z",
            "n_steps": 1,
            "total_dv_internal_tokens": 9999,   # wrong
            "total_model_facing_tokens": 425,
            "savings_pct_total": 50.0,
        }
        trace = _build_raw_trace(
            _good_header_dict(), [s], bad_summary, tmp_path,
        )
        report = validate_trace(trace, check_paths=False)
        assert not report.ok
        assert any("total_dv_internal_tokens" in e for e in report.errors)


class TestStructuralErrors:
    def test_missing_header_line(self, tmp_path):
        p = tmp_path / "trace.jsonl"
        p.write_text(json.dumps({"_kind": "step", "step_idx": 0}) + "\n")
        with pytest.raises(ValueError, match="step before header"):
            parse_trace(p)

    def test_duplicate_header(self, tmp_path):
        p = tmp_path / "trace.jsonl"
        h = _good_header_dict()
        p.write_text(
            json.dumps({"_kind": "header", **h}) + "\n"
            + json.dumps({"_kind": "header", **h}) + "\n"
        )
        with pytest.raises(ValueError, match="more than one header"):
            parse_trace(p)

    def test_unknown_kind(self, tmp_path):
        p = tmp_path / "trace.jsonl"
        h = _good_header_dict()
        p.write_text(
            json.dumps({"_kind": "header", **h}) + "\n"
            + json.dumps({"_kind": "diagnostic", "extra": 1}) + "\n"
        )
        with pytest.raises(ValueError, match="unknown _kind"):
            parse_trace(p)


# ---------------------------------------------------------------------------
# File-on-disk hash check
# ---------------------------------------------------------------------------

class TestFilePathChecks:
    def test_frame_path_hash_match_passes(self, tmp_path):
        # Write a real PNG-ish blob, hash it, embed in step, validate.
        frame_bytes = b"\x89PNG\r\n\x1a\nfake png bytes for test"
        frame_file = tmp_path / "frame.png"
        frame_file.write_bytes(frame_bytes)
        real_hash = bytes_sha256(frame_bytes)

        s = _good_step_dict(0)
        s["frame_sha256"] = real_hash
        s["frame_path"] = "frame.png"   # relative to trace
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=True)
        assert report.ok, f"errors: {report.errors}"

    def test_frame_path_hash_mismatch_errors(self, tmp_path):
        frame_file = tmp_path / "frame.png"
        frame_file.write_bytes(b"actual bytes")
        s = _good_step_dict(0)
        s["frame_sha256"] = "f" * 64   # wrong hash
        s["frame_path"] = "frame.png"
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=True)
        assert not report.ok
        assert any("sha256 mismatch" in e for e in report.errors)

    def test_missing_frame_path_errors(self, tmp_path):
        s = _good_step_dict(0)
        s["frame_path"] = "does_not_exist.png"
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=True)
        assert not report.ok
        assert any("not found" in e for e in report.errors)

    def test_check_paths_false_skips_disk_io(self, tmp_path):
        # Even with a bad hash, check_paths=False shouldn't error on it.
        s = _good_step_dict(0)
        s["frame_sha256"] = "f" * 64
        s["frame_path"] = "does_not_exist.png"
        trace = _build_raw_trace(_good_header_dict(), [s], None, tmp_path)
        report = validate_trace(trace, check_paths=False)
        assert report.ok, f"errors: {report.errors}"


# ---------------------------------------------------------------------------
# config_hash determinism
# ---------------------------------------------------------------------------

def test_config_hash_stable_across_calls():
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})  # different insertion order
    assert h1 == h2

def test_config_hash_distinguishes_values():
    assert config_hash({"a": 1}) != config_hash({"a": 2})
