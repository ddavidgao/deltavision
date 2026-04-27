"""
BenchmarkTrace — the single trace format every DeltaVision benchmark emits.

A trace is one run of one (agent_loop_run, observation_mode) pair. Sibling
files share a `trial_group_id` to pair FF and DV runs of the same task.

File format: JSONL, kind-tagged.
    Line 1   : {"_kind": "header", ...}
    Line 2..N: {"_kind": "step",   ...}
    Line N+1 : {"_kind": "summary", ...}     (optional but recommended)

Why JSONL: append-friendly during a live run (proxy can write each step as
it happens, survives mid-run crashes) and the kind-tag lets a streaming
verifier read the header before processing steps.

The schema is validated by `results/trace.py:validate_trace()` and exposed
to users via the `deltavision verify-trace <path>` CLI.

Stable JSON keys (do not rename without bumping schema_version):
    Header:  schema_version, trace_id, trial_group_id, observation_mode,
             dv_version, dv_config_hash, task_id, task_description,
             model, model_seed, started_at, host_arch, host_python
    Step:    step_idx, ts, url, obs_type, trigger,
             dv_internal_tokens, model_facing_tokens,
             frame_sha256, frame_path?,
             payload_images[], payload_image_sha256, payload_text_sha256,
             payload_path?, payload_file_sha256?,
             crop_bboxes_px, crop_bboxes_norm?,
             cv_timing_ms?
    Summary: ended_at, n_steps, total_dv_internal_tokens,
             total_model_facing_tokens, savings_pct_total

The cost-split semantics are critical:
    dv_internal_tokens   = tokens for screenshots DV consumed (always full
                           frame — DV needs the full frame internally to
                           compute the next diff). Equals 1365 × n_steps for
                           a standard 1280×800 viewport.
    model_facing_tokens  = tokens DV actually put in front of the model.
                           1365 on full_frame steps, smaller on delta steps.
                           This is the number that drives savings claims.

When a paper or README says "DV saved X% tokens," that X% MUST be derived
from `model_facing_tokens` totals, not `dv_internal_tokens`.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1"

ObservationMode = Literal["ff", "dv"]
ObsType = Literal["full_frame", "delta"]


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def file_sha256(path: str | Path) -> str:
    """sha256 of the bytes of a file. Used for frame_sha256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    """sha256 of an in-memory blob."""
    return hashlib.sha256(data).hexdigest()


def canonical_image_manifest(
    images: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Hash a list of model-payload image blocks deterministically.

    Each image is a dict with at minimum `sha256`, `media_type`, `bytes`. The
    canonical manifest is the input list with each entry restricted to those
    three keys, in input order. The returned hash is sha256 of the canonical
    JSON-serialized manifest. This handles thumbnail + crops in one shot
    without the brittleness of "concatenate all bytes."

    Returns (payload_image_sha256, canonical_manifest).
    """
    canonical = [
        {
            "sha256": img["sha256"],
            "media_type": img["media_type"],
            "bytes": img["bytes"],
        }
        for img in images
    ]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return bytes_sha256(blob), canonical


def payload_text_sha256(text: str) -> str:
    """sha256 of the text portion of a model payload (UTF-8 encoded)."""
    return bytes_sha256(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TraceHeader:
    """First line of a trace file. Identifies the run and pins reproducibility
    metadata (DV version, config hash, host) so a stale trace can be reproduced
    or its drift located."""
    trace_id: str
    trial_group_id: str
    observation_mode: ObservationMode
    dv_version: str
    dv_config_hash: str
    task_id: str
    task_description: str
    model: str | None = None
    model_seed: int | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    host_arch: str = field(default_factory=platform.machine)
    host_python: str = field(default_factory=lambda: sys.version.split()[0])
    schema_version: str = SCHEMA_VERSION

    def to_jsonl_line(self) -> str:
        return json.dumps({"_kind": "header", **asdict(self)}, sort_keys=True)


@dataclass
class TraceStep:
    """One screenshot's worth of trace data. The step is what the proxy
    writes per `process_screenshot` call.

    Hash policy: `frame_sha256` is required (the bytes DV consumed).
    `payload_image_sha256` is the hash of the canonical payload_images
    manifest, not a raw file hash. If `payload_path` is given,
    `payload_file_sha256` is required and is checked against the bytes on disk.
    """
    step_idx: int
    ts: str
    url: str | None
    obs_type: ObsType
    trigger: str
    dv_internal_tokens: int
    model_facing_tokens: int
    frame_sha256: str
    payload_image_sha256: str
    payload_text_sha256: str
    payload_images: list[dict[str, Any]] = field(default_factory=list)
    crop_bboxes_px: list[list[int]] = field(default_factory=list)
    crop_bboxes_norm: list[list[float]] | None = None
    frame_path: str | None = None
    payload_path: str | None = None
    payload_file_sha256: str | None = None
    cv_timing_ms: dict[str, float] | None = None

    def to_jsonl_line(self) -> str:
        d = {"_kind": "step", **asdict(self)}
        # Strip None-valued optional fields for cleaner traces. Required
        # fields above are non-Optional, so this only affects optional ones.
        d = {k: v for k, v in d.items() if v is not None}
        return json.dumps(d, sort_keys=True)


@dataclass
class TraceSummary:
    """Closing line of a trace file. Optional but strongly recommended —
    lets a streaming run signal "I finished cleanly" and gives the verifier
    a number to compare its own recomputed total against.
    """
    ended_at: str
    n_steps: int
    total_dv_internal_tokens: int
    total_model_facing_tokens: int
    savings_pct_total: float

    def to_jsonl_line(self) -> str:
        return json.dumps({"_kind": "summary", **asdict(self)}, sort_keys=True)


# ---------------------------------------------------------------------------
# Reader / parser
# ---------------------------------------------------------------------------

@dataclass
class ParsedTrace:
    """In-memory view of a parsed trace file."""
    header: dict[str, Any]
    steps: list[dict[str, Any]]
    summary: dict[str, Any] | None
    path: Path


def parse_trace(path: str | Path) -> ParsedTrace:
    """Read a JSONL trace file into a ParsedTrace.

    Raises ValueError on structural issues (missing header, malformed JSON,
    out-of-order kinds). Does NOT validate semantics — that's validate_trace.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"trace file does not exist: {p}")

    header: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None

    for i, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p.name} line {i}: malformed JSON ({e})") from e
        kind = obj.get("_kind")
        if kind == "header":
            if header is not None:
                raise ValueError(f"{p.name}: more than one header line")
            if steps or summary is not None:
                raise ValueError(f"{p.name}: header must come before steps")
            header = obj
        elif kind == "step":
            if header is None:
                raise ValueError(f"{p.name}: step before header")
            if summary is not None:
                raise ValueError(f"{p.name}: step after summary")
            steps.append(obj)
        elif kind == "summary":
            if summary is not None:
                raise ValueError(f"{p.name}: more than one summary line")
            summary = obj
        else:
            raise ValueError(
                f"{p.name} line {i}: unknown _kind {kind!r}; "
                f"expected one of header/step/summary"
            )

    if header is None:
        raise ValueError(f"{p.name}: missing header line")

    return ParsedTrace(header=header, steps=steps, summary=summary, path=p)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_HEADER_KEYS = {
    "_kind", "schema_version", "trace_id", "trial_group_id",
    "observation_mode", "dv_version", "dv_config_hash",
    "task_id", "task_description", "started_at",
    "host_arch", "host_python",
}

REQUIRED_STEP_KEYS = {
    "_kind", "step_idx", "ts", "obs_type", "trigger",
    "dv_internal_tokens", "model_facing_tokens",
    "frame_sha256", "payload_images", "payload_image_sha256", "payload_text_sha256",
    "crop_bboxes_px",
}

REQUIRED_SUMMARY_KEYS = {
    "_kind", "ended_at", "n_steps",
    "total_dv_internal_tokens", "total_model_facing_tokens",
    "savings_pct_total",
}

VALID_OBS_MODES = {"ff", "dv"}
VALID_OBS_TYPES = {"full_frame", "delta"}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    n_steps: int
    total_dv_internal_tokens: int
    total_model_facing_tokens: int
    savings_pct_total: float


def validate_trace(
    trace: ParsedTrace, *, check_paths: bool = True,
) -> ValidationReport:
    """Run every invariant the schema makes against the parsed trace.

    Args:
        trace: a ParsedTrace from parse_trace().
        check_paths: if True and a step has frame_path / payload_path,
            verify the file exists and its sha256 matches the recorded file hash.

    Returns a ValidationReport. ok=True iff errors is empty. Warnings flag
    things that don't violate the schema but look suspicious (savings_pct
    mismatched between summary and recomputed, model_facing > dv_internal,
    etc.).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Header invariants ---
    h = trace.header
    missing = REQUIRED_HEADER_KEYS - h.keys()
    if missing:
        errors.append(f"header: missing required keys {sorted(missing)}")
    if h.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"header: schema_version={h.get('schema_version')!r}; "
            f"this validator only knows {SCHEMA_VERSION!r}"
        )
    if h.get("observation_mode") not in VALID_OBS_MODES:
        errors.append(
            f"header: observation_mode={h.get('observation_mode')!r}; "
            f"must be one of {sorted(VALID_OBS_MODES)}"
        )

    # --- Step invariants ---
    total_dv = 0
    total_mf = 0
    seen_idx: set[int] = set()
    for i, s in enumerate(trace.steps):
        prefix = f"step[{i}]"
        smissing = REQUIRED_STEP_KEYS - s.keys()
        if smissing:
            errors.append(f"{prefix}: missing required keys {sorted(smissing)}")
            continue
        if s["obs_type"] not in VALID_OBS_TYPES:
            errors.append(
                f"{prefix}: obs_type={s['obs_type']!r}; "
                f"must be one of {sorted(VALID_OBS_TYPES)}"
            )
        if s["step_idx"] in seen_idx:
            errors.append(f"{prefix}: duplicate step_idx={s['step_idx']}")
        seen_idx.add(s["step_idx"])

        payload_images = s.get("payload_images")
        if not isinstance(payload_images, list):
            errors.append(f"{prefix}: payload_images must be a list")
        else:
            try:
                actual_payload_hash, _ = canonical_image_manifest(payload_images)
            except KeyError as e:
                errors.append(
                    f"{prefix}: payload_images entries must include "
                    f"sha256/media_type/bytes; missing {e}"
                )
            else:
                if actual_payload_hash != s["payload_image_sha256"]:
                    errors.append(
                        f"{prefix}: payload_image_sha256 mismatch — "
                        f"trace says {s['payload_image_sha256']}, "
                        f"payload_images canonical manifest is {actual_payload_hash}"
                    )

        dv_t = int(s["dv_internal_tokens"])
        mf_t = int(s["model_facing_tokens"])
        total_dv += dv_t
        total_mf += mf_t

        # Cost-split sanity: DV consumed at least as much as it sent.
        if mf_t > dv_t:
            errors.append(
                f"{prefix}: model_facing_tokens ({mf_t}) > "
                f"dv_internal_tokens ({dv_t}) — DV cannot send more than it "
                f"consumed."
            )

        # On full_frame steps, DV sent what it consumed (no compression).
        if s["obs_type"] == "full_frame" and mf_t != dv_t:
            warnings.append(
                f"{prefix}: full_frame step has model_facing != dv_internal "
                f"({mf_t} vs {dv_t}). Either it's a custom payload or a bug."
            )

        # bbox shape
        for j, bbox in enumerate(s.get("crop_bboxes_px", [])):
            if not (isinstance(bbox, list) and len(bbox) == 4):
                errors.append(
                    f"{prefix}: crop_bboxes_px[{j}] not [x,y,w,h]: {bbox!r}"
                )

        # Optional file checks
        if check_paths:
            path_hash_pairs = [("frame_path", "frame_sha256")]
            if "payload_path" in s:
                if "payload_file_sha256" not in s:
                    errors.append(
                        f"{prefix}: payload_path is set but payload_file_sha256 "
                        f"is missing"
                    )
                else:
                    path_hash_pairs.append(
                        ("payload_path", "payload_file_sha256")
                    )
            for path_key, hash_key in path_hash_pairs:
                if path_key in s and hash_key in s:
                    fp = _resolve(trace.path, s[path_key])
                    if not fp.exists():
                        errors.append(
                            f"{prefix}: {path_key}={s[path_key]!r} not found "
                            f"(resolved to {fp})"
                        )
                    else:
                        actual = file_sha256(fp)
                        if actual != s[hash_key]:
                            errors.append(
                                f"{prefix}: {hash_key} mismatch — "
                                f"trace says {s[hash_key]}, file is {actual}"
                            )

    # --- Summary invariants (optional line) ---
    if trace.summary is not None:
        sm = trace.summary
        smissing = REQUIRED_SUMMARY_KEYS - sm.keys()
        if smissing:
            errors.append(f"summary: missing required keys {sorted(smissing)}")
        else:
            if sm["n_steps"] != len(trace.steps):
                errors.append(
                    f"summary: n_steps={sm['n_steps']} but trace has "
                    f"{len(trace.steps)} step records"
                )
            if sm["total_dv_internal_tokens"] != total_dv:
                errors.append(
                    f"summary: total_dv_internal_tokens={sm['total_dv_internal_tokens']} "
                    f"but recomputed sum is {total_dv}"
                )
            if sm["total_model_facing_tokens"] != total_mf:
                errors.append(
                    f"summary: total_model_facing_tokens={sm['total_model_facing_tokens']} "
                    f"but recomputed sum is {total_mf}"
                )
            recomputed_savings = (
                (1 - total_mf / total_dv) * 100 if total_dv else 0.0
            )
            if abs(sm["savings_pct_total"] - recomputed_savings) > 0.05:
                errors.append(
                    f"summary: savings_pct_total={sm['savings_pct_total']:.3f}% "
                    f"but recomputed is {recomputed_savings:.3f}%"
                )
    else:
        warnings.append("trace has no summary line; run may be incomplete")

    savings = (1 - total_mf / total_dv) * 100 if total_dv else 0.0
    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        n_steps=len(trace.steps),
        total_dv_internal_tokens=total_dv,
        total_model_facing_tokens=total_mf,
        savings_pct_total=savings,
    )


def _resolve(trace_path: Path, ref: str) -> Path:
    """Resolve a frame_path/payload_path reference. Absolute paths are taken
    as-is; relative paths are interpreted relative to the trace file's
    directory (the natural convention when a trace lives next to its
    screenshots directory)."""
    p = Path(ref)
    if p.is_absolute():
        return p
    return trace_path.parent / p


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class TraceWriter:
    """Append-only writer for a single trace file. Use as a context manager
    so `__exit__` writes the summary line.

    Example:
        with TraceWriter(path, header) as w:
            for step in run():
                w.write_step(step)
        # summary auto-written at __exit__
    """

    def __init__(self, path: str | Path, header: TraceHeader):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header = header
        self._n_steps = 0
        self._total_dv = 0
        self._total_mf = 0
        self._fh = self.path.open("w")
        self._fh.write(header.to_jsonl_line() + "\n")
        self._fh.flush()

    def write_step(self, step: TraceStep) -> None:
        self._fh.write(step.to_jsonl_line() + "\n")
        self._fh.flush()
        self._n_steps += 1
        self._total_dv += step.dv_internal_tokens
        self._total_mf += step.model_facing_tokens

    def close(self, *, write_summary: bool = True) -> TraceSummary | None:
        if self._fh.closed:
            return None
        summary = None
        if write_summary:
            savings = (
                (1 - self._total_mf / self._total_dv) * 100
                if self._total_dv else 0.0
            )
            summary = TraceSummary(
                ended_at=datetime.now(UTC).isoformat(),
                n_steps=self._n_steps,
                total_dv_internal_tokens=self._total_dv,
                total_model_facing_tokens=self._total_mf,
                savings_pct_total=round(savings, 3),
            )
            self._fh.write(summary.to_jsonl_line() + "\n")
        self._fh.close()
        return summary

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # On exception, still close the file but skip the summary so a
        # crashed trace doesn't claim it finished cleanly. The verifier
        # treats "no summary" as "run did not complete," which is correct.
        self.close(write_summary=exc_type is None)


def config_hash(config_obj: Any) -> str:
    """Stable hash of a DeltaVisionConfig instance (or any dataclass).

    Used to populate header.dv_config_hash so a trace records exactly which
    threshold values produced its outputs. If two traces share a
    dv_config_hash they had identical CV settings; if they differ, that's
    the first thing to investigate when comparing.
    """
    if hasattr(config_obj, "__dataclass_fields__"):
        d = asdict(config_obj)
    elif isinstance(config_obj, dict):
        d = config_obj
    else:
        d = {"repr": repr(config_obj)}
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return bytes_sha256(blob)[:16]   # short prefix for readability
