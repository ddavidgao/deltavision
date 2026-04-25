"""
Discovery helpers for real agent-trace fixtures.

Tests don't copy or symlink the benchmark screenshots — that would duplicate
~50MB of data that already lives in version-controlled benchmark dirs. Instead,
this module locates them in place and yields path triples / metadata that
parametrized tests can iterate over.

If a future contributor runs the suite on a clean clone before any benchmarks
have been executed, the discovery functions return empty lists and the
parametrized tests collect zero items (pytest reports them as deselected,
not failed).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class FramePair:
    """One (t0, t1) pair from a real agent trace, ready to feed the classifier."""
    run_name: str       # "run_18_dv_parallel"
    pair_idx: int       # 0-based index of this pair within the run
    t0: Path
    t1: Path
    meta: dict          # arbitrary metadata loaded from sibling JSONL log if available

    @property
    def id(self) -> str:
        """Human-readable test ID for pytest output."""
        return f"{self.run_name}::pair_{self.pair_idx:02d}"


@dataclass
class GeneralizationScenario:
    """One curated scenario from benchmarks/generalization/frames/."""
    name: str           # "wikipedia_nav_article_to_article"
    t0: Path
    t1: Path
    meta: dict          # loaded from meta.json — expected_transition, etc.

    @property
    def id(self) -> str:
        return self.name


def discover_maps_sheets_runs(min_frames: int = 10) -> list[Path]:
    """
    Return paths to every benchmarks/mapsheets/results/run_*/screenshots/
    directory that contains at least `min_frames` PNGs.

    Used to parametrize tests over real agent traces.
    """
    base = REPO_ROOT / "benchmarks" / "mapsheets" / "results"
    if not base.exists():
        return []
    out = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        shots = run_dir / "screenshots"
        if not shots.exists():
            continue
        n = sum(1 for f in shots.iterdir() if f.suffix == ".png")
        if n >= min_frames:
            out.append(shots)
    return out


def iter_consecutive_pairs(screenshots_dir: Path) -> Iterator[FramePair]:
    """
    Yield consecutive (step_N, step_N+1) FramePair items from a run directory.

    Filenames are naturally-sorted (handles both "step_NN.png" and
    "dvp_NN_label.png" patterns) so consecutive ordering is preserved.
    """
    files = sorted(p for p in screenshots_dir.iterdir() if p.suffix == ".png")
    if len(files) < 2:
        return
    run_name = screenshots_dir.parent.name
    log_path = _find_jsonl_for_run(screenshots_dir.parent)
    log_steps = _load_log(log_path) if log_path else {}
    for i in range(len(files) - 1):
        # The log's step N corresponds to file index N (1-indexed in the log,
        # 0-indexed here). Pair i compares file[i] (=step N) to file[i+1] (=step N+1).
        meta = {
            "t0_step": log_steps.get(i + 1),
            "t1_step": log_steps.get(i + 2),
        }
        yield FramePair(
            run_name=run_name,
            pair_idx=i,
            t0=files[i],
            t1=files[i + 1],
            meta=meta,
        )


def discover_generalization_scenarios() -> list[GeneralizationScenario]:
    """
    Return curated scenarios from benchmarks/generalization/frames/.
    Each scenario is a dir containing t0.png, t1.png, and meta.json.
    """
    base = REPO_ROOT / "benchmarks" / "generalization" / "frames"
    if not base.exists():
        return []
    out = []
    for scen_dir in sorted(base.iterdir()):
        if not scen_dir.is_dir():
            continue
        t0 = scen_dir / "t0.png"
        t1 = scen_dir / "t1.png"
        meta_p = scen_dir / "meta.json"
        if not (t0.exists() and t1.exists()):
            continue
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        out.append(GeneralizationScenario(
            name=scen_dir.name, t0=t0, t1=t1, meta=meta,
        ))
    return out


def _find_jsonl_for_run(run_dir: Path) -> Path | None:
    """Heuristic: pick the dv_proxy_run_*.jsonl in the run_dir, or None."""
    candidates = list(run_dir.glob("dv_proxy_run_*.jsonl"))
    if not candidates:
        # Sometimes the log is in the dv_runs/ directory at repo root, not in run_dir
        return None
    return candidates[0]


def _load_log(jsonl_path: Path) -> dict[int, dict]:
    """Load a DV proxy log into a {step: record} dict."""
    out = {}
    if not jsonl_path or not jsonl_path.exists():
        return out
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "step" in rec:
            out[rec["step"]] = rec
    return out
