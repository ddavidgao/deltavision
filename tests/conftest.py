"""
Shared pytest fixtures for the DeltaVision test suite.

Real-screenshot regression fixtures are exposed here. Synthetic-image fixtures
stay inline in the individual test files (they're cheap and self-documenting).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the regression-fixture discovery module importable.
# We intentionally don't put it on sys.path globally; importing here keeps the
# coupling explicit.
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "regression"
sys.path.insert(0, str(_FIXTURE_DIR))

from benchmark_traces import (  # noqa: E402
    FramePair,
    GeneralizationScenario,
    discover_generalization_scenarios,
    discover_maps_sheets_runs,
    iter_consecutive_pairs,
)

# -----------------------------------------------------------------------------
# Curated real_sites fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mcgrawhill_fixture_dir() -> Path:
    """
    Path to the McGraw-Hill regression PNGs (3 files).

    Test using this fixture must check `path.exists()` and skip if absent
    (matches the existing pattern in test_mcgrawhill_real.py).
    """
    return Path(__file__).parent / "fixtures" / "regression" / "real_sites" / "mcgrawhill"


# -----------------------------------------------------------------------------
# Real agent-trace fixtures (parametrize-friendly)
# -----------------------------------------------------------------------------

def maps_sheets_runs() -> list[Path]:
    """
    Module-level helper for parametrize. Returns screenshot dirs from every
    benchmarks/mapsheets/results/run_*/ that has >= 10 frames.

    Use as: @pytest.mark.parametrize("run_dir", maps_sheets_runs(), ids=lambda p: p.parent.name)
    """
    return discover_maps_sheets_runs()


def maps_sheets_pairs() -> list[FramePair]:
    """
    Flatten all consecutive frame pairs across all maps→sheets runs.
    Returns up to several hundred (t0, t1) pairs for batch regression testing.

    Use as: @pytest.mark.parametrize("pair", maps_sheets_pairs(), ids=lambda p: p.id)
    """
    out = []
    for run_dir in discover_maps_sheets_runs():
        out.extend(iter_consecutive_pairs(run_dir))
    return out


def generalization_scenarios() -> list[GeneralizationScenario]:
    """
    Returns the curated benchmarks/generalization/frames/* scenarios with
    a t0/t1 pair and meta.json each.

    Use as: @pytest.mark.parametrize("scenario", generalization_scenarios(), ids=lambda s: s.id)
    """
    return discover_generalization_scenarios()
