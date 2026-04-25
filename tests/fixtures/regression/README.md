# Regression Test Fixtures

This directory holds the **real-screenshot fixtures** the regression test suite
runs against. The point: catch CV-classifier bugs that synthetic PIL images
silently miss (e.g. the v1.0.4 map-pan misclassification — synthetic tests
all passed; real Maps frame-pairs would have failed).

## Layout

```
tests/fixtures/regression/
├── README.md                # this file
├── real_sites/              # hand-picked frame triples committed to git
│   └── mcgrawhill/          # 3 PNGs from real McGraw-Hill quiz UI
└── benchmark_traces.py      # discovery helpers — points at benchmarks/.../screenshots
                             # (don't duplicate data; tests read directly)
```

## Why two locations

- **`real_sites/`** holds small, curated, manually-vetted frame sets that we
  treat as ground truth for specific regression cases. Committed to git as
  PNGs. Add a new subdir per site/scenario.

- **`benchmark_traces.py`** is a discovery module. Real agent traces from the
  `benchmarks/mapsheets/results/run_*/` directories are large and already
  versioned alongside their proxy logs. Tests **read** them in place — they
  are NOT copied or symlinked here. This avoids double-storage.

## How tests use these

Pytest fixtures defined in `tests/conftest.py`:

- `mcgrawhill_fixtures` — yields paths to the 3 PNGs above.
- `maps_sheets_run_dirs` — parametrize over every `run_*/screenshots/` dir
  under `benchmarks/mapsheets/results/` that has ≥10 screenshots.
- `generalization_scenarios` — parametrize over every dir under
  `benchmarks/generalization/frames/` that has both `t0.png` and `t1.png`.

Each fixture skips gracefully if the source data is missing (e.g. a clean
clone before benchmarks have been run).

## Adding a new fixture set

1. **For a curated regression case:** mkdir under `real_sites/`, add t0/t1
   PNGs + a `meta.json` with `expected_transition`, `expected_diff_ratio`
   bounds, and a one-line description.
2. **For a real agent trace:** save it under
   `benchmarks/<benchmark_name>/results/run_<N>/screenshots/`. The discovery
   helper in `benchmark_traces.py` will pick it up automatically.

## Anti-patterns to avoid

- Don't commit fixtures larger than ~5 MB without thought (use git-lfs or a
  download helper).
- Don't write tests that hard-code expected pixel-exact values — pin
  semantic behavior (transition class, bounded diff ratio) with tolerance.
- Don't use `@pytest.fixture` for parameterized data; use
  `@pytest.mark.parametrize` so failing cases show up by name in the test
  output, not as a single fixture-level error.
