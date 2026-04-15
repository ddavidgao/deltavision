"""
Simple result saver. Writes structured JSON to results/ directory.
No database, no warehouse — just accumulating JSON files.
Query with: find results/ -name '*.json' | xargs jq '.metrics'
"""

import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent


def save_result(
    benchmark: str,
    backend: str,
    metrics: dict,
    transition_log: list = None,
    config: dict = None,
    notes: str = "",
) -> str:
    """
    Save a benchmark result to a JSON file.
    Returns the file path.
    """
    timestamp = datetime.now().isoformat()
    filename = f"{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = RESULTS_DIR / filename

    result = {
        "benchmark": benchmark,
        "timestamp": timestamp,
        "backend": backend,
        "metrics": metrics,
        "config": config or {},
        "notes": notes,
    }

    if transition_log:
        result["transition_log"] = transition_log

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Result saved: {path}")
    return str(path)


def load_results(benchmark: str = None, limit: int = 20) -> list:
    """Load recent results, optionally filtered by benchmark."""
    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files[:limit * 3]:  # oversample then filter
        with open(f) as fh:
            r = json.load(fh)
            if benchmark and r.get("benchmark") != benchmark:
                continue
            r["_file"] = f.name
            results.append(r)
            if len(results) >= limit:
                break
    return results


def summarize(benchmark: str = None):
    """Print a summary table of recent results."""
    results = load_results(benchmark, limit=50)
    if not results:
        print("No results found.")
        return

    print(f"\n{'Benchmark':<25} {'Backend':<20} {'Date':<12} {'Key Metric'}")
    print("-" * 80)
    for r in results:
        date = r["timestamp"][:10]
        metrics = r.get("metrics", {})
        # Pick the most interesting metric to show
        key_metric = ""
        for k in ["avg_reaction_ms", "best_reaction_ms", "delta_ratio",
                   "questions_answered", "steps", "total_time_s"]:
            if k in metrics:
                key_metric = f"{k}={metrics[k]}"
                break
        if not key_metric and metrics:
            k, v = next(iter(metrics.items()))
            key_metric = f"{k}={v}"

        print(f"{r['benchmark']:<25} {r.get('backend', '?'):<20} {date:<12} {key_metric}")
