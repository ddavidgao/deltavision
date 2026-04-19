"""
ingest.py — take agent_head_to_head summary JSONs, save into ResultStore,
create a comparison row. Returns run IDs so the video composition can
reference them.

Usage:
    python benchmarks/agent_head_to_head/ingest.py \
        --dv <path-to-dv-summary.json> \
        --ff <path-to-ff-summary.json>

Prints:
    DV run id: 42
    FF run id: 43
    Comparison id: 17 (savings: 38.4%)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

from results.store import ResultStore  # noqa: E402


def ingest(dv_json: Path, ff_json: Path) -> tuple[int, int, int]:
    dv = json.loads(dv_json.read_text())
    ff = json.loads(ff_json.read_text())

    store = ResultStore()

    # DV arm run
    dv_id = store.save(
        benchmark="agent_multitab_apartments",
        backend=f"{dv['model']}__dv",
        metrics={
            "steps": dv["steps"],
            "done": dv["done"],
            "crashed": dv.get("crashed", False),
            "total_input_tokens": dv["total_input_tokens"],
            "total_output_tokens": dv["total_output_tokens"],
            "image_tokens_sent_estimate": dv["image_tokens_sent_estimate"],
            "screenshots_taken": dv["screenshots_taken"],
            "elapsed_s": dv["elapsed_s"],
            "max_steps_hit": dv["max_steps_hit"],
        },
        config={
            "task": dv["task"],
            "sheet_url": dv["sheet_url"],
            "arm": "dv",
        },
        transition_log=dv.get("step_log", []),
        notes="Free-running Claude CU. Google Maps -> Google Sheets. DV observation.",
    )
    ff_id = store.save(
        benchmark="agent_multitab_apartments",
        backend=f"{ff['model']}__ff",
        metrics={
            "steps": ff["steps"],
            "done": ff["done"],
            "crashed": ff.get("crashed", False),
            "total_input_tokens": ff["total_input_tokens"],
            "total_output_tokens": ff["total_output_tokens"],
            "image_tokens_sent_estimate": ff["image_tokens_sent_estimate"],
            "screenshots_taken": ff["screenshots_taken"],
            "elapsed_s": ff["elapsed_s"],
            "max_steps_hit": ff["max_steps_hit"],
        },
        config={
            "task": ff["task"],
            "sheet_url": ff["sheet_url"],
            "arm": "ff",
        },
        transition_log=ff.get("step_log", []),
        notes="Free-running Claude CU. Google Maps -> Google Sheets. Raw FF observation (baseline).",
    )

    # Image-token savings (the delta-aware headline)
    savings_pct = round(
        100.0 * (1 - dv["image_tokens_sent_estimate"] / ff["image_tokens_sent_estimate"]),
        1,
    ) if ff["image_tokens_sent_estimate"] > 0 else 0.0

    store.conn.execute(
        """INSERT INTO comparisons (run_a, run_b, improvement_pct, notes)
           VALUES (?, ?, ?, ?)""",
        (dv_id, ff_id, savings_pct,
         f"agent_head_to_head: DV image tokens vs FF image tokens. "
         f"DV steps={dv['steps']} done={dv['done']}. "
         f"FF steps={ff['steps']} done={ff['done']}."),
    )
    store.conn.commit()
    cmp_id = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    print(f"\nDV run id: {dv_id}")
    print(f"FF run id: {ff_id}")
    print(f"Comparison id: {cmp_id} (image-token savings: {savings_pct}%)")
    print(f"\nDV image tokens: {dv['image_tokens_sent_estimate']:,}")
    print(f"FF image tokens: {ff['image_tokens_sent_estimate']:,}")
    print(f"DV steps/done:   {dv['steps']} / {dv['done']}")
    print(f"FF steps/done:   {ff['steps']} / {ff['done']}")

    return dv_id, ff_id, cmp_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dv", type=Path, required=True)
    ap.add_argument("--ff", type=Path, required=True)
    args = ap.parse_args()
    if not args.dv.exists():
        sys.exit(f"not found: {args.dv}")
    if not args.ff.exists():
        sys.exit(f"not found: {args.ff}")
    ingest(args.dv, args.ff)


if __name__ == "__main__":
    main()
