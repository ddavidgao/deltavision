#!/usr/bin/env python3
"""
Head-to-head FF vs DV comparison for SF Maps->Sheets task.

Compares:
  - run_14_sf_fixed (DV, 46 steps, 40.5% savings)
  - run_15_sf_ff    (FF, 29 steps, baseline)

Reports multiple savings framings since the agents took different-length trajectories.
"""
import json
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_14_sf_fixed/dv_proxy_run_1777061077.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_15_sf_ff/dv_proxy_run_1777066006_ff.jsonl"
DV_SHOTS = ROOT / "benchmarks/mapsheets/results/run_14_sf_fixed/screenshots"
FF_SHOTS = ROOT / "benchmarks/mapsheets/results/run_15_sf_ff/screenshots"


def load_steps(path):
    steps = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if "step" in d:
                steps.append(d)
    return steps


def on_task_mask(shot_dir):
    """Mark each screenshot as on-task (not byte-identical to previous)."""
    files = sorted(os.listdir(shot_dir))
    hashes = []
    for f in files:
        with open(shot_dir / f, "rb") as fh:
            hashes.append(hashlib.md5(fh.read()).hexdigest())
    mask = [True]
    for i in range(1, len(hashes)):
        mask.append(hashes[i] != hashes[i - 1])
    return mask, files


def totals(steps, mask=None):
    ff = dv = 0
    n = 0
    for i, s in enumerate(steps):
        if mask is not None and not mask[i]:
            continue
        ff += s["ff_tokens"]
        dv += s["dv_tokens"]
        n += 1
    return ff, dv, n


def main():
    dv_steps = load_steps(DV_LOG)
    ff_steps = load_steps(FF_LOG)

    dv_mask, dv_files = on_task_mask(DV_SHOTS)
    ff_mask, ff_files = on_task_mask(FF_SHOTS)

    assert len(dv_mask) == len(dv_steps), f"{len(dv_mask)} dv masks vs {len(dv_steps)} steps"
    assert len(ff_mask) == len(ff_steps), f"{len(ff_mask)} ff masks vs {len(ff_steps)} steps"

    dv_ff_raw, dv_dv_raw, n_dv_raw = totals(dv_steps)
    dv_ff_ot, dv_dv_ot, n_dv_ot = totals(dv_steps, dv_mask)
    ff_ff_raw, ff_dv_raw, n_ff_raw = totals(ff_steps)
    ff_ff_ot, ff_dv_ot, n_ff_ot = totals(ff_steps, ff_mask)

    print("=" * 72)
    print("FF vs DV — SF Maps->Sheets")
    print("=" * 72)

    print(f"\n                       {'DV run_14':>16}  {'FF run_15':>16}")
    print(f"Total steps:           {n_dv_raw:>16}  {n_ff_raw:>16}")
    print(f"On-task steps:         {n_dv_ot:>16}  {n_ff_ot:>16}")
    print(f"Duplicates:            {n_dv_raw - n_dv_ot:>16}  {n_ff_raw - n_ff_ot:>16}")

    print("\n--- Framing A: self-reported tokens (each run's own proxy) ---")
    print(f"{'':<22} DV_run                FF_run")
    print(f"Raw total:             FF_sim={dv_ff_raw:>6,}  DV={dv_dv_raw:>6,}    "
          f"FF={ff_ff_raw:>6,}")
    print(f"DV savings vs its own synthetic FF: "
          f"{(1 - dv_dv_raw / max(dv_ff_raw, 1)) * 100:.1f}%")

    print("\n--- Framing B: real DV vs real FF (different trajectories!) ---")
    print(f"FF actual tokens:      {ff_ff_raw:>7,} ({n_ff_raw} steps)")
    print(f"DV actual tokens:      {dv_dv_raw:>7,} ({n_dv_raw} steps)")
    print(f"Tokens per step, FF:   {ff_ff_raw / n_ff_raw:>7.0f}")
    print(f"Tokens per step, DV:   {dv_dv_raw / n_dv_raw:>7.0f}")
    print(f"Per-step savings:      "
          f"{(1 - (dv_dv_raw / n_dv_raw) / (ff_ff_raw / n_ff_raw)) * 100:.1f}%  "
          f"(this is the cleanest apples-to-apples)")

    if n_dv_raw > n_ff_raw:
        # DV took more steps — to compare directly, truncate DV to same step count as FF
        dv_truncated_dv = sum(s["dv_tokens"] for s in dv_steps[:n_ff_raw])
        dv_truncated_ff_sim = sum(s["ff_tokens"] for s in dv_steps[:n_ff_raw])
        print(f"\n--- Framing C: truncate DV to FF's step count ({n_ff_raw} steps) ---")
        print(f"FF actual:             {ff_ff_raw:>7,}")
        print(f"DV truncated:          {dv_truncated_dv:>7,}")
        print(f"Savings:               {(1 - dv_truncated_dv / ff_ff_raw) * 100:.1f}%")

    print("\n--- Framing D: on-task only (deduplicate both sides) ---")
    print(f"FF on-task tokens:     {ff_ff_ot:>7,} ({n_ff_ot} steps)")
    print(f"DV on-task tokens:     {dv_dv_ot:>7,} ({n_dv_ot} steps)")
    print(f"Per-step, FF on-task:  {ff_ff_ot / max(n_ff_ot, 1):>7.0f}")
    print(f"Per-step, DV on-task:  {dv_dv_ot / max(n_dv_ot, 1):>7.0f}")
    print(f"On-task per-step save: "
          f"{(1 - (dv_dv_ot / max(n_dv_ot, 1)) / (ff_ff_ot / max(n_ff_ot, 1))) * 100:.1f}%")

    out = {
        "dv_run": {
            "log": str(DV_LOG.relative_to(ROOT)),
            "total_steps": n_dv_raw,
            "on_task_steps": n_dv_ot,
            "dv_tokens": dv_dv_raw,
            "ff_sim_tokens": dv_ff_raw,
        },
        "ff_run": {
            "log": str(FF_LOG.relative_to(ROOT)),
            "total_steps": n_ff_raw,
            "on_task_steps": n_ff_ot,
            "ff_tokens": ff_ff_raw,
        },
        "framings": {
            "a_self_reported_dv_savings_pct": round((1 - dv_dv_raw / max(dv_ff_raw, 1)) * 100, 1),
            "b_per_step_savings_pct": round((1 - (dv_dv_raw / n_dv_raw) / (ff_ff_raw / n_ff_raw)) * 100, 1),
            "d_on_task_per_step_savings_pct": round((1 - (dv_dv_ot / max(n_dv_ot, 1)) / (ff_ff_ot / max(n_ff_ot, 1))) * 100, 1),
        },
    }
    out_path = ROOT / "benchmarks/mapsheets/results/ff_vs_dv_comparison.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
