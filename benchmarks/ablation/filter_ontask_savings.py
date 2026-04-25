#!/usr/bin/env python3
"""
Post-filter analysis: recompute DV savings excluding off-task / redundant steps.

"Off-task" = the step is byte-identical to the previous step's screenshot
            (agent took the same shot twice, no work happened).

This gives an "on-task only" savings figure to complement the raw 40.5%.
"""
import json
import hashlib
import os

SHOTS_DIR = "benchmarks/mapsheets/results/run_14_sf_fixed/screenshots"
LOG_PATH = "dv_runs/dv_proxy_run_1777061077.jsonl"


def hash_file(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    files = sorted(os.listdir(SHOTS_DIR))
    hashes = [hash_file(os.path.join(SHOTS_DIR, f)) for f in files]

    steps = []
    with open(LOG_PATH) as fh:
        for line in fh:
            d = json.loads(line)
            if "step" in d:
                steps.append(d)

    # Align: steps[i] corresponds to files[i]
    assert len(steps) == len(files), f"{len(steps)} log steps vs {len(files)} files"

    # Mark each step: on_task=False if identical to previous frame
    on_task = [True]  # step 1 is always on-task
    for i in range(1, len(files)):
        on_task.append(hashes[i] != hashes[i - 1])

    # Recompute cumulative FF / DV over on-task steps only
    ff_cum_raw = 0
    dv_cum_raw = 0
    ff_cum_ot = 0
    dv_cum_ot = 0
    n_ot = 0
    off_task_steps = []

    for i, s in enumerate(steps):
        ff_cum_raw += s["ff_tokens"]
        dv_cum_raw += s["dv_tokens"]
        if on_task[i]:
            ff_cum_ot += s["ff_tokens"]
            dv_cum_ot += s["dv_tokens"]
            n_ot += 1
        else:
            off_task_steps.append((s["step"], files[i], s["dv_tokens"]))

    raw_savings = (1 - dv_cum_raw / max(ff_cum_raw, 1)) * 100
    ot_savings = (1 - dv_cum_ot / max(ff_cum_ot, 1)) * 100

    print("=" * 70)
    print("RUN_14 SF Maps->Sheets — on-task-filtered savings")
    print("=" * 70)
    print(f"\nTotal steps:        {len(steps)}")
    print(f"On-task steps:      {n_ot}")
    print(f"Off-task (dupes):   {len(off_task_steps)}")

    print("\n--- RAW (all steps) ---")
    print(f"FF cumulative:  {ff_cum_raw:>7,}")
    print(f"DV cumulative:  {dv_cum_raw:>7,}")
    print(f"Savings:        {raw_savings:>5.1f}%")

    print("\n--- ON-TASK ONLY ---")
    print(f"FF cumulative:  {ff_cum_ot:>7,}")
    print(f"DV cumulative:  {dv_cum_ot:>7,}")
    print(f"Savings:        {ot_savings:>5.1f}%")

    print("\n--- Off-task steps (redundant screenshots) ---")
    for step, fname, dv_t in off_task_steps:
        print(f"  step {step:>2}  {fname:<35}  dv_tok={dv_t}")

    # Also export the on-task mask for downstream use (e.g. video regeneration)
    out = {
        "total_steps": len(steps),
        "on_task_steps": n_ot,
        "off_task_steps": len(off_task_steps),
        "raw_savings_pct": round(raw_savings, 1),
        "on_task_savings_pct": round(ot_savings, 1),
        "ff_cum_raw": ff_cum_raw,
        "dv_cum_raw": dv_cum_raw,
        "ff_cum_ontask": ff_cum_ot,
        "dv_cum_ontask": dv_cum_ot,
        "on_task_mask": on_task,
        "off_task_details": [
            {"step": s, "file": f, "dv_tokens": t}
            for s, f, t in off_task_steps
        ],
    }
    out_path = "benchmarks/mapsheets/results/run_14_sf_fixed/filtered_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
