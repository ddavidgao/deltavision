#!/usr/bin/env python3
"""
Head-to-head: FF (run_19) vs DV (run_18) parallel matched runs.

Both subagents ran in parallel on separate sheets with matched 1280x800 viewport.
Both followed the matched-trajectory rule (Maps sidebar only, no website fallback).
DV proxy had the new token-cap + nudge fixes shipped 2026-04-24.
"""
import json
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DV_LOG = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/dv_proxy_run_1777072627.jsonl"
FF_LOG = ROOT / "benchmarks/mapsheets/results/run_19_ff_parallel/dv_proxy_run_1777072642_ff.jsonl"
DV_SHOTS = ROOT / "benchmarks/mapsheets/results/run_18_dv_parallel/screenshots"
FF_SHOTS = ROOT / "benchmarks/mapsheets/results/run_19_ff_parallel/screenshots"


def load_steps(p):
    out = []
    for line in open(p):
        d = json.loads(line)
        if "step" in d:
            out.append(d)
    return out


def on_task_mask(d):
    files = sorted(os.listdir(d))
    hashes = [hashlib.md5(open(d / f, "rb").read()).hexdigest() for f in files]
    mask = [True]
    for i in range(1, len(hashes)):
        mask.append(hashes[i] != hashes[i - 1])
    return mask, files


def main():
    dv = load_steps(DV_LOG)
    ff = load_steps(FF_LOG)

    dv_mask, _ = on_task_mask(DV_SHOTS)
    ff_mask, _ = on_task_mask(FF_SHOTS)

    # Sanity: in DV, dv_tokens should NEVER exceed FULL_FRAME_TOKENS=1365
    over = [s for s in dv if s["dv_tokens"] > 1365]
    print(f"=== Sanity: DV steps where dv_tokens > 1365 (should be 0): {len(over)} ===")

    dv_total = sum(s["dv_tokens"] for s in dv)
    ff_total = sum(s["dv_tokens"] for s in ff)  # in FF mode, dv_tokens == ff_tokens
    dv_n = len(dv)
    ff_n = len(ff)

    dv_ot = [(s, m) for s, m in zip(dv, dv_mask, strict=True) if m]
    ff_ot = [(s, m) for s, m in zip(ff, ff_mask, strict=True) if m]
    dv_total_ot = sum(s["dv_tokens"] for s, _ in dv_ot)
    ff_total_ot = sum(s["dv_tokens"] for s, _ in ff_ot)

    print()
    print("=" * 76)
    print("FF (run_19) vs DV (run_18) — parallel matched runs, token-cap active")
    print("=" * 76)
    print(f"\n{'':<28}{'FF':>14}{'DV':>14}")
    print(f"Total steps:               {ff_n:>14}{dv_n:>14}")
    print(f"On-task (dedup'd) steps:   {len(ff_ot):>14}{len(dv_ot):>14}")
    print(f"Total tokens:              {ff_total:>14,}{dv_total:>14,}")
    print(f"On-task tokens:            {ff_total_ot:>14,}{dv_total_ot:>14,}")

    print()
    print("--- Framing A: per-step (apples-to-apples per frame sent to model) ---")
    print(f"FF tokens/step (raw):      {ff_total/ff_n:>14.0f}")
    print(f"DV tokens/step (raw):      {dv_total/dv_n:>14.0f}")
    saving_per_step = (1 - (dv_total/dv_n) / (ff_total/ff_n)) * 100
    print(f"Per-step savings:          {saving_per_step:>13.1f}%")

    print()
    print("--- Framing B: total task cost (full trajectory) ---")
    saving_total = (1 - dv_total / ff_total) * 100
    print(f"FF task total:             {ff_total:>14,} tokens")
    print(f"DV task total:             {dv_total:>14,} tokens")
    print(f"Total-task savings:        {saving_total:>13.1f}%")

    print()
    print("--- Framing C: on-task only (dedup'd both sides) ---")
    saving_ot_per_step = (1 - (dv_total_ot/max(len(dv_ot),1)) / (ff_total_ot/max(len(ff_ot),1))) * 100
    print(f"FF tokens/on-task step:    {ff_total_ot/max(len(ff_ot),1):>14.0f}")
    print(f"DV tokens/on-task step:    {dv_total_ot/max(len(dv_ot),1):>14.0f}")
    print(f"On-task per-step savings:  {saving_ot_per_step:>13.1f}%")

    print()
    print("--- Framing D: trajectory-matched (truncate longer to shorter) ---")
    n = min(dv_n, ff_n)
    dv_trunc = sum(s["dv_tokens"] for s in dv[:n])
    ff_trunc = sum(s["dv_tokens"] for s in ff[:n])
    print(f"First {n} steps each:")
    print(f"FF total:                  {ff_trunc:>14,}")
    print(f"DV total:                  {dv_trunc:>14,}")
    print(f"Savings:                   {(1 - dv_trunc/ff_trunc)*100:>13.1f}%")

    print()
    # Trigger breakdown for DV
    print("--- DV trigger breakdown ---")
    triggers = {}
    for s in dv:
        triggers[s["trigger"]] = triggers.get(s["trigger"], 0) + 1
    for t, c in sorted(triggers.items(), key=lambda x: -x[1]):
        print(f"  {t:<24}{c:>4}")

    out = {
        "dv_run": {"path": str(DV_LOG.relative_to(ROOT)), "steps": dv_n,
                   "total_tokens": dv_total, "ontask_steps": len(dv_ot),
                   "ontask_tokens": dv_total_ot},
        "ff_run": {"path": str(FF_LOG.relative_to(ROOT)), "steps": ff_n,
                   "total_tokens": ff_total, "ontask_steps": len(ff_ot),
                   "ontask_tokens": ff_total_ot},
        "framings": {
            "per_step_savings_pct": round(saving_per_step, 1),
            "total_task_savings_pct": round(saving_total, 1),
            "ontask_per_step_savings_pct": round(saving_ot_per_step, 1),
            "first_n_truncated_savings_pct": round((1 - dv_trunc/ff_trunc)*100, 1),
        },
        "sanity": {"dv_steps_over_full_frame_tokens": len(over)},
    }
    out_path = ROOT / "benchmarks/mapsheets/results/ff_vs_dv_parallel.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
