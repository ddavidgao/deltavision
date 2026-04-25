#!/usr/bin/env python3
"""
Offline classifier regression harness.

Replays:
  1. Generalization frames (6 scenarios, ground truth in meta.json)
  2. SF DV Maps->Sheets run (29 consecutive step pairs)

Under multiple WARP_MIN_INLIER_RATIO values.
No network, no API, no Playwright. Just saved PNGs and classifier logic.

Run: .venv/bin/python3.14 benchmarks/ablation/replay_classifier_offline.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import DeltaVisionConfig
from vision.classifier import classify_transition, extract_anchor


GEN_DIR = ROOT / "benchmarks" / "generalization" / "frames"
SF_DIR = ROOT / "benchmarks" / "mapsheets" / "results" / "run_02" / "screenshots"


def replay_generalization(config: DeltaVisionConfig):
    """Replay the 6 captured scenarios. Returns list of (name, expected, actual, trigger)."""
    results = []
    for scen_dir in sorted(GEN_DIR.iterdir()):
        meta_path = scen_dir / "meta.json"
        t0_path = scen_dir / "t0.png"
        t1_path = scen_dir / "t1.png"
        if not (meta_path.exists() and t0_path.exists() and t1_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        t0 = Image.open(t0_path).convert("RGB")
        t1 = Image.open(t1_path).convert("RGB")
        anchor = extract_anchor(t0, config)
        r = classify_transition(
            t0, t1,
            url_before=meta.get("url_before", ""),
            url_after=meta.get("url_after", ""),
            anchor_template=anchor,
            config=config,
            last_action_type=meta.get("last_action_type"),
        )
        results.append({
            "name": scen_dir.name,
            "expected": meta["classification"],
            "actual": r.transition.value,
            "trigger": r.trigger,
            "diff": round(r.diff_ratio, 3),
            "phash": r.phash_distance,
            "correct": meta["classification"] == r.transition.value,
        })
    return results


def replay_sf_run(config: DeltaVisionConfig):
    """Replay consecutive step pairs from SF run. Ground truth = the live run's log."""
    log_path = ROOT / "dv_runs" / "dv_proxy_run_1776665530.jsonl"
    live_log = {}
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "step" not in rec:
            continue
        live_log[rec["step"]] = rec

    pngs = sorted(SF_DIR.glob("step_*.png"))
    results = []
    for i in range(len(pngs) - 1):
        t0 = Image.open(pngs[i]).convert("RGB")
        t1 = Image.open(pngs[i + 1]).convert("RGB")
        anchor = extract_anchor(t0, config)
        r = classify_transition(
            t0, t1,
            url_before="",  # no URL info in screenshots; fine — step_5-8 URL unchanged in live run
            url_after="",
            anchor_template=anchor,
            config=config,
            last_action_type=None,
        )
        live_step = i + 2  # step_00 vs step_01 = live run's "step 2" (step 1 is initial)
        live = live_log.get(live_step, {})
        results.append({
            "pair": f"{i:02d}->{i+1:02d}",
            "live_step": live_step,
            "live_transition": live.get("transition", "?"),
            "live_trigger": live.get("trigger", "?"),
            "offline_transition": r.transition.value,
            "offline_trigger": r.trigger,
            "diff": round(r.diff_ratio, 3),
        })
    return results


def run(warp_ratio: float):
    cfg = DeltaVisionConfig(WARP_MIN_INLIER_RATIO=warp_ratio)
    print(f"\n===== WARP_MIN_INLIER_RATIO = {warp_ratio} =====")

    print("\n-- Generalization frames --")
    gen = replay_generalization(cfg)
    for r in gen:
        ok = "OK" if r["correct"] else "XX"
        print(f"  [{ok}] {r['name']:<45} expected={r['expected']:<9} got={r['actual']:<9} trigger={r['trigger']}")
    correct = sum(1 for r in gen if r["correct"])
    print(f"  Generalization: {correct}/{len(gen)}")

    print("\n-- SF Maps->Sheets run (29 consecutive pairs) --")
    sf = replay_sf_run(cfg)
    flips = 0
    for r in sf:
        same = r["live_transition"] == r["offline_transition"]
        mark = "  " if same else "<>"
        if not same:
            flips += 1
        print(f"  {mark} step{r['live_step']:>2}  live={r['live_transition']:<9} ({r['live_trigger']:<14}) "
              f"offline={r['offline_transition']:<9} ({r['offline_trigger']:<16}) diff={r['diff']}")
    print(f"  SF flips vs live log: {flips}")

    # Count new_pages
    live_np = sum(1 for r in sf if r["live_transition"] == "new_page")
    offline_np = sum(1 for r in sf if r["offline_transition"] == "new_page")
    print(f"  NEW_PAGE count: live={live_np}  offline={offline_np}")
    return gen, sf


if __name__ == "__main__":
    # Legacy gated behavior at the old production value
    run(0.5)
    # Legacy gated behavior at intermediate value
    run(0.2)
    # NEW default: residual-first — always try warp, keep if it helps
    run(0.0)
