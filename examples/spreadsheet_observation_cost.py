"""
Scripted, deterministic spreadsheet observation-cost benchmark.

Why this exists
---------------
Our earlier subagent-driven Google Sheets runs (v6/v7) produced token savings
numbers that looked great — but they depend on which decisions the subagent
makes, so two runs of "the same task" give different trajectories and
different numbers. That breaks the reproducibility discipline every other
DV benchmark follows.

This file fixes that: it drives a **local HTML mock spreadsheet** (checked
in at `examples/assets/sheet_mock.html`) through a fixed, deterministic
action sequence — six headers, three data rows, all typed in a canonical
order with Tab/Enter navigation. Same inputs → same screenshots → same
observations → same numbers, every run, on any machine. No Google account,
no login wall, no Cloudflare.

What it measures
----------------
For every step of the scripted trajectory, two distinct token costs:

  - dv_internal_tokens  — what DV consumed internally (always full-frame
                          size). DV needs the full screenshot to compute
                          the next diff; this is the infrastructure cost.
                          Equivalent to the old "ff_tokens" / FF baseline.
  - model_facing_tokens — what DV actually shipped to the model (smaller
                          than dv_internal on delta steps). This is the
                          number that drives savings claims.

Savings = 1 - sum(model_facing_tokens) / sum(dv_internal_tokens).

This file used to call those numbers `ff_tokens` and `dv_tokens`, which
ambiguously conflated "cost of the screenshot DV consumed" with "cost of
DV-the-product's payload to the model." The output JSON keeps the legacy
keys (`ff_tokens`, `dv_tokens`, `ff_total_tokens`, `dv_total_tokens`,
`total_savings_pct`) as aliases for one release so existing CI gates and
downstream tooling don't break, but new code should read the explicit
keys: `dv_internal_tokens`, `model_facing_tokens`, etc.

Outputs:
  - examples/spreadsheet_observation_cost.json  — per-step numbers
  - examples/spreadsheet_frames/step_NN.png     — captured screenshots

Reproducibility check: run twice, diff the JSON — the only drift should be
in absolute pHash values if rendering fonts vary across machines; the
per-step obs_type and trigger must match exactly.

Run:
    python examples/spreadsheet_observation_cost.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.async_api import async_playwright

from observer import DeltaVisionObserver

# =============================================================================
# Trajectory — fixed, deterministic, NO subagent involved
# =============================================================================

HEADERS = ["Address", "Price", "Neighborhood", "Amenities", "URL", "Notes"]

ROWS = [
    [
        "1277 E 14th St Brooklyn NY",
        "3405",
        "Prospect Park",
        "W/D in unit, dishwasher, AC",
        "https://apartments.com/vitagraph",
        "Closest to subway",
    ],
    [
        "1515 Surf Ave Brooklyn NY",
        "3199",
        "Coney Island",
        "W/D in unit, pool, doorman",
        "https://apartments.com/1515-surf",
        "Best price for 2BR",
    ],
    [
        "1 Ocean Dr Brooklyn NY",
        "3995",
        "Brighton Beach",
        "W/D in unit, waterfront, pool",
        "https://apartments.com/ocean-dr",
        "Most amenities",
    ],
]


# =============================================================================
# Measurement
# =============================================================================
#
# The cost-split numbers come straight off the DVObservation:
#   obs.dv_internal_tokens()   — what DV consumed (always full-frame size)
#   obs.model_facing_tokens()  — what DV shipped to the model
# Both go through observer._image_tokens_from_size() / _image_tokens(), which
# uses the same Anthropic-style formula max(75, w*h/750) we used to compute
# inline here, but anchored to the canonical observer code path so a future
# token-cost rescale only has to happen in one place.

async def _capture(page) -> bytes:
    return await page.screenshot(type="png", full_page=False)


async def run_benchmark(examples_dir: Path) -> dict:
    examples_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = examples_dir / "spreadsheet_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = examples_dir / "spreadsheet_obs"
    obs_dir.mkdir(parents=True, exist_ok=True)

    mock_url = (examples_dir / "assets" / "sheet_mock.html").resolve().as_uri()

    observer = DeltaVisionObserver()

    steps: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--window-size=1280,800"],
        )
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        await page.goto(mock_url, wait_until="domcontentloaded")
        await page.wait_for_selector("input[data-row='0'][data-col='0']")

        async def snap_and_measure(step_idx: int, action_desc: str) -> None:
            png = await _capture(page)
            (frames_dir / f"step_{step_idx:02d}.png").write_bytes(png)
            obs = observer.observe(png, url=mock_url, last_action=action_desc)
            # Cost split (preferred):
            #   dv_internal_tokens  = what DV consumed (always full-frame)
            #   model_facing_tokens = what DV shipped to the model
            dv_internal_tokens = obs.dv_internal_tokens()
            model_facing_tokens = obs.model_facing_tokens()
            # Legacy aliases (kept for one release for back-compat):
            #   ff_tokens = dv_internal_tokens (the "FF baseline" was always
            #               the cost of the full frame DV consumed)
            #   dv_tokens = model_facing_tokens (what DV-the-product billed)
            ff_tokens = dv_internal_tokens
            dv_tokens = model_facing_tokens
            # Serialize crop bounding boxes for downstream visualization.
            # crop_bboxes is a list of (x1, y1, x2, y2) tuples in viewport coords.
            crop_bboxes = [list(b) for b in (obs.crop_bboxes or [])]

            # Save what DV actually emits: the thumbnail it sends to the model,
            # and the per-crop PNGs. Stored alongside the raw capture so the
            # video can visualize "what FF sends" vs "what DV sends" side by side.
            step_obs_dir = obs_dir / f"step_{step_idx:02d}"
            step_obs_dir.mkdir(parents=True, exist_ok=True)
            crop_files = []
            if obs.thumbnail is not None:
                thumb_path = step_obs_dir / "thumbnail.png"
                obs.thumbnail.save(thumb_path)
            if obs.crops:
                for i, crop in enumerate(obs.crops):
                    cp = step_obs_dir / f"crop_{i:02d}.png"
                    crop.save(cp)
                    crop_files.append(f"spreadsheet_obs/step_{step_idx:02d}/crop_{i:02d}.png")
            # If DV emitted a full frame (only happens on initial + NEW_PAGE),
            # save that too for the video to use.
            if obs.obs_type == "full_frame":
                frame_path = step_obs_dir / "full_frame.png"
                if hasattr(obs, "frame") and obs.frame is not None:
                    obs.frame.save(frame_path)

            savings_pct = round(
                (1 - model_facing_tokens / dv_internal_tokens) * 100, 1
            ) if dv_internal_tokens else 0.0
            steps.append({
                "step": step_idx,
                "action": action_desc,
                "obs_type": obs.obs_type,
                "trigger": obs.trigger,
                "diff_ratio": round(obs.diff_ratio or 0.0, 4),
                "phash_distance": obs.phash_distance,
                "anchor_score": (
                    round(obs.anchor_score, 4)
                    if obs.anchor_score is not None
                    else None
                ),
                "crop_bboxes": crop_bboxes,
                "crop_files": crop_files,
                "thumbnail_file": f"spreadsheet_obs/step_{step_idx:02d}/thumbnail.png"
                    if obs.thumbnail is not None else None,
                # Cost split (preferred names)
                "dv_internal_tokens": dv_internal_tokens,
                "model_facing_tokens": model_facing_tokens,
                "savings_pct": savings_pct,
                # Back-compat aliases (deprecated; remove in v1.1.0)
                "ff_tokens": ff_tokens,
                "dv_tokens": dv_tokens,
            })

        # Step 0 — initial state, no edits yet
        step = 0
        await snap_and_measure(step, "load")

        # Fill header row (A1..F1). Tab after each cell except the last; Enter at end.
        for col, header in enumerate(HEADERS):
            step += 1
            await page.keyboard.type(header, delay=15)
            if col < len(HEADERS) - 1:
                await page.keyboard.press("Tab")
                await snap_and_measure(step, f"header {chr(65 + col)}1 + Tab")
            else:
                await page.keyboard.press("Enter")
                await snap_and_measure(step, "header F1 + Enter (wraps to A2)")

        # Fill 3 data rows (A2..F4). Tab between cells; Enter at row end.
        for r_idx, row in enumerate(ROWS):
            for c_idx, val in enumerate(row):
                step += 1
                await page.keyboard.type(val, delay=8)
                if c_idx < len(row) - 1:
                    await page.keyboard.press("Tab")
                    await snap_and_measure(
                        step, f"row {r_idx+2} col {chr(65 + c_idx)} + Tab"
                    )
                else:
                    await page.keyboard.press("Enter")
                    await snap_and_measure(step, f"row {r_idx+2} col F + Enter")

        await browser.close()

    # Aggregate. Cost-split totals are the canonical numbers; legacy
    # ff_total / dv_total are kept as aliases of dv_internal_total /
    # model_facing_total respectively, for one release of back-compat
    # (CI assertions and downstream readers still on the old key names).
    dv_internal_total = sum(s["dv_internal_tokens"] for s in steps)
    model_facing_total = sum(s["model_facing_tokens"] for s in steps)
    total_savings = (
        round((1 - model_facing_total / dv_internal_total) * 100, 1)
        if dv_internal_total else 0.0
    )
    n_steps = max(1, len(steps))
    per_step_internal = dv_internal_total / n_steps
    per_step_model_facing = model_facing_total / n_steps
    per_step_savings = round(
        (1 - per_step_model_facing / per_step_internal) * 100, 1
    ) if per_step_internal else 0.0

    trigger_counts: dict[str, int] = {}
    obs_counts: dict[str, int] = {}
    for s in steps:
        trigger_counts[s["trigger"]] = trigger_counts.get(s["trigger"], 0) + 1
        obs_counts[s["obs_type"]] = obs_counts.get(s["obs_type"], 0) + 1

    summary = {
        "framing": (
            "Deterministic scripted spreadsheet observation-cost benchmark. "
            "Drives a local HTML mock through a fixed trajectory. No agent, "
            "no trajectory variance. Reproducible on any machine."
        ),
        "trajectory_spec": {
            "mock_url": mock_url,
            "headers": HEADERS,
            "n_rows": len(ROWS),
            "n_cells": len(HEADERS) + sum(len(r) for r in ROWS),
            "total_steps": len(steps),
        },
        "summary": {
            # Cost split (preferred names — read these in new code)
            "dv_internal_total_tokens": dv_internal_total,
            "model_facing_total_tokens": model_facing_total,
            "total_savings_pct": total_savings,
            "per_step_avg_dv_internal": round(per_step_internal, 1),
            "per_step_avg_model_facing": round(per_step_model_facing, 1),
            "per_step_savings_pct": per_step_savings,
            "trigger_counts": trigger_counts,
            "obs_type_counts": obs_counts,
            # Back-compat aliases (deprecated; remove in v1.1.0)
            # ff = "what FF would have cost" = "the full frame DV consumed"
            #    = dv_internal. dv = "what DV billed" = model_facing.
            "ff_total_tokens": dv_internal_total,
            "dv_total_tokens": model_facing_total,
            "per_step_avg_ff": round(per_step_internal, 1),
            "per_step_avg_dv": round(per_step_model_facing, 1),
        },
        "steps": steps,
    }
    return summary


# =============================================================================
# Entry
# =============================================================================

def main() -> None:
    out_file = Path(__file__).parent / "spreadsheet_observation_cost.json"
    summary = asyncio.run(run_benchmark(Path(__file__).parent))
    out_file.write_text(json.dumps(summary, indent=2))

    s = summary["summary"]
    n = summary["trajectory_spec"]["total_steps"]
    print("=" * 72)
    print("Spreadsheet observation-cost benchmark")
    print("=" * 72)
    print(f"Steps:                       {n}")
    print(f"DV internal total tokens:    {s['dv_internal_total_tokens']:>7}")
    print("  (what DV consumed; cost of all full screenshots, every step)")
    print(f"Model-facing total tokens:   {s['model_facing_total_tokens']:>7}")
    print("  (what DV shipped to the model; smaller on delta steps)")
    print(f"Total savings:               {s['total_savings_pct']:>6.1f}%")
    print(f"Per-step DV internal avg:    {s['per_step_avg_dv_internal']:>7.1f}")
    print(f"Per-step model-facing avg:   {s['per_step_avg_model_facing']:>7.1f}")
    print(f"Per-step savings:            {s['per_step_savings_pct']:>6.1f}%")
    print(f"Triggers:                    {s['trigger_counts']}")
    print(f"Obs types:                   {s['obs_type_counts']}")
    print(f"\nArtifact:  {out_file}")


if __name__ == "__main__":
    main()
