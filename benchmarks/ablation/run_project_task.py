"""
Project-task workflow runner — 10-step task with full JSON capture per step.

Task: "Manage a project checklist:
  - Add 5 work tasks (write report, review PR, send invoice, update docs, schedule meeting)
  - Complete 2 tasks (write report, review PR) using checkboxes
  - Navigate filters: Active → Completed → All

This gives a mix of:
  - 5 DELTA steps (new todo row appears, ~5% of pixels change)
  - 2 DELTA steps (checkbox toggle + strikethrough, ~1-2% of pixels)
  - 3 NEW_PAGE steps (URL hash changes: #/active, #/completed, #/all → DV sends full frame)

All per-step JSON is saved to disk so bbox data is available for the video builder.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_DIR = Path(__file__).parent / "runs" / "project_task"
URL = "https://todomvc.com/examples/react/dist/"

FULL_FRAME_TOKENS = 1600
DELTA_TOKENS_EST  = {
    # (thumbnail ~60 tok) + (1 crop ~180 tok) = ~240
    # (thumbnail ~60 tok) + (2 crops ~360 tok) = ~420
    # using conservative estimate per step
    1: 300,   # 1 crop
}


def dv_observe(mode: str, t1: str, step: int, out_dir: str,
               t0: str = None, url_before: str = "", url_after: str = "",
               last_action: str = "") -> dict:
    cmd = [
        sys.executable, "benchmarks/ablation/dv_observe.py",
        "--mode", mode,
        "--t1", t1,
        "--step", str(step),
        "--output-dir", out_dir,
        "--url-before", url_before,
        "--url-after", url_after,
        "--last-action", last_action,
    ]
    if t0:
        cmd += ["--t0", t0]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(Path(__file__).parents[2]))
    if r.returncode != 0:
        print(f"  [dv_observe ERROR] {r.stderr[:300]}")
        return {}
    return json.loads(r.stdout)


async def run_project_task(mode: str = "delta"):
    run_dir = BASE_DIR / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-step JSON storage
    steps_data = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1200, "height": 700})

        await page.goto(URL)
        await page.wait_for_load_state("networkidle")

        # ── Step 0: baseline ──────────────────────────────────────────
        s0_dir = run_dir / "step_0"
        s0_dir.mkdir(exist_ok=True)
        t0_path = str(s0_dir / "t0.png")
        await page.screenshot(path=t0_path, type="png")
        obs0 = dv_observe("full_frame", t0_path, 0, str(s0_dir), url_after=URL)
        obs0["action"] = "initial_load"
        obs0["action_label"] = "Initial page load"
        obs0["t1_path"] = t0_path
        obs0["t0_path"] = t0_path
        # save step JSON
        (s0_dir / "obs.json").write_text(json.dumps(obs0, indent=2))
        steps_data.append(obs0)

        # Running token tallies
        ff_tokens_cumul = FULL_FRAME_TOKENS  # step 0
        dv_tokens_cumul = FULL_FRAME_TOKENS  # step 0

        prev_t1 = t0_path
        prev_url = URL

        # ── Scripted actions ──────────────────────────────────────────
        ACTIONS = [
            # (action_type, payload, label)
            ("type_enter", "write report",       "type('write report') + Enter"),
            ("type_enter", "review PR",           "type('review PR') + Enter"),
            ("type_enter", "send invoice",        "type('send invoice') + Enter"),
            ("type_enter", "update docs",         "type('update docs') + Enter"),
            ("type_enter", "schedule meeting",    "type('schedule meeting') + Enter"),
            ("check_todo", "write report",        "click checkbox: 'write report'"),
            ("check_todo", "review PR",           "click checkbox: 'review PR'"),
            ("click_filter", "Active",            "click filter: Active"),
            ("click_filter", "Completed",         "click filter: Completed"),
            ("click_filter", "All",               "click filter: All"),
        ]

        for step_i, (action_type, payload, label) in enumerate(ACTIONS, start=1):
            t_start = time.perf_counter()

            if action_type == "type_enter":
                inp = page.get_by_test_id("text-input")
                await inp.fill(payload)
                await page.keyboard.press("Enter")
            elif action_type == "check_todo":
                toggle = (
                    page.get_by_role("listitem")
                    .filter(has_text=payload)
                    .get_by_test_id("todo-item-toggle")
                )
                await toggle.click()
            elif action_type == "click_filter":
                await page.get_by_role("link", name=payload).click()

            await page.wait_for_timeout(300)
            curr_url = page.url

            step_dir = run_dir / f"step_{step_i}"
            step_dir.mkdir(exist_ok=True)
            t1_path = str(step_dir / "t1.png")
            await page.screenshot(path=t1_path, type="png")

            t_screenshot = time.perf_counter()

            # Run DV observation
            obs = dv_observe(
                mode if mode != "full_frame" else "full_frame",
                t1=t1_path,
                step=step_i,
                out_dir=str(step_dir),
                t0=prev_t1,
                url_before=prev_url,
                url_after=curr_url,
                last_action=action_type,
            )

            t_obs = time.perf_counter()
            elapsed_ms = int((t_obs - t_start) * 1000)

            # Token accounting
            obs_type = obs.get("obs_type", "full_frame")
            num_crops = len(obs.get("crops", []))
            if obs_type == "delta" and num_crops > 0:
                # thumbnail (~60 tok) + N crops (~180 tok each, capped at 400px)
                step_dv_tok = 60 + num_crops * 180
            else:
                step_dv_tok = FULL_FRAME_TOKENS  # new_page → full frame

            step_ff_tok = FULL_FRAME_TOKENS  # ff always sends full frame
            ff_tokens_cumul += step_ff_tok
            dv_tokens_cumul += step_dv_tok

            obs["step"]           = step_i
            obs["action"]         = action_type
            obs["payload"]        = payload
            obs["action_label"]   = label
            obs["t0_path"]        = prev_t1
            obs["t1_path"]        = t1_path
            obs["elapsed_ms"]     = elapsed_ms
            obs["step_ff_tokens"] = step_ff_tok
            obs["step_dv_tokens"] = step_dv_tok
            obs["ff_tokens_cumul"]= ff_tokens_cumul
            obs["dv_tokens_cumul"]= dv_tokens_cumul

            transition = obs.get("transition", "?")
            print(f"  step {step_i:2d}: {transition:10s}  diff={obs.get('diff_ratio',0):.3f}  "
                  f"crops={num_crops}  {elapsed_ms}ms  "
                  f"dv_tok={step_dv_tok}  ff_tok={step_ff_tok}")

            (step_dir / "obs.json").write_text(json.dumps(obs, indent=2))
            steps_data.append(obs)

            prev_t1 = t1_path
            prev_url = curr_url

        await browser.close()

    # Save full run summary
    summary = {
        "mode": mode,
        "task": "project_task_10step",
        "steps": len(steps_data),
        "ff_tokens_cumul": ff_tokens_cumul,
        "dv_tokens_cumul": dv_tokens_cumul,
        "savings_pct": round((1 - dv_tokens_cumul / ff_tokens_cumul) * 100, 1) if ff_tokens_cumul else 0,
        "step_data": steps_data,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  FF total: {ff_tokens_cumul:,} tokens")
    print(f"  DV total: {dv_tokens_cumul:,} tokens  ({summary['savings_pct']}% less)")
    return summary


async def main():
    print("=" * 60)
    print("  Project Task — DeltaVision mode")
    print("=" * 60)
    dv_summary = await run_project_task(mode="delta")

    print()
    print("=" * 60)
    print("  Project Task — Full-Frame baseline")
    print("=" * 60)
    ff_summary = await run_project_task(mode="full_frame")

    print("\n\nFINAL COMPARISON")
    print(f"  DeltaVision:  {dv_summary['dv_tokens_cumul']:>7,} tokens cumulative")
    print(f"  Full-Frame:   {ff_summary['ff_tokens_cumul']:>7,} tokens cumulative")
    savings = 1 - dv_summary['dv_tokens_cumul'] / ff_summary['ff_tokens_cumul']
    print(f"  Savings:      {savings*100:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
