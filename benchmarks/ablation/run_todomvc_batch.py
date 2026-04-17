"""
Batch runner for TodoMVC DV vs FF ablation — runs N trials per mode,
captures per-step DV observations, logs all metrics to SQLite.

Usage:
  python benchmarks/ablation/run_todomvc_batch.py --runs 3 --start-run 2
  python benchmarks/ablation/run_todomvc_batch.py --task extended --runs 1
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from results.store import ResultStore


# ── Constants ─────────────────────────────────────────────────────────
FULL_FRAME_TOKENS = 1600
DELTA_TOKENS_AVG  = 400

BASE_DIR = Path(__file__).parent / "runs"

TASKS = {
    "basic": {
        # 4 actions: add 3 todos + mark 1 complete
        "url": "https://todomvc.com/examples/react/dist/",
        "desc": "Add buy milk, feed cat, pay rent; check feed cat",
        "max_steps": 20,
    },
    "extended": {
        # 10 actions: 5 adds + 2 checks + 3 filter navigations (mixed transitions)
        "url": "https://todomvc.com/examples/react/dist/",
        "desc": "Add 5 todos, check 2, navigate All/Active/Completed filter views",
        "max_steps": 20,
    },
}


# ── DV observe helper ─────────────────────────────────────────────────
def dv_observe(mode: str, t1: str, step: int, out_dir: str,
               t0: str = None, url_before: str = "", url_after: str = "",
               last_action: str = "", no_change_count: int = 0) -> dict:
    cmd = [
        sys.executable, "benchmarks/ablation/dv_observe.py",
        "--mode", mode,
        "--t1", t1,
        "--step", str(step),
        "--output-dir", out_dir,
        "--url-before", url_before,
        "--url-after", url_after,
        "--last-action", last_action,
        "--no-change-count", str(no_change_count),
    ]
    if t0:
        cmd += ["--t0", t0]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parents[2]))
    if result.returncode != 0:
        print(f"  [dv_observe ERROR] {result.stderr[:200]}")
        return {}
    return json.loads(result.stdout)


# ── Basic task executor ───────────────────────────────────────────────
async def run_basic(page, mode: str, run_idx: int) -> dict:
    """4-step TodoMVC: add 3 todos + mark feed cat complete."""
    run_dir = BASE_DIR / f"todomvc_{mode}/run{run_idx}"
    url = TASKS["basic"]["url"]

    def step_dir(s): return str(run_dir / f"step_{s}")
    def step_path(s, name): return str(run_dir / f"step_{s}" / name)

    (run_dir / "step_0").mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        (run_dir / f"step_{i}").mkdir(parents=True, exist_ok=True)

    await page.goto(url)
    await page.wait_for_load_state("networkidle")

    # Step 0: baseline full_frame
    t0_path = step_path(0, "t0.png")
    await page.screenshot(path=t0_path, type="png")
    obs0 = dv_observe("full_frame", t0_path, 0, step_dir(0), url_after=url)

    steps_log = [{"step": 0, "obs": obs0}]
    prev_t1 = t0_path
    prev_url = url
    no_change = 0
    full_frames = 1
    deltas = 0

    actions = [
        ("type_enter", "buy milk"),
        ("type_enter", "feed cat"),
        ("type_enter", "pay rent"),
        ("check_todo", "feed cat"),
    ]

    for step_i, (action_type, payload) in enumerate(actions, start=1):
        t_start = time.perf_counter()

        if action_type == "type_enter":
            inp = page.get_by_test_id("text-input")
            await inp.fill(payload)
            await page.keyboard.press("Enter")
            last_action = f"type('{payload}')+key(Enter)"
        elif action_type == "check_todo":
            toggle = page.get_by_role("listitem").filter(has_text=payload).get_by_test_id("todo-item-toggle")
            await toggle.click()
            last_action = f"click_toggle('{payload}')"

        await page.wait_for_timeout(200)
        curr_url = page.url

        t1_path = step_path(step_i, "t1.png")
        await page.screenshot(path=t1_path, type="png")

        if mode == "full_frame":
            obs = dv_observe("full_frame", t1_path, step_i, step_dir(step_i),
                             url_after=curr_url, last_action=last_action)
            full_frames += 1
        else:
            obs = dv_observe("delta", t1_path, step_i, step_dir(step_i),
                             t0=prev_t1, url_before=prev_url, url_after=curr_url,
                             last_action=last_action, no_change_count=no_change)
            if obs.get("obs_type") == "full_frame":
                full_frames += 1
                prev_t1 = t1_path  # reset anchor on new_page
            else:
                deltas += 1

        prev_url = curr_url
        elapsed = time.perf_counter() - t_start
        steps_log.append({"step": step_i, "obs": obs, "elapsed_ms": round(elapsed*1000)})
        transition = obs.get("transition", "full_frame")
        diff = obs.get("diff_ratio", 0)
        crops = len(obs.get("crops", []))
        print(f"    step {step_i}: {transition:<10} diff={diff:.3f}  crops={crops}  {elapsed*1000:.0f}ms")

    estimated_tokens = (full_frames * FULL_FRAME_TOKENS) + (deltas * DELTA_TOKENS_AVG)
    total_steps = len(actions)
    delta_ratio = deltas / (full_frames + deltas) if (full_frames + deltas) > 0 else 0

    return {
        "run_idx": run_idx, "mode": mode, "task": "todomvc_basic",
        "steps": total_steps, "done": True,
        "full_frames_sent": full_frames, "deltas_sent": deltas,
        "new_page_count": 0, "delta_ratio": round(delta_ratio, 3),
        "estimated_image_tokens": estimated_tokens,
        "steps_log": steps_log,
    }


# ── Extended task executor ─────────────────────────────────────────────
async def run_extended(page, mode: str, run_idx: int) -> dict:
    """10-step TodoMVC: 5 adds + 2 checks + 3 filter nav (mixed transitions)."""
    run_dir = BASE_DIR / f"todomvc_ext_{mode}/run{run_idx}"
    url = TASKS["extended"]["url"]

    def step_dir(s): return str(run_dir / f"step_{s}")
    def step_path(s, name): return str(run_dir / f"step_{s}" / name)

    for i in range(12):
        (run_dir / f"step_{i}").mkdir(parents=True, exist_ok=True)

    await page.goto(url)
    await page.wait_for_load_state("networkidle")

    t0_path = step_path(0, "t0.png")
    await page.screenshot(path=t0_path, type="png")
    obs0 = dv_observe("full_frame", t0_path, 0, step_dir(0), url_after=url)

    steps_log = [{"step": 0, "obs": obs0}]
    prev_t1 = t0_path
    prev_url = url
    no_change = 0
    full_frames = 1
    deltas = 0
    new_page_count = 0

    todos = ["buy milk", "feed cat", "pay rent", "call doctor", "read paper"]
    check_todos = ["feed cat", "read paper"]

    actions = (
        [("type_enter", t) for t in todos] +
        [("check_todo", t) for t in check_todos] +
        [("nav_filter", "#/active"),
         ("nav_filter", "#/completed"),
         ("nav_filter", "#/")]
    )

    for step_i, (action_type, payload) in enumerate(actions, start=1):
        t_start = time.perf_counter()

        if action_type == "type_enter":
            inp = page.get_by_test_id("text-input")
            await inp.fill(payload)
            await page.keyboard.press("Enter")
            last_action = f"type('{payload}')+Enter"
        elif action_type == "check_todo":
            toggle = page.get_by_role("listitem").filter(has_text=payload).get_by_test_id("todo-item-toggle")
            await toggle.click()
            last_action = f"toggle('{payload}')"
        elif action_type == "nav_filter":
            await page.click(f'a[href="{payload}"]')
            last_action = f"click_filter('{payload}')"

        await page.wait_for_timeout(200)
        curr_url = page.url

        t1_path = step_path(step_i, "t1.png")
        await page.screenshot(path=t1_path, type="png")

        if mode == "full_frame":
            obs = dv_observe("full_frame", t1_path, step_i, step_dir(step_i),
                             url_after=curr_url, last_action=last_action)
            full_frames += 1
        else:
            obs = dv_observe("delta", t1_path, step_i, step_dir(step_i),
                             t0=prev_t1, url_before=prev_url, url_after=curr_url,
                             last_action=last_action, no_change_count=no_change)
            if obs.get("obs_type") == "full_frame":
                full_frames += 1
                new_page_count += 1
                prev_t1 = t1_path
            else:
                deltas += 1

        prev_url = curr_url
        elapsed = time.perf_counter() - t_start
        steps_log.append({"step": step_i, "obs": obs, "elapsed_ms": round(elapsed*1000)})
        transition = obs.get("transition", "full_frame")
        diff = obs.get("diff_ratio", 0)
        crops = len(obs.get("crops", []))
        print(f"    step {step_i}: {transition:<10} diff={diff:.3f}  crops={crops}  {elapsed*1000:.0f}ms")

    estimated_tokens = (full_frames * FULL_FRAME_TOKENS) + (deltas * DELTA_TOKENS_AVG)
    total_steps = len(actions)
    delta_ratio = deltas / (full_frames + deltas) if (full_frames + deltas) > 0 else 0

    return {
        "run_idx": run_idx, "mode": mode, "task": "todomvc_extended",
        "steps": total_steps, "done": True,
        "full_frames_sent": full_frames, "deltas_sent": deltas,
        "new_page_count": new_page_count, "delta_ratio": round(delta_ratio, 3),
        "estimated_image_tokens": estimated_tokens,
        "steps_log": steps_log,
    }


# ── Wikipedia multi-hop ────────────────────────────────────────────────
async def run_wiki_multihop(page, mode: str, run_idx: int) -> dict:
    """3 chained URL navigations: Main -> CV article -> ML article -> Deep learning."""
    run_dir = BASE_DIR / f"wiki_multihop_{mode}/run{run_idx}"
    for i in range(6):
        (run_dir / f"step_{i}").mkdir(parents=True, exist_ok=True)

    def step_dir(s): return str(run_dir / f"step_{s}")
    def step_path(s, name): return str(run_dir / f"step_{s}" / name)

    start_url = "https://en.wikipedia.org/wiki/Main_Page"
    await page.goto(start_url)
    await page.wait_for_load_state("domcontentloaded")

    t0_path = step_path(0, "t0.png")
    await page.screenshot(path=t0_path, type="png")
    obs0 = dv_observe("full_frame", t0_path, 0, step_dir(0), url_after=start_url)

    steps_log = [{"step": 0, "obs": obs0}]
    prev_t1 = t0_path
    prev_url = start_url
    full_frames = 1
    deltas = 0
    new_page_count = 0

    # Action sequence: type search + Enter, then 2 in-article link clicks
    actions = [
        ("search",    "computer vision"),
        ("click_link", "Machine learning"),
        ("click_link", "Deep learning"),
    ]

    for step_i, (action_type, payload) in enumerate(actions, start=1):
        t_start = time.perf_counter()

        if action_type == "search":
            box = page.get_by_role("searchbox")
            await box.fill(payload)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("domcontentloaded")
            last_action = f"search('{payload}')"
        elif action_type == "click_link":
            # Navigate directly to target Wikipedia article — simulates clicking
            # an in-article link (tests NEW_PAGE detection, same as click nav)
            href_slug = payload.replace(" ", "_")
            await page.goto(f"https://en.wikipedia.org/wiki/{href_slug}")
            await page.wait_for_load_state("domcontentloaded")
            last_action = f"click_link('{payload}')"

        await page.wait_for_timeout(500)
        curr_url = page.url

        t1_path = step_path(step_i, "t1.png")
        await page.screenshot(path=t1_path, type="png")

        if mode == "full_frame":
            obs = dv_observe("full_frame", t1_path, step_i, step_dir(step_i),
                             url_after=curr_url, last_action=last_action)
            full_frames += 1
        else:
            obs = dv_observe("delta", t1_path, step_i, step_dir(step_i),
                             t0=prev_t1, url_before=prev_url, url_after=curr_url,
                             last_action=last_action)
            if obs.get("obs_type") == "full_frame":
                full_frames += 1
                new_page_count += 1
                prev_t1 = t1_path
            else:
                deltas += 1

        prev_url = curr_url
        elapsed = time.perf_counter() - t_start
        steps_log.append({"step": step_i, "obs": obs, "elapsed_ms": round(elapsed*1000)})
        transition = obs.get("transition", "full_frame")
        diff = obs.get("diff_ratio", 0)
        print(f"    step {step_i}: {transition:<10} diff={diff:.3f}  url={curr_url[-40:]}  {elapsed*1000:.0f}ms")

    done = "deep_learning" in page.url.lower()
    estimated_tokens = (full_frames * FULL_FRAME_TOKENS) + (deltas * DELTA_TOKENS_AVG)
    total_steps = len(actions)
    delta_ratio = deltas / (full_frames + deltas) if (full_frames + deltas) > 0 else 0

    return {
        "run_idx": run_idx, "mode": mode, "task": "wikipedia_multihop",
        "steps": total_steps, "done": done, "final_url": page.url,
        "full_frames_sent": full_frames, "deltas_sent": deltas,
        "new_page_count": new_page_count, "delta_ratio": round(delta_ratio, 3),
        "estimated_image_tokens": estimated_tokens,
        "steps_log": steps_log,
    }


# ── Main ───────────────────────────────────────────────────────────────
async def main(task: str, modes: list, start_run: int, num_runs: int):
    db = ResultStore()
    all_results = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})

        for run_idx in range(start_run, start_run + num_runs):
            for mode in modes:
                page = await context.new_page()
                print(f"\n{'='*60}")
                print(f"  Run {run_idx}  |  task={task}  |  mode={mode}")
                print(f"{'='*60}")

                t_start = time.perf_counter()
                try:
                    if task == "basic":
                        result = await run_basic(page, mode, run_idx)
                    elif task == "extended":
                        result = await run_extended(page, mode, run_idx)
                    elif task == "wiki":
                        result = await run_wiki_multihop(page, mode, run_idx)
                    else:
                        raise ValueError(f"Unknown task: {task}")
                    result["wall_time_s"] = round(time.perf_counter() - t_start, 1)
                    result["error"] = None
                except Exception as e:
                    result = {"run_idx": run_idx, "mode": mode, "task": task,
                              "done": False, "error": str(e), "estimated_image_tokens": 0}
                    print(f"  ERROR: {e}")

                all_results.append(result)

                backend = f"claude-sonnet-4-6_{mode}"
                db.save(
                    benchmark=f"sonnet_{task}",
                    backend=backend,
                    metrics={k: v for k, v in result.items() if k != "steps_log"},
                    config={"task": task, "mode": mode, "run_idx": run_idx},
                    notes=f"Batch run {run_idx}: {task} / {mode}",
                )
                print(f"  -> tokens={result.get('estimated_image_tokens',0):,}  "
                      f"done={result.get('done')}  delta_ratio={result.get('delta_ratio',0):.0%}  "
                      f"time={result.get('wall_time_s','?')}s")
                await page.close()

        await browser.close()
    db.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {task}")
    print(f"{'='*70}")
    print(f"  {'Mode':<20} {'Runs':>5} {'Avg Tokens':>12} {'Done%':>8} {'Avg Delta%':>12}")
    for mode in modes:
        runs = [r for r in all_results if r.get("mode") == mode and "error" not in r or r.get("error") is None]
        if not runs:
            continue
        avg_tok = sum(r.get("estimated_image_tokens", 0) for r in runs) / len(runs)
        done_pct = sum(1 for r in runs if r.get("done")) / len(runs) * 100
        avg_delta = sum(r.get("delta_ratio", 0) for r in runs) / len(runs) * 100
        print(f"  {mode:<20} {len(runs):>5} {avg_tok:>12,.0f} {done_pct:>7.0f}% {avg_delta:>11.0f}%")

    out = Path(__file__).parent / f"batch_{task}_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["basic", "extended", "wiki"], default="basic")
    p.add_argument("--modes", nargs="+", default=["deltavision", "full_frame"])
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--start-run", type=int, default=2)
    args = p.parse_args()
    asyncio.run(main(args.task, args.modes, args.start_run, args.runs))
