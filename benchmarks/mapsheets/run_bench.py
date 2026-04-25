"""
Maps→Sheets benchmark orchestrator.

Each trial:
  1. Clear the sheet (manual or AppleScript)
  2. Run task as a Claude Code subagent via Agent tool (has MCP access)
     -- OR -- run scoring/measurement on an existing run
  3. Measure FF vs DV token savings from screenshots
  4. Score by reading the final screenshot
  5. Write results/run_N/metrics.json

Usage (from repo root):
    # Measure/score an existing run (screenshots already in results/run_N/screenshots/)
    .venv/bin/python3 benchmarks/mapsheets/run_bench.py --score-only --trial 2
    .venv/bin/python3 benchmarks/mapsheets/run_bench.py --clear-only

    # Trials are run interactively via the Agent tool in Claude Code (has MCP access)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

SHEET_URL = "https://docs.google.com/spreadsheets/d/1_WQ2e9-7CS6NbFZ-3WsdP5Mrfptb1e4_CWIlVOszKzc/edit"
TASK_PROMPT_FILE = Path(__file__).parent / "TASK_PROMPT.txt"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Sheet clear — via AppleScript
# ---------------------------------------------------------------------------

CLEAR_APPLESCRIPT = f"""
tell application "Google Chrome"
    activate
    tell front window to make new tab with properties {{URL:"{SHEET_URL}"}}
    delay 5
end tell

tell application "System Events"
    tell process "Google Chrome"
        key code 53
        delay 0.5
        keystroke "a" using {{command down}}
        delay 0.5
        key code 51
        delay 1
        keystroke "w" using {{command down}}
        delay 0.3
    end tell
end tell
"""


def clear_sheet_applescript() -> None:
    print("[clear] Using AppleScript to clear sheet in Chrome...")
    result = subprocess.run(
        ["osascript", "-e", CLEAR_APPLESCRIPT],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[clear] AppleScript error: {result.stderr}")
    else:
        print("[clear] Done.")


# ---------------------------------------------------------------------------
# Token measurement
# ---------------------------------------------------------------------------

def measure_tokens(shot_dir: Path) -> dict:
    script = Path(__file__).parent / "measure_tokens.py"
    r = subprocess.run(
        [sys.executable, str(script), str(shot_dir)],
        capture_output=True, text=True, cwd=str(REPO)
    )
    if r.returncode != 0:
        return {"error": r.stderr.strip()}
    out = r.stdout
    try:
        return json.loads(out[out.index("{"):out.rindex("}")+1])
    except (ValueError, json.JSONDecodeError):
        return {"error": "parse failed", "raw": out[:500]}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_from_screenshot(run_dir: Path) -> dict:
    shot_dir = run_dir / "screenshots"
    import re as _re
    shots = sorted(
        shot_dir.glob("step_*.png"),
        key=lambda p: int(_re.search(r"(\d+)", p.stem).group(1))
    )
    if not shots:
        return {"error": "no screenshots found"}
    final = shots[-1]
    return {
        "final_screenshot": str(final),
        "n_screenshots": len(shots),
        "note": "review final screenshot manually",
    }


def finalize_trial(trial: int, model: str, elapsed: float) -> dict:
    """Collect screenshots from .playwright-mcp/, compute metrics, write metrics.json."""
    run_dir = RESULTS_DIR / f"run_{trial:02d}"
    shot_dir = run_dir / "screenshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    shot_dir.mkdir(parents=True, exist_ok=True)

    pw_dir = REPO / ".playwright-mcp"
    pw_shots = sorted(pw_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
    import shutil
    for i, src in enumerate(pw_shots):
        shutil.copy2(src, shot_dir / f"step_{i:02d}.png")

    n_shots = len(list(shot_dir.glob("step_*.png")))
    tokens = measure_tokens(shot_dir) if n_shots >= 2 else {"error": "need ≥2 screenshots"}
    score = score_from_screenshot(run_dir)

    metrics = {
        "trial": trial, "model": model, "elapsed_s": elapsed,
        "n_screenshots": n_shots, "tokens": tokens, "score": score,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _print_summary(metrics)
    return metrics


def _print_summary(m: dict) -> None:
    tk = m.get("tokens", {})
    print("\n=== SUMMARY ===")
    print(f"Model:      {m.get('model', '—')}")
    print(f"Time:       {m.get('elapsed_s', '—')}s   Screenshots: {m.get('n_screenshots', '—')}")
    if "savings_pct" in tk:
        print(f"FF tokens:  {tk['ff_total_tokens']:,}   DV tokens: {tk['dv_total_tokens']:,}   Savings: {tk['savings_pct']}%")
    elif "error" in tk:
        print(f"Tokens:     {tk['error']}")
    sc = m.get("score", {})
    if "final_screenshot" in sc:
        print(f"Final shot: {sc['final_screenshot']}")


# ---------------------------------------------------------------------------
# CLI (scoring/clear only — trials run via Agent tool in Claude Code)
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--clear-only", action="store_true")
    args = p.parse_args()

    if args.clear_only:
        clear_sheet_applescript()
    elif args.score_only:
        run_dir = RESULTS_DIR / f"run_{args.trial:02d}"
        tokens = measure_tokens(run_dir / "screenshots")
        score = score_from_screenshot(run_dir)
        print(json.dumps({"tokens": tokens, "score": score}, indent=2))
    else:
        print("Trials must be run interactively via the Agent tool in Claude Code.")
        print("This script handles --score-only and --clear-only only.")


if __name__ == "__main__":
    main()
