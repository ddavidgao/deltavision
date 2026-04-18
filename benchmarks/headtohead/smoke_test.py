"""One-trial smoke test before kicking off the full head-to-head."""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.headtohead.run_head_to_head import run_trial, TASKS


async def main():
    task = TASKS[0]
    print(f"SMOKE TEST: {task['name']}, DV-enabled, 1 trial")
    r = await run_trial(task, dv_enabled=True, trial=0)
    if "error" in r:
        print(f"FAILED: {r['error']}")
        sys.exit(1)
    # Don't dump full usage log to stdout
    r_display = {k: v for k, v in r.items() if k not in ("usage_log", "transition_log")}
    r_display["n_usage_log_entries"] = len(r.get("usage_log", []))
    r_display["n_transitions"] = len(r.get("transition_log", []))
    print(json.dumps(r_display, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
