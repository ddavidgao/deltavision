"""
After run_head_to_head.py finishes, plug the real numbers into:
  1. /Users/davidgao/Projects/dv-video-scratch/code_review_demo/remotion/src/DeltaVision.tsx
     (the H2H constants block)
  2. /Users/davidgao/Projects/deltavision/launch/twitter_thread.md
     (replaces [BRACKETS] with real values)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO / "benchmarks" / "headtohead" / "head_to_head_results.json"
TSX_PATH = Path("/Users/davidgao/Projects/dv-video-scratch/code_review_demo/remotion/src/DeltaVision.tsx")
THREAD_PATH = REPO / "launch" / "twitter_thread.md"


def summarize(results: list[dict]) -> dict:
    """Produce head-to-head summary: DV vs FF with means and ranges."""
    dv = [r for r in results if r.get("config") == "DV" and "error" not in r]
    ff = [r for r in results if r.get("config") == "FF" and "error" not in r]

    def stats(rs, key):
        vals = [r[key] for r in rs]
        if not vals:
            return None
        return {
            "mean": round(sum(vals) / len(vals), 1),
            "min": min(vals),
            "max": max(vals),
            "values": vals,
        }

    def success(rs):
        return f"{sum(1 for r in rs if r['done'])}/{len(rs)}"

    dv_tok = stats(dv, "total_input_tokens")
    ff_tok = stats(ff, "total_input_tokens")

    token_savings_pct = None
    if dv_tok and ff_tok and ff_tok["mean"]:
        token_savings_pct = round(
            (ff_tok["mean"] - dv_tok["mean"]) / ff_tok["mean"] * 100, 1
        )

    return {
        "n_trials": len(dv),
        "dv": {
            "n": len(dv),
            "steps": stats(dv, "steps"),
            "tokens_in": dv_tok,
            "tokens_out": stats(dv, "total_output_tokens"),
            "wall_time_sec": stats(dv, "wall_time_sec"),
            "success": success(dv),
        },
        "ff": {
            "n": len(ff),
            "steps": stats(ff, "steps"),
            "tokens_in": ff_tok,
            "tokens_out": stats(ff, "total_output_tokens"),
            "wall_time_sec": stats(ff, "wall_time_sec"),
            "success": success(ff),
        },
        "token_savings_pct": token_savings_pct,
    }


def fmt_steps(s):
    if not s:
        return "—"
    return f"{s['mean']} (range {s['min']}–{s['max']})"


def fmt_tokens(s):
    if not s:
        return "—"
    return f"{int(s['mean']):,} (range {int(s['min']):,}–{int(s['max']):,})"


def update_tsx(summary: dict):
    """Rewrite the H2H constants block in DeltaVision.tsx."""
    dv = summary["dv"]
    ff = summary["ff"]

    new_block = f"""const H2H = {{
  // Auto-generated from benchmarks/headtohead/head_to_head_results.json
  // Do not hand-edit. Re-run the benchmark + this updater script to refresh.
  dv_steps: "{dv['steps']['mean'] if dv['steps'] else '—'}",
  dv_tokens: "{int(dv['tokens_in']['mean']):,}" if True else "—",
  dv_success: "{dv['success']}",
  ff_steps: "{ff['steps']['mean'] if ff['steps'] else '—'}",
  ff_tokens: "{int(ff['tokens_in']['mean']):,}" if True else "—",
  ff_success: "{ff['success']}",
  token_savings_pct: "{summary['token_savings_pct']}",
  n_trials: {summary['n_trials']},
  task: "TodoMVC · add 3 items + complete 1",
}};"""
    # Clean the accidental "if True else" — I need valid JS, not Python expr
    new_block_js = f"""const H2H = {{
  // Auto-generated from benchmarks/headtohead/head_to_head_results.json
  // Do not hand-edit. Re-run the benchmark + this updater script to refresh.
  dv_steps: "{dv['steps']['mean']}",
  dv_tokens: "{int(dv['tokens_in']['mean']):,}",
  dv_success: "{dv['success']}",
  ff_steps: "{ff['steps']['mean']}",
  ff_tokens: "{int(ff['tokens_in']['mean']):,}",
  ff_success: "{ff['success']}",
  token_savings_pct: "{summary['token_savings_pct']}",
  n_trials: {summary['n_trials']},
  task: "TodoMVC · add 3 items + complete 1",
}};"""

    src = TSX_PATH.read_text()
    # Match the H2H = { ... }; block (from const H2H = { to };)
    pattern = r"const H2H = \{[^}]*\};"
    new = re.sub(pattern, new_block_js, src, flags=re.DOTALL)
    if new == src:
        print("WARNING: H2H block not found in TSX, nothing updated")
        return
    TSX_PATH.write_text(new)
    print(f"Updated {TSX_PATH.name}:")
    print(new_block_js)


def update_thread(summary: dict):
    """Replace [PLACEHOLDERS] in the thread draft with real numbers."""
    src = THREAD_PATH.read_text()
    replacements = {
        "[N]": str(summary["n_trials"]),
        "[FF_STEPS]": fmt_steps(summary["ff"]["steps"]),
        "[FF_TOK]": fmt_tokens(summary["ff"]["tokens_in"]),
        "[FF_SUCC]": summary["ff"]["success"],
        "[DV_STEPS]": fmt_steps(summary["dv"]["steps"]),
        "[DV_TOK]": fmt_tokens(summary["dv"]["tokens_in"]),
        "[DV_SUCC]": summary["dv"]["success"],
        "[SAVINGS]": str(summary["token_savings_pct"]),
    }
    for k, v in replacements.items():
        src = src.replace(k, v)
    THREAD_PATH.write_text(src)
    print(f"Updated {THREAD_PATH.name} with real numbers.")


def main():
    if not RESULTS_PATH.exists():
        raise SystemExit(f"missing {RESULTS_PATH}")
    data = json.loads(RESULTS_PATH.read_text())
    summary = summarize(data["results"])
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print()

    update_tsx(summary)
    update_thread(summary)

    # Dump summary JSON for the README to reference
    out = REPO / "benchmarks" / "headtohead" / "head_to_head_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
