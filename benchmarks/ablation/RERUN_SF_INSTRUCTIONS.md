# SF Maps→Sheets rerun — when you're back

Shipped the residual-first classifier refactor (2026-04-24). Offline replay
on the current SF screenshots confirms steps 5, 7, 11 flip from `new_page`
to `delta (transform_delta)`. Live rerun will show the real savings number.

## One-command kickoff (when Chrome is open)

```bash
cd ~/Projects/deltavision
.venv/bin/python3 benchmarks/mapsheets/run_bench.py --clear-only
```

Then in Claude Code, spawn a subagent with the task prompt:

```
Read /Users/davidgao/Projects/deltavision/benchmarks/mapsheets/TASK_PROMPT.txt
and execute the task. Use mcp__dv-playwright__* tools exclusively.
Take screenshots after each major state change (aim for 20-25 total).
```

After the subagent finishes:

```bash
# Copy screenshots to a fresh run dir
.venv/bin/python3 benchmarks/mapsheets/run_bench.py --trial 3
# (the finalize_trial function copies .playwright-mcp/*.png to run_03/screenshots/)

# Check savings
.venv/bin/python3 benchmarks/mapsheets/run_bench.py --score-only --trial 3
```

## Expected outcome

- **Old (0.5 inlier gate):** 25.6% savings, steps 5-8 all NEW_PAGE (map pans)
- **New (residual-first):** expecting 35-45% savings, map pans now DELTA

The latest DV proxy log lands in `dv_runs/dv_proxy_run_{timestamp}.jsonl`.

## What changed in the classifier

- `vision/transform.py` — `detect_similarity` default threshold 0.5 → 0.0
- `vision/classifier.py` — always try warp, keep only if it reduces residual
- `config.py` — added `IDENTICAL_DIFF_EPSILON: 0.003` for zero-token fast exit

All 239 tests pass. Offline replay harness at
`benchmarks/ablation/replay_classifier_offline.py`.
