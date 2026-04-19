# Agent Head-to-Head (reserved)

Scaffolding for a future free-running Claude CU agent comparison between
DeltaVision-wrapped observations and raw full-frame observations.

Components:
- `ingest.py` — saves two summary.json files (DV arm, FF arm) into the SQLite
  ResultStore + creates a comparison row.
- `../../examples/agent_head_to_head.py` — the actual harness. Uses
  `claude-sonnet-4-5-20250929` via Anthropic computer-use tool type, drives
  Playwright, optionally wraps observations with `DeltaVisionObserver`.

**Status:** not run. Paid API spend required (est. $3–$10 per full
head-to-head depending on rate-limit backoffs). Requires explicit
green-light from David before any new execution.

Previous attempts:
- 2026-04-19: DV arm ran 40 steps, got 2.5 apartments documented, hit max
  steps; artifacts were accidentally deleted before FF arm could run.
  ~$2.24 spent, zero video produced. Lesson captured in
  `~/.claude/memory/learnings.md` under "HARD RULE: never spend David's
  API budget without explicit in-chat green light."

When to revisit:
- David explicitly funds a real head-to-head
- Harness is first modified to: prune conversation history (cap per-call
  input tokens to ~8K) to reduce rate-limit thrash, and write partial
  summary after every step rather than only at end.
- Before executing: estimate cost in chat, get explicit go-ahead per run,
  copy video + screenshots + summary off Windows BEFORE running any `rmdir`
  for the next attempt.
