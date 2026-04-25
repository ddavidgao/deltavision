# DeltaVision

Delta-first computer use agent framework. The model's primary observation is the **delta**, not the full frame.

## Core success criterion (the one that matters)

**At worst the same number of steps · always fewer tokens.**

DV is meaningful only if it doesn't make the agent dumber. If DV halves token cost
but doubles step count, total work goes up — that's a regression dressed up as a
win. The honest test is two-axis:

1. **Steps to completion** must be ≤ FF on the same task. If DV consistently
   takes more steps, the observation pipeline is hurting agent decision-making
   (block-count bloat, missed context, scattered crop attention) and that
   debt eats the per-frame savings.
2. **Total tokens** must be < FF. Per-frame savings are necessary but not
   sufficient — they only translate to real cost reduction when step count
   doesn't inflate.

Once we have multi-trial trajectories, run a **null-hypothesis test**: H₀ = "DV
takes the same or fewer steps than FF on the same task." Reject if step-count
distributions cross with FF strictly below. This frames the demo numbers as
statistical claims (p < 0.05, n ≥ 5) rather than n=1 anecdotes.

Pending fix in flight (2026-04-25): the proxy's diff engine emits up to
`MAX_REGIONS=6` separate crop bboxes per step. On steps with one big real change
plus several scattered tiny ones, this fragments into 6 expensive crops totaling
> 1365 tokens, tripping the token-cap fallback and forcing a full frame. A
**greedy rectangle-merge optimizer** (merge any pair where `cost(A∪B) < cost(A)
+ cost(B)`) lifted simulated savings from 37.2% → 56.2% on the run_20 SF trace
without touching the classifier — file: `benchmarks/ablation/simulate_greedy_merge.py`.
Ship the merge into `vision/diff.py` next, then re-benchmark.

## TLDR — What Matters

**What it is:** Observation middleware for GUI agents. Sits between the browser and any model (Claude, GPT-4o, Qwen, Hermes via Ollama). Sends the model only what changed on screen instead of a full screenshot every step.

**Results so far:**

| Benchmark | DeltaVision | Full-Frame Baseline | Notes |
|-----------|-------------|---------------------|---------|
| TodoMVC 9-step (measured integration) | 6,133 img tokens | 13,824 img tokens | **55.6% savings** — same task, same format, same agent |
| Wikipedia search | Completed in 3 steps | Hit 50-step limit, did not complete | Task-outcome comparison (not a direct % since FF did not complete) |
| Wikipedia multi-step (3-article) | 5 steps, ~4.8k tokens | 12 steps, ~20.8k tokens | ~77% fewer tokens, both completed |
| Reaction time (best, CV-only) | 74ms | Claude CU (n=1): 13,491ms | Unfair comparison (CV vs full agent), but shows pure-CV ceiling |
| Classifier generalization | 17/17 (100%) | 8 sites, default config | No site-specific tuning |

| Ablation Detail | DeltaVision | Full-Frame Only |
|-----------------|-------------|-----------------|
| Task completed (simple) | Yes | No (hit 50 step limit) |
| Task completed (multi-step) | Yes (5 steps) | Yes (12 steps) |
| Est. image tokens (simple) | 4,000 | 81,600 |
| Delta ratio | 80% | 0% |

| McGraw-Hill Metric | Value | Notes |
|--------------------|-------|-------|
| Anchor match score | **1.000** | Nav bar matches perfectly across all states |
| Primary NEW_PAGE trigger | pHash (dist>18) | NOT diff_ratio -- white pages stay low |
| diff_ratio (Q->Q) | 0.107 | 10.7% pixel change on question swap |
| diff_ratio (Q->Reading) | 0.022 | Only 2.2% despite full content swap |

**Novelty (researched):** No prior published work does observation-level gating with a CV pipeline before the model. Closest: Agent-E (DOM-text, not visual), GUIPruner/GUI-KV (internal model optimization). DeltaVision is the first to decide what the model sees using zero-LLM classification.

**Key insight:** Speed comes from sending LESS to the model, not from skipping the model. The model still reasons — it just reasons about less.

## Architecture

```
Browser → [DeltaVision CV Pipeline] → Observation (delta or full frame) → Any Model → Action
```

- `vision/` — Pure CV. No LLM. Diff engine, 4-layer classifier cascade, perceptual hashing.
- `agent/` — Loop, state, typed actions. Safety layer blocks dangerous actions regardless of backend.
- `observation/` — Builds typed observations (FullFrame or Delta) for model consumption.
- `model/` — Pluggable: Claude (`claude.py`), OpenAI (`openai.py`), Ollama (`ollama.py`), local transformers (`local.py`).
- `safety.py` — Model-agnostic. Credential detection, URL validation, shortener blocking. Critical for uncensored models.
- `config.py` — All tunable constants. Site-specific presets (e.g. `MCGRAWHILL_CONFIG`).
- `results/store.py` — SQLite result store. Query with `db.summary()` or raw SQL.
- `benchmarks/sites/registry.py` — 7 benchmark sites across 3 difficulty tiers.

## Running

```bash
# Claude backend
python main.py --task "Complete the quiz" --url https://example.com --backend claude

# OpenAI
python main.py --task "..." --url ... --backend openai

# Any model via Ollama (Hermes, Qwen, LLaVA, etc.)
python main.py --task "..." --url ... --backend ollama --model qwen2.5-vl:7b

# Text-only model — gets structured descriptions instead of images
python main.py --task "..." --url ... --backend ollama --model hermes3:8b

# With safety (permissive | strict | educational | none)
python main.py --task "..." --url ... --safety educational

# Reaction time benchmark (no model needed — pure CV)
python benchmarks/reaction/run_reaction.py --rounds 5
```

## Tests

```bash
pytest tests/ -v  # 56 tests, no API keys needed
```

Test coverage: diff engine, pHash, classifier cascade (all 4 layers), observation builder, action parsing, agent state, pipeline simulation, live Playwright (button click, navigation, SPA content swap), real McGraw-Hill screenshots.

## Results

```bash
python -c "from results.store import ResultStore; ResultStore().summary()"
```

Stored in `results/deltavision.db` (SQLite). Each run records benchmark, backend, metrics, config, and transition log.

## Visual Artifacts & Evaluation Data

**Captured frames live in `benchmarks/generalization/frames/`:**
Each scenario folder contains: `t0.png` (before), `t1.png` (after), `diff.png` (binary diff mask overlay), `crop_N_before/after.png` (extracted delta regions), `meta.json` (all classifier metrics).

Key scenarios with frames:
- `dynamic_spa_full_content_replacement/` — JS replaces entire page (same URL). pHash=28, diff=0.51. Classified: NEW_PAGE via Layer 3.
- `todomvc_spa_add_items/` — 3 todo items added. diff=0.039, pHash=10. Classified: DELTA. Only the list region lights up.
- `wikipedia_scroll_long_article/` — Big scroll. diff=0.33, pHash=28 — but scroll_bypass gate fires. Classified: DELTA.
- `wikipedia_nav_article_to_article/` — Standard URL nav. Classified: NEW_PAGE via Layer 1. Layers 2-4 never run.
- `hackernews_idle_no_change/` — Zero change baseline. diff=0.000, pHash=0. Classified: DELTA.

**To regenerate frames:** `python benchmarks/generalization/capture_frames.py`
**To run full generalization test:** `python benchmarks/generalization/test_classifier_diverse.py`

**Classifier generalization (17/17 sites, default config):**

| Site | Scenarios | Accuracy | Layers Tested |
|------|-----------|----------|---------------|
| Wikipedia | 4 | 100% | L1 (url), L1, idle, L1 |
| HumanBenchmark | 3 | 100% | L1, L1, idle |
| Hacker News | 3 | 100% | L1, idle, L1 |
| TodoMVC (SPA) | 1 | 100% | Layers 2-3 (diff+pHash) |
| Dynamic SPA | 2 | 100% | L3 (pHash=22), idle |
| Scroll | 2 | 100% | scroll_bypass gate |
| example.com | 1 | 100% | idle baseline |

**SQLite results:** `python -c "from results.store import ResultStore; ResultStore().summary()"`

Stored in `results/deltavision.db`. Each run records benchmark, backend, metrics, config snapshot, and full transition log. Query with `db.query("SELECT ...")`.

## Evaluation Playbook

```bash
# 1. CV-only: reaction time (no model, no API key)
python benchmarks/reaction/run_reaction.py --rounds 10

# 2. Classifier generalization (captures + classifies across 7 sites)
python benchmarks/generalization/test_classifier_diverse.py

# 3. Visual frame capture (saves t0, t1, diff, crops as PNGs)
python benchmarks/generalization/capture_frames.py

# 4. Full agent loop with Ollama VLM
ollama serve  # if not running
python main.py --task "Search Wikipedia for neural networks" \
    --url https://en.wikipedia.org --backend ollama --model qwen2.5-vl:7b

# 5. Query results
python -c "from results.store import ResultStore; ResultStore().summary()"
python -c "from results.store import ResultStore; print(ResultStore().query('SELECT * FROM runs'))"
```

**What to measure for the paper:**
- Token reduction: full frame ~1600 tokens vs delta crops ~200-400 tokens
- Steps to complete task: DeltaVision-wrapped vs raw screenshots
- Classifier accuracy across site categories (easy/medium/hard)
- Speed: reaction time, per-step latency, end-to-end completion time
- Model-size scaling: same task with 7B vs 14B vs frontier

## Key Invariants

1. The **model never decides** transition type. That's the CV classifier's job.
2. `t0` is always the last anchor frame (reset on NEW_PAGE events).
3. The no_change_streak mechanism forces full-frame refresh after N stuck steps.
4. All thresholds live in `config.py` — no magic numbers in logic code.
5. **Safety layer runs regardless of backend** — the framework enforces safety, not the model.

## Paper

Draft outline at `paper/outline.md`. Every figure and table mapped to a specific SQLite run or frame capture. Structured as a 7-section paper with ablation as the central result.

## Private/Public Sync

Private repo (source of truth). Run `./sync-public.sh "message"` to mirror to public (excludes `.claude/`, `.env`, credentials).

- Private: github.com/ddavidgao/deltavision-private
- Public: github.com/ddavidgao/deltavision

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
