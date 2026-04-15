# DeltaVision

Delta-first computer use agent framework. The model's primary observation is the **delta**, not the full frame.

## TLDR — What Matters

**What it is:** Observation middleware for GUI agents. Sits between the browser and any model (Claude, GPT-4o, Qwen, Hermes via Ollama). Sends the model only what changed on screen instead of a full screenshot every step.

**Results so far:**

| Benchmark | DeltaVision | Standard Claude CU | Human | Speedup |
|-----------|-------------|---------------------|-------|---------|
| Reaction time (best) | **412ms** | 13,491ms | 273ms | **30x** |
| Reaction time (avg) | **755ms** | 13,491ms | 273ms | **16x** |
| Detection→click | **6ms** | ~5,000ms | ~50ms | **800x** |

| McGraw-Hill Metric | Value | Notes |
|--------------------|-------|-------|
| Anchor match score | **1.000** | Nav bar matches perfectly across all states |
| Primary NEW_PAGE trigger | pHash (dist>18) | NOT diff_ratio — white pages stay low |
| diff_ratio (Q→Q) | 0.107 | 10.7% pixel change on question swap |
| diff_ratio (Q→Reading) | 0.022 | Only 2.2% despite full content swap |

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
pytest tests/ -v  # 46 tests, ~7s, no API keys needed
```

Test coverage: diff engine, pHash, classifier cascade (all 4 layers), observation builder, action parsing, agent state, pipeline simulation, live Playwright (button click, navigation, SPA content swap), real McGraw-Hill screenshots.

## Results

```bash
python -c "from results.store import ResultStore; ResultStore().summary()"
```

Stored in `results/deltavision.db` (SQLite). Each run records benchmark, backend, metrics, config, and transition log.

## Key Invariants

1. The **model never decides** transition type. That's the CV classifier's job.
2. `t0` is always the last anchor frame (reset on NEW_PAGE events).
3. The no_change_streak mechanism forces full-frame refresh after N stuck steps.
4. All thresholds live in `config.py` — no magic numbers in logic code.
5. **Safety layer runs regardless of backend** — the framework enforces safety, not the model.

## Private/Public Sync

Private repo (source of truth). Run `./sync-public.sh "message"` to mirror to public (excludes `.claude/`, `.env`, credentials).

- Private: github.com/ddavidgao/deltavision-private
- Public: github.com/ddavidgao/deltavision
