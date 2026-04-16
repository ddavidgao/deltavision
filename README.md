# DeltaVision

**Observation middleware for GUI agents.** A CV pipeline sits between the browser and the model, sending only what changed on screen instead of a full screenshot every step.

The model still reasons — it just reasons about less.

## Why This Matters

Standard computer use agents send a full 1280x900 screenshot (~1600 tokens) on every step, whether 1 pixel changed or the entire page swapped. DeltaVision puts a 4-layer CV classifier in front of the model that decides: did the page change, or just a region? Send accordingly.

**Fair comparison: same model (Claude Sonnet 4.6), same task, same browser, 3 runs each:**

| | DeltaVision | Standard Agent |
|---|---|---|
| Steps | **4 every run** | 6 every run |
| Total tokens | **~12,800** | ~36,900 |
| History cost/step | **~130 tokens (text)** | ~1,650 tokens (screenshot) |
| Token savings | **65%** | -- |

Both complete the task. DeltaVision is faster because it tells the model "your action had no effect" via text metadata -- the standard approach relies on the model visually comparing screenshots in conversation history, which wastes steps.

**4 steps vs 6, 12.8k tokens vs 36.9k -- same model, same result.**

## How It Works

```
Browser Action
    |
    v
+--------------------------------------+
|  DeltaVision CV Pipeline (no LLM)    |
|                                       |
|  Layer 1: URL change (free)           |
|  Layer 2: Pixel diff ratio (numpy)    |
|  Layer 3: Perceptual hash (PIL)       |
|  Layer 4: Anchor template match (cv2) |
|  + Scroll bypass gate                 |
|  + Animation guard                    |
+------------------+-------------------+
                   |
          +--------+--------+
          |                 |
      DELTA path       NEW_PAGE path
    crops + diff     full screenshot
    (~400 tokens)    (~1600 tokens)
          |                 |
          v                 v
+--------------------------------------+
|  Any Model Backend                    |
|  Claude / GPT-4o / Ollama / Local     |
+--------------------------------------+
```

**Classifier accuracy:** 17/17 scenarios across 7 diverse websites with default config. No site-specific tuning needed.

## Quick Start

```bash
git clone https://github.com/ddavidgao/deltavision.git
cd deltavision
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# Run tests (no API keys needed)
pytest tests/ -v  # 56 tests

# Reaction time benchmark (pure CV, no model)
python benchmarks/reaction/run_reaction.py --rounds 5 --headless

# Classifier generalization test (17 scenarios, 7 sites)
python benchmarks/generalization/test_classifier_diverse.py
```

### With a model backend

```bash
# Ollama (free, local, no API key)
ollama pull qwen2.5vl:7b
python main.py --task "Search Wikipedia for 'computer vision'" \
    --url https://en.wikipedia.org --backend ollama --model qwen2.5vl:7b --headless

# Claude API
export ANTHROPIC_API_KEY=sk-...
python main.py --task "..." --url ... --backend claude

# OpenAI
export OPENAI_API_KEY=sk-...
python main.py --task "..." --url ... --backend openai

# Ablation: same task without delta gating (for comparison)
python main.py --task "..." --url ... --backend ollama --model qwen2.5vl:7b --force-full-frame
```

### Safety modes

```bash
python main.py --task "..." --url ... --safety strict       # block credentials, shorteners
python main.py --task "..." --url ... --safety educational   # allowlist edu sites only
python main.py --task "..." --url ... --safety permissive    # log warnings only
```

## Architecture

```
deltavision/
  vision/           # CV pipeline: diff engine, pHash, 4-layer classifier, capture
  agent/            # Agent loop, state machine, typed actions
  observation/      # Builds typed observations (FullFrame or Delta)
  model/            # Pluggable backends: claude, openai, ollama, local, scripted
  safety.py         # Model-agnostic action validation
  config.py         # All thresholds in one place, site-specific presets
  results/          # SQLite result store (query with db.summary() or raw SQL)
  benchmarks/
    reaction/       # CV-only reaction time benchmark
    generalization/ # Classifier accuracy across diverse sites + visual frame capture
    ablation/       # DeltaVision vs full-frame controlled comparison
    sites/          # Benchmark site registry (7 sites, 3 difficulty tiers)
  tests/            # 56 tests: unit, integration, live Playwright, real screenshots
  paper/            # Paper outline with figure/table mapping to data
```

## Results

All results stored in `results/deltavision.db` (SQLite). Query:
```bash
python -c "from results.store import ResultStore; ResultStore().summary()"
```

### Ablation: Delta Gating vs Full-Frame

| Metric | DeltaVision | Full-Frame Only | Savings |
|--------|-------------|-----------------|---------|
| Steps (simple task) | 3 | 50 (failed) | - |
| Steps (multi-step) | 5 | 12 | 2.4x fewer |
| Est. image tokens (simple) | 4,000 | 81,600 | 95% |
| Est. image tokens (multi) | 4,800 | 20,800 | 77% |
| Task completion (simple) | Yes | No | - |

### Classifier Generalization

| Site | Type | Scenarios | Accuracy |
|------|------|-----------|----------|
| Wikipedia | Traditional nav | 4 | 100% |
| HumanBenchmark | Dynamic content | 3 | 100% |
| Hacker News | Minimal HTML | 3 | 100% |
| TodoMVC | SPA (React) | 1 | 100% |
| GitHub | SPA (Turbo) | 1 | 100% |
| Dynamic SPA | JS content injection | 2 | 100% |
| Scroll test | Viewport shift | 2 | 100% |

### Reaction Time (CV pipeline only, no model)

| | DeltaVision | Human Median | Claude CU |
|---|---|---|---|
| Best | 74ms | 273ms | 13,491ms |
| Average | 100ms | 273ms | 13,491ms |

Note: This measures CV pipeline speed (screenshot + color detect + click). The comparison to Claude CU is unfair — Claude runs full model inference per step. The reaction benchmark demonstrates that simple visual tasks don't need a model at all.

## Demo Video

Pre-recorded comparison videos in `benchmarks/demo/`:

| File | Content |
|------|---------|
| `deltavision_full_demo.mp4` | Complete demo: title + simple task + multi-step task |
| `deltavision_demo.mp4` | Simple Wikipedia search, side-by-side |
| `deltavision_demo_multistep.mp4` | 3-article navigation, DV completes in 4 steps vs baseline fails at 30 |

Record your own:
```bash
# Needs Ollama running with a VLM
ollama serve
python benchmarks/demo/record_comparison.py --task wikipedia
python benchmarks/demo/record_comparison.py --task wikipedia_multi
```

Videos are recorded by Playwright at 60fps. ffmpeg combines them side-by-side with labels.

## Key Design Decisions

1. **The model never decides transition type.** The CV classifier is deterministic, sub-millisecond, and testable.
2. **Speed comes from sending less, not skipping the model.** The model still reasons; it gets cropped regions instead of full screenshots.
3. **Safety is framework-level.** Critical for uncensored local models that won't refuse dangerous actions.
4. **Backend-agnostic.** Same observation format for Claude, GPT-4o, Qwen, or UI-TARS.
5. **Scroll-aware.** Scrolling shifts the viewport but doesn't change page state. The classifier knows this.
6. **Animation-resistant.** Subtle animations (spinners, fades) don't trigger false page transitions.

## License

MIT
