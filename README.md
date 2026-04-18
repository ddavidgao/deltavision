# DeltaVision

**Observation middleware for GUI agents.** A CV pipeline sits between the browser and the model, sending only what changed on screen instead of a full screenshot every step.

The model still reasons — it just reasons about less.

## Why This Matters

Standard computer use agents send a full 1280x900 screenshot (~1600 tokens) on every step, whether 1 pixel changed or the entire page swapped. DeltaVision puts a 4-layer CV classifier in front of the model that decides: did the page change, or just a region? Send accordingly.

**Measured token savings on a real 9-step TodoMVC agent run** (Playwright +
Anthropic tool_result format, same task both ways — only the observation
pipeline differs):

| | Baseline (full-frame) | With DeltaVision |
|---|---|---|
| Image tokens | 13,824 | **6,133** (−55.6%) |
| Wire bytes | 1,033 KB | **604 KB** (−41.5%) |

Reproducible: `python examples/observer_integration_proof.py`. Raw metrics:
`examples/observer_proof_results.json`. The savings grow with task length —
on longer SPA workflows where the page structure is sticky across steps,
DeltaVision tends to stay on the DELTA path for 80%+ of steps, each of which
costs ~3-7× less than a full frame.

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

**Classifier accuracy:** 17/17 scenarios across 8 diverse websites with default config. No site-specific tuning needed. Source: `benchmarks/generalization/results.json`.

## Quick Start

```bash
git clone https://github.com/ddavidgao/deltavision.git
cd deltavision
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install as an editable package (recommended for development)
pip install -e ".[claude]"      # or [openai], [ollama], [all], [dev]
playwright install chromium

# Or plain requirements.txt (legacy)
pip install -r requirements.txt

# Run tests (no API keys needed) — 217 total, 210 pass offline
pytest tests/ -q --ignore=tests/test_e2e_live.py --ignore=tests/test_live_capture.py

# Reaction time benchmark (pure CV, no model)
python benchmarks/reaction/run_reaction.py --rounds 5 --headless

# Classifier generalization test (17 scenarios, 8 sites)
python benchmarks/generalization/test_classifier_diverse.py
```

### With a model backend

```bash
# Ollama (free, local, no API key)
ollama pull qwen2.5vl:7b
python main.py --task "Search Wikipedia for 'computer vision'" \
    --url https://en.wikipedia.org --backend ollama --model qwen2.5vl:7b --headless

# Claude API (Sonnet 4.6 default; Opus 4.7 and newer models work via --model)
export ANTHROPIC_API_KEY=sk-...
python main.py --task "..." --url ... --backend claude
python main.py --task "..." --url ... --backend claude --model claude-opus-4-7-20260417

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
  tests/            # 217 tests: unit, integration, live Playwright, real screenshots
                    # See TESTS.md for a visual coverage map
  paper/            # Paper outline with figure/table mapping to data
```

## Testing

See [TESTS.md](TESTS.md) for a per-module table of what every test verifies.

| Suite | Tests | Covers |
|---|---|---|
| CV pipeline | 34 | diff, pHash, 4-layer classifier cascade, real McGraw-Hill frames |
| Model response parsing | 33 | JSON extraction, VLM output quirks (fences, preamble, nested confidence) |
| Safety layer | 37 | URL safety, credential detection, action limits, preset configs |
| Config validation | 45 | every threshold range, every field type, bbox coherence |
| Results store | 19 | SQLite save/query/best, schema, persistence across reopen |
| Integration | 15 | observation builder, action parser, agent state, simulated pipeline |
| Observer API | 34 | lifecycle + 5 format adapters (Anthropic/OpenAI/Browser Use/Skyvern/Stagehand) |
| Live (CI-skipped) | 7 | browser E2E, live capture |
| **Total** | **217** | |

```bash
pytest tests/ -q                    # full offline suite (183 pass)
pytest tests/test_safety.py -v      # single module
pytest tests/ --cov=. --cov-report=term-missing  # coverage
```

## Using as a library

```python
from playwright.async_api import async_playwright
from config import DeltaVisionConfig
from agent.loop import run_agent
from model.claude import ClaudeModel
from safety import STRICT

async def go():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        state = await run_agent(
            task="Find the capital of France on Wikipedia",
            start_url="https://en.wikipedia.org",
            model=ClaudeModel(api_key="sk-...", model="claude-sonnet-4-6"),
            browser_page=page,
            config=DeltaVisionConfig(MAX_STEPS=20),
            safety=STRICT,
        )
        print(f"Done in {state.step} steps, delta ratio: {state.delta_ratio:.1%}")
        await browser.close()
```

The `run_agent` function is the single entry point. All tunable behavior is in `DeltaVisionConfig`. Results are in `state` (dataclass with `observations`, `responses`, `transition_log`, `step`, `done`, etc.).

## Results

All results stored in `results/deltavision.db` (SQLite). Query:
```bash
python -c "from results.store import ResultStore; ResultStore().summary()"
```

### V2 (OS-level): matched-trajectory ablation — 68.2% savings

The sibling repo `deltavision-os` (mss + pyautogui for OS-level desktop agents)
published its own independent ablation:

- **Same 10-step trajectory run twice** on a real Mac desktop, natural
  Qwen2.5-VL behavior. No cherry-picking — exact same actions each time.
- Forced full-frame: 17,600 image tokens
- Delta-gated: 5,600 image tokens
- **68.2% savings, zero difference in task outcome.**

Plus a threshold sweep (3 trajectories × 3 values of `NEW_PAGE_DIFF_THRESHOLD`)
empirically confirmed that the pHash layer dominates: diff-threshold in
[0.30, 0.75] produced identical classifications. Paper-grade finding.

Raw data: `deltavision-os/benchmarks/ablation_result.json` and
`ablation_sweep_result.json`. Plus a smoke-test run on the **ScreenSpot-v2**
community benchmark (Qwen2.5-VL-7B, 80% desktop accuracy on n=15) proves the
V2 stack works end-to-end with a real VLM.

Repo: https://github.com/ddavidgao/deltavision-os

---

### Browser side: savings depend heavily on the workload

Two benchmarks, both same Anthropic `tool_result` format, both real Playwright,
same `DeltaVisionObserver` wrapping — nothing else changes between baseline
and DV. Savings vary by task type; here's the honest shape:

**SPA + mixed browser tasks (5 tasks, 40 steps)** — reproduce: `python examples/multi_site_benchmark.py`

| Task | Steps | Token savings |
|---|---|---|
| TodoMVC: add 3 + filter | 9 | **55.6%** |
| TodoMVC: add 3, check 2, clear | 10 | **63.6%** |
| Wikipedia: search + navigate + scroll | 9 | 47.6% |
| Hacker News: browse + scroll + open thread | 7 | 26.2% |
| example.com idle | 5 | 72.6% |
| **Aggregate** | **40** | **52.8%** |

**Scroll-dominated media exploration (10 WebVoyager sites, 70 steps)** — reproduce: `python examples/webvoyager_subset.py`

| Site | Steps | Token savings |
|---|---|---|
| huggingface | 7 | 33.2% |
| wolfram | 7 | 23.2% |
| cambridge | 7 | 18.7% |
| github | 7 | 18.2% |
| apple | 7 | 12.7% |
| allrecipes | 7 | 12.6% |
| bbc_news | 7 | 10.3% |
| coursera | 7 | 7.4% |
| arxiv | 7 | 6.2% |
| espn | 7 | 4.7% |
| **Aggregate** | **70** | **14.7%** |

#### What the two benchmarks tell you

- **DV excels on SPA / mixed-interaction tasks** (55-65% savings). These are
  the real CU-agent workloads: click buttons, type in inputs, navigate
  with URL changes. Small region changes compress well into crops.
- **DV's savings shrink on pure-scroll exploration** (5-33% savings). The
  `scroll_bypass` gate correctly classifies scrolled frames as DELTAs, but
  the resulting delta crops are near-full-viewport because each scroll
  exposes a large band of new content. Token savings stay positive but the
  wire-byte savings can go slightly negative (PNG-of-thumbnail + PNG-of-
  large-crop can exceed a single PNG-of-full-frame).
- **Best case** (idle, static, tiny local change): 70%+ savings.
- **Worst case** (ESPN, full-page scrolling media): 4.7% token savings.

This is a fundamental property of the technique, not a bug: DeltaVision is
an **observation-level optimization for sticky-context workflows.** On
sites where every scroll reveals mostly-new pixels, there's less redundant
observation to strip.

**Practical takeaway for CU agent builders:** if your workflow is mostly
typing / clicking / form interactions, expect 40-70% token savings. If it's
mostly scrolling through long feeds, expect 5-20%. Real agents do both, so
a typical run lands between those endpoints (20-40% is a reasonable
expectation for a mixed agent workload).

### Multi-step ablation on Wikipedia (Qwen2.5-VL-7B)

Wikipedia search-and-navigate task. Both agents use the same model, same
prompt, same browser.

| | DeltaVision | Full-Frame (baseline) |
|---|---|---|
| Outcome | Task completed at step 3 | Hit 50-step limit, did not complete |
| Image tokens used | ~4,000 | ~81,600 |
| Delta ratio | 67% | 0% |

Caveat: the full-frame baseline didn't complete the task. Token counts are
cumulative over the steps each agent actually executed, not directly
comparable as "tokens to complete the same work." The meaningful claim is
that DeltaVision completed a task that the full-frame path failed on, at a
fraction of the per-step observation cost. Data: DB Runs 11/12.

### Classifier Generalization

| Site | Type | Scenarios | Accuracy |
|------|------|-----------|----------|
| Wikipedia | Traditional nav | 4 | 100% |
| HumanBenchmark | Dynamic content | 3 | 100% |
| Hacker News | Minimal HTML | 3 | 100% |
| Dynamic SPA | JS content injection | 2 | 100% |
| Scroll test | Viewport shift | 2 | 100% |
| TodoMVC | SPA (React) | 1 | 100% |
| GitHub (public browse) | SPA (Turbo) | 1 | 100% |
| example.com | Static minimal | 1 | 100% |
| **Total** | **8 sites** | **17** | **100%** |

Source: `benchmarks/generalization/results.json`. All scenarios run with default config, no per-site tuning.

### Reaction Time (CV pipeline only, no model)

| | DeltaVision (5 clean rounds) | Human Median | Claude CU (n=1) |
|---|---|---|---|
| Best | 74ms | 273ms | 13,491ms |
| Average | 100ms | 273ms | 13,491ms (single measurement) |

Data sources: DeltaVision values from DB Run 10 (5 clean rounds of 10, fixed state machine, Windows RTX 5080). Claude CU baseline from `results/humanbenchmark_reaction_20260414_220141.json` (backend label: `claude_standard_cu`, 1 round — model version not recorded in the JSON).

Note: This measures CV pipeline speed (screenshot + color detect + click). The comparison to Claude CU is unfair — Claude runs full model inference per step — and the Claude baseline is n=1, so the "avg" cell repeats the single measurement. The reaction benchmark demonstrates that simple visual tasks don't need a model at all.

## Demo Video

Pre-recorded comparison video in `benchmarks/demo/`:

| File | Content |
|------|---------|
| `deltavision_final_demo.mp4` | Complete side-by-side comparison: DeltaVision vs full-frame on Wikipedia search (multi-step) |

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
4. **Backend-agnostic.** Same observation format for Claude (Sonnet 4.6, Opus 4.7), GPT-4o, Qwen, or UI-TARS. Add a backend in `model/` with `BaseModel`'s interface and DV's classifier output drops in unchanged.
5. **Scroll-aware.** Scrolling shifts the viewport but doesn't change page state. The classifier knows this.
6. **Animation-resistant.** Subtle animations (spinners, fades) don't trigger false page transitions.

## Troubleshooting

**"ANTHROPIC_API_KEY not set"** — copy `.env.example` to `.env` and fill in your key, or export it: `export ANTHROPIC_API_KEY=sk-...`. The CLI also loads `.env` from the project root automatically.

**Ollama connection refused** — start the server first: `ollama serve` in another terminal. Check the model is pulled: `ollama list`. Default host is `http://localhost:11434`.

**`ModuleNotFoundError: No module named 'numpy'`** — the project venv is separate from system Python. Use `.venv/bin/python3` explicitly or activate the venv first. macOS's system Python is externally managed.

**Playwright browser not found** — run `playwright install chromium` after the first `pip install`.

**Classifier misbehaves on a custom site** — dump `meta.json` from `benchmarks/generalization/frames/` to see what the CV pipeline measured. Tune `PHASH_DISTANCE_THRESHOLD` (default 20) or `NEW_PAGE_DIFF_THRESHOLD` (default 0.75) in `config.py`. All thresholds are validated at construction — bad values raise `ConfigError` immediately.

**`ConfigError` at startup** — you set a threshold out of range. Every field has documented bounds in `config.py::DeltaVisionConfig.__post_init__`. The error names the field and the valid range.

## V1 vs V2

This is **V1 (browser-focused)**. If you need OS-level or OSWorld-VM observation, see [`deltavision-os`](https://github.com/ddavidgao/deltavision-os) (active development). V1 is frozen at the paper-artifact version for reproducibility.

## License

MIT
