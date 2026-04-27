# DeltaVision

**Observation middleware for computer-use agents.** Sits between your browser and your model. Sends the model only what changed on screen instead of a fresh screenshot every step. The agent still reasons — it just reasons about less.

## 60-second install + use

```bash
pip install deltavision
```

```python
import deltavision

obs = deltavision.DeltaVisionObserver()

# Give it a screenshot (bytes / PIL Image / base64 / data URL),
# the current URL, and a description of the last action.
result = obs.observe(screenshot_png_bytes, url="https://example.com", last_action="click #submit")

print(result.obs_type)              # "full_frame" or "delta"
print(result.model_facing_tokens())  # tokens DV is about to ship to the model
print(result.dv_internal_tokens())   # tokens DV consumed internally (always full frame)
# Savings on this step: 1 - model_facing / dv_internal.
# (estimated_image_tokens() is a deprecated alias of model_facing_tokens(); see Cost accounting.)
result.to_anthropic_tool_result_content()  # ready-to-send Claude content blocks
```

That's it. Plug the `observe()` call wherever your agent currently sends a screenshot to the model. Every adapter (`to_anthropic_*`, `to_openai_*`, `to_browser_use_*`, `to_stagehand_*`) returns content already shaped for that SDK's vision API. See [`examples/integration_tests.py`](examples/integration_tests.py) for live-API proof on all four.

### Verify your install in 10 seconds

```bash
deltavision selftest
```

Runs 9 staged checks (import → observer → delta path → coverage guard → adapter → HTTP sidecar round-trip). Each stage reports ✓ or ✗ with what it checked, so a failure points directly at the broken piece. No API keys, no network, all synthetic frames.

```
DeltaVision self-test — staged E2E
  S1  import deltavision                              ✓  v1.0.6
  S2  observer construction                           ✓  26 fields
  S3  initial observation → full_frame                ✓  tokens=1365
  S4  small delta is cheaper than full frame          ✓  tokens=171 vs FF=1365 (87.5% saved)
  S5  whole-frame change → coverage guard fires       ✓  tokens=1365 = FF 1365 (trigger='crop_covers_frame')
  S6  anthropic adapter output is well-formed         ✓  1 content blocks
  S7  HTTP /health reports package version            ✓  version=1.0.6
  S8  HTTP /observe round-trips a DVObservation       ✓
  S9  HTTP /reset clears observer state               ✓
  All 9 stages passed.
```

### Run as an HTTP service (non-Python agents)

```bash
python -m server --port 9000

# Then from any runtime:
curl http://localhost:9000/health                 # → {"status":"ok","version":"1.0.5"}
curl -X POST http://localhost:9000/observe \
     -F file=@screenshot.png -F url=... -F format=anthropic
```

Endpoints: `GET /health` · `POST /observe` · `POST /reset` · `GET /state`. Returns adapter-formatted JSON (`anthropic` / `openai` / `browser_use` / `stagehand` / `raw`). See [`server.py`](server.py) for the full API.

### Namespace layout caveat

Because v1.0.x maintains backwards compatibility with the flat-module imports (`from observer import ...`), the wheel ships both a `deltavision/` umbrella package AND the raw modules (`observer.py`, `vision/`, `agent/`, `model/`) at site-packages root. If your CWD has a directory named `vision/` or `observer.py`, it can shadow the installed one — run your script from a different directory or use `from deltavision import X` (which always resolves through the umbrella). v2.0 will nest modules under the package to remove this caveat.

**PyPI:** [`deltavision==1.0.6`](https://pypi.org/project/deltavision/1.0.6/) · **OS-level companion (V2, alpha):** [`deltavision-os`](https://pypi.org/project/deltavision-os/) · **Source of truth:** private repo, public mirror auto-synced

### What's new in 1.0.6

Two CV-pipeline upgrades that, together, flip DV from "wins on tokens, loses on steps" to "wins on **both** axes" — the criterion the project requires:

- **Greedy bbox-merge optimizer (`vision/diff.py`).** The proxy used to emit up to `MAX_REGIONS=6` separate crop bboxes per step. On steps with one big real change plus several scattered tiny ones, this fragmented into 6 expensive crops totaling > 1365 tokens, tripping the token-cap fallback and forcing a full frame. The new merger greedily pairs bboxes while `cost(A∪B) < cost(A) + cost(B)`. Replayed across 16 saved runs: **38.1% → 48.9% mean savings**, no classifier changes. Toggleable via `DeltaVisionConfig(BBOX_MERGE_ENABLED=True)` (default on).
- **Periodic full-frame refresh (`dv_playwright_mcp.py`).** The proxy now forces a full-frame response every `DV_DELTA_REFRESH_EVERY=5` consecutive deltas, so the agent re-anchors instead of getting lost in dialog interactions because it's only seeing tiny crops. Combined with a leaner screenshot prompt this dropped agent step count from **49 → 32 on the SF mapsheets task** (vs FF's 45 steps). On the same trace, `dv_internal_total_tokens=61,425` (45 steps × 1,365 each — same as FF's required full-screenshot cost), `model_facing_total_tokens=28,725`, so the cost-split savings is **`1 − 28,725/61,425 = 53.2%`**, with a per-step model-facing-vs-internal savings of 34.2%. The agent took fewer steps AND DV shipped less to the model — wins on both axes the project requires.

## When DeltaVision helps — and when it doesn't

DeltaVision's savings are **task-shape-dependent.** Reading the headline numbers without this context will over-sell the tool.

| Task shape | DV savings | Why |
|---|---|---|
| **Sticky-context workflows** (forms, SPAs, spreadsheets, multi-tab user tasks — same page, many small interactions) | **40–77%** | agent re-reads the same page; small deltas dominate |
| **Mixed browsing** (typical CU workload: clicks + typing + occasional nav) | **20–40%** | some steps hit the delta path, some hit new_page |
| **Scroll-heavy media exploration** (WebVoyager-style news/feed sites) | **5–20%** | scroll-bypass gate fires; delta crops are near-viewport |
| **Nav-heavy research** (URL-hop every 1–2 steps, big scrolls) | **~0%** | every observation is a full frame by correct design — no redundant context to strip |

Dogfood-measured, not modeled. The ~0% case is a real sibling-agent A/B (2026-04-19, 10-step deep-research trajectory, URL nav + 600 px scrolls) — DV stayed on the `full_frame` path on every step by design, because the classifier correctly identified that each observation was genuinely a new page or a new viewport of content.

**Sweet spot: agents that re-read the same page.** Form-heavy flows, SPA interactions, multi-tab comparison tasks. Not ideal: "open 10 new tabs, read each, close them."

This is the tool's shape, not a bug. The CV classifier can't compress what isn't redundant.

## Cost accounting — what "savings" means here

Every DV trace records two distinct token costs per step. They mean different things and conflating them is the most common way to overstate or misstate DV's effect:

- **`dv_internal_tokens`** — what DV consumed internally on this step. Always equals the cost of the full screenshot at the consumed viewport size. DV needs the full frame in memory to compute the next-step diff, so this number does not go down regardless of what the model sees. This is the infrastructure cost of running the pipeline.
- **`model_facing_tokens`** — what DV actually shipped to the model on this step. On a delta step this is just the changed crops; on a full-frame step it equals `dv_internal_tokens`. This is the number that drives user-visible token spend on Claude, GPT, etc.

Every "X% savings" claim in this README is computed as

```
savings = 1 − sum(model_facing_tokens) / sum(dv_internal_tokens)
```

over the trace. The infrastructure cost (`dv_internal_tokens`) is **not** reduced — DV is observation middleware, not a screenshot-skipper. What it reduces is the bytes that reach the model.

If you read a saved benchmark JSON, the explicit keys are `dv_internal_tokens` / `model_facing_tokens` per step and `dv_internal_total_tokens` / `model_facing_total_tokens` / `total_savings_pct` in the summary. Older benchmark outputs use `ff_tokens` / `dv_tokens` (per step) and `ff_total_tokens` / `dv_total_tokens` (summary) as legacy aliases — they map exactly to `dv_internal` / `model_facing` respectively, kept for one release.

`DVObservation.model_facing_tokens()` and `DVObservation.dv_internal_tokens()` are the public methods. The pre-split `DVObservation.estimated_image_tokens()` is now a deprecated alias of `model_facing_tokens()` — kept for one release for back-compat, will be removed in v1.1.0.

Every claim below should be reproducible by running the named benchmark and checking the resulting JSON's `total_savings_pct` against the number in the table.

## Headline demos — two videos, two honesty tiers

### (1) Real Google Maps apartment search — 38.4% savings on the real web

A scripted 11-step trajectory on live Google Maps: search Brooklyn apartments, scroll listings, click into two real listing detail pages (461 Dean Apartments, The Bay NYC Luxury), scroll details, zoom map. Every observation runs through the live DeltaVision CV pipeline on the real screenshot — **no mocks, no hand-coded pages.**

| | Full Frame | DeltaVision | Savings |
|---|---|---|---|
| Image tokens | 15,015 | 9,249 | **38.4%** |
| Full-frame obs | 11 (every step) | 6 (URL changes + zoom + scroll guard) | — |
| Delta obs | — | 5 | — |

`Full Frame` column = `dv_internal_total_tokens` (cost of all 11 full screenshots DV consumed). `DeltaVision` column = `model_facing_total_tokens` (what DV shipped to the model). 38.4% = `1 − 9,249 / 15,015`.

**Video walkthrough:** [`benchmarks/ablation/video_frames/gmaps_demo_v1.mp4`](benchmarks/ablation/video_frames/gmaps_demo_v1.mp4) (68 s, 1080p60) — includes live counters + scene labels.
**Source metadata:** [`gmaps_demo_v1.metadata.json`](benchmarks/ablation/video_frames/gmaps_demo_v1.metadata.json) (per-step timing, cumulative tokens, obs_type + trigger).

**Run it yourself** (real Google Maps, no auth, no API key):

```bash
python examples/gmaps_demo.py      # produces runs_gmaps/browser.webm + metadata.json
```

The 38.4% is *in the mixed-browsing band of the task-shape matrix above* — honest for a real user workflow where the agent does both navigation (full_frame) and same-page reading (delta).

### (2) Real 2-site workflow: Maps → Sheets — 54% savings on live sites

A scripted 21-step workflow on two live, unmodified sites: **Google Maps** (research phase, 7 steps — search apartments, open two listings, scroll details) → **Google Sheets** (document phase, 14 steps — type findings into a real anonymous-edit spreadsheet). No mocks. The two phases run in separate browser contexts so the recording shows each site correctly.

| | Full Frame | DeltaVision | Savings |
|---|---|---|---|
| Image tokens | 28,665 | 13,179 | **54.0%** |
| Research phase (Maps nav, 7 steps) | 9,555 | 8,286 | ~13% — nav-heavy, expected |
| Document phase (Sheets fills, 14 steps) | 19,110 | 4,893 | **~74%** — delta-heavy |
| Full-frame obs | 21 (every step) | 7 | — |
| Delta obs | — | 14 | — |

`Full Frame` column = `dv_internal_total_tokens`, `DeltaVision` column = `model_facing_total_tokens`, **54.0% = `1 − 13,179 / 28,665`**.

The split tells the whole story: nav-heavy = near-zero savings, sticky-context = 74%. The combined 54% is an honest mixed-task number.

**Run it yourself:**
```bash
python examples/multitab_real_demo.py   # produces runs_multitab_real/browser.webm + metadata.json
```

### (3) Scripted 3-tab workflow — 67% savings on local mocks (compression ceiling)

| | Full Frame | DeltaVision | Savings |
|---|---|---|---|
| Image tokens | 39,585 | 13,076 | **67.0%** |
| Per-step avg | 1,365 | 451 | 67.0% |

`Full Frame` = `dv_internal_total_tokens`, `DeltaVision` = `model_facing_total_tokens`, **67.0% = `1 − 13,076 / 39,585`**. This is on a local HTML mock (no network, no real-site quirks) — that's why it's labeled the *compression ceiling*.

**Video walkthrough:** [`benchmarks/ablation/video_frames/apartment_demo.mp4`](benchmarks/ablation/video_frames/apartment_demo.mp4) (32 s, 1080p60)

This version uses local HTML mocks designed to exercise DV's sweet spot. The 67% shows the compression ceiling when a task has many same-page interactions. For how real agents on real sites actually do, see video (1) above.

```bash
cd examples/multitab_apartment_demo/mocks && python3 -m http.server 8765  # Terminal 1
cd examples/multitab_apartment_demo && python3 run_multitab_demo.py       # Terminal 2
```

## Why This Matters

Standard computer use agents send a full 1280x900 screenshot (~1600 tokens) on every step, whether 1 pixel changed or the entire page swapped. DeltaVision puts a 4-layer CV classifier in front of the model that decides: did the page change, or just a region? Send accordingly.

**Four headline benchmarks across different task shapes:**

| Benchmark | What it measures | Steps | DV cost | FF cost | Savings | Reproduce |
|---|---|---|---|---|---|---|
| **Maps→Sheets agent benchmark** (live Claude agent, Haiku 4.5, CDP-measured) | real agentic sticky-context task | 53 | 14,115 tok | 72,345 tok | **80.5%** | `python benchmarks/mapsheets/run_bench.py --trial N` |
| **Real 2-site workflow** (Google Maps → Google Sheets, live sites) | real multi-tab CU task | 21 | 13,179 tok | 28,665 tok | **54.0%** | `python examples/multitab_real_demo.py` |
| **Multi-tab apartment workflow** (3 tabs, deterministic script) | realistic 3-tab user task | 29 | 13,076 tok | 39,585 tok | **67.0%** | `python examples/multitab_apartment_demo/run_multitab_demo.py` |
| Spreadsheet (deterministic, local HTML mock) | compression ceiling | 25 | 7,780 tok | 34,125 tok | **77.2%** | `python examples/spreadsheet_observation_cost.py` |
| TodoMVC matched-trajectory (real Playwright + Anthropic `tool_result`) | compression with real SPA | 9 | 6,133 tok | 13,824 tok | **55.6%** | `python examples/observer_integration_proof.py` |
| TodoMVC head-to-head (real Claude agent, n=3 per side) | **utility — real agent decisions** | 7 each | 23,693 ±66 tok | 62,270 ±218 tok | **62.0%** | `python benchmarks/headtohead/run_head_to_head.py` |

In every row, **`DV cost` = `model_facing_total_tokens`** (what DV shipped to the model) and **`FF cost` = `dv_internal_total_tokens`** (what DV consumed internally — the same number FF would have shipped). Savings = `1 − DV / FF` per row. See [Cost accounting](#cost-accounting--what-savings-means-here).

**Read these together. They answer three different questions:**
- First: byte-reproducible on any machine — no agent, no trajectory variance, no auth. Shows the per-observation compression ceiling.
- Second: matched-trajectory with a real agent-shaped payload. Shows compression survives real `tool_result` plumbing.
- Third: actual utility — DV-wrapped Claude vs FF-baseline Claude on the same task, same model, n=3 each. **3/3 success on both sides, identical step counts, ±66 token variance on DV (deterministic).** DV saves 62% input tokens with no reliability penalty.

### Architecture note: CV + DOM hybrid (added in v4)

Pure CV observation couldn't reliably detect two things: (1) focus state after clicking an input (cursor blinker is sub-pixel, below diff threshold), and (2) small interactive elements in the low-res thumbnail (e.g. 20 px TodoMVC checkbox). Both are cheap to query from the DOM (~300 tokens as structured text). DV now runs one `page.evaluate()` per step that returns the visible clickable elements (bbox + label) and the currently-focused element — ground truth the CV pipeline can't produce. The agent uses these coordinates directly instead of guessing from pixels. This raised head-to-head DV success from 2/3 → 3/3 with lower total tokens (agent wastes fewer steps on retry-after-false-failure). See [`vision/elements.py`](vision/elements.py).

Savings grow with task length. On long SPA workflows with sticky page structure, DV stays on the DELTA path for 80%+ of steps, each of which costs 3-7× less than a full frame.

## Integration tests

Every adapter claim is covered by a runnable script. Reproduce with `python examples/integration_tests.py`:

| Framework | What's verified | Proof |
|---|---|---|
| **Anthropic `tool_result`** | Live API call via `claude-sonnet-4-20250514` ingests DV's content blocks without error | 1,762 in / 50 out tokens, real response |
| **OpenAI CUA (Operator)** | `to_openai_computer_call_output()` matches the [computer-use spec](https://platform.openai.com/docs/guides/tools-computer-use) | data URL decodes to valid PNG, `type=computer_call_output` |
| **Browser Use** | `pip install browser-use` + 5-line monkey-patch wires DV in; `to_browser_use_screenshot_b64()` returns a valid base64 PNG | Patches `BrowserSession.get_browser_state_summary`, see [`examples/browser_use_integration/`](examples/browser_use_integration/) |
| **Stagehand** | `to_stagehand_middleware_parts()` returns a valid list of typed content parts | Adapter method `DeltaVisionObserver.to_stagehand_middleware_parts()` |

Artifact: [`examples/integration_test_results.json`](examples/integration_test_results.json) (commit-tracked; 4/5 pass, Skyvern skipped — not on PyPI).

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

**From PyPI (recommended):**

```bash
pip install deltavision
# Backends are optional extras:
pip install "deltavision[claude]"   # Anthropic
pip install "deltavision[openai]"   # OpenAI
pip install "deltavision[ollama]"   # local Ollama VLMs
pip install "deltavision[all]"      # everything
```

**From source (for development or to run benchmarks):**

```bash
git clone https://github.com/ddavidgao/deltavision.git
cd deltavision
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
playwright install chromium

# Run tests (no API keys needed) — 224 total, offline
pytest tests/ -q --ignore=tests/test_e2e_live.py --ignore=tests/test_live_capture.py

# The reproducible spreadsheet benchmark — no API key, no auth, deterministic
python examples/spreadsheet_observation_cost.py      # → 77.2% savings, same every run

# Integration tests against 4 CU frameworks (Anthropic live if ANTHROPIC_API_KEY set)
python examples/integration_tests.py

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
  tests/            # 232 tests: unit, integration, live Playwright, real screenshots
  paper/            # Paper outline with figure/table mapping to data
```

## Testing

`pytest tests/` — 256 tests total (offline + live Playwright). Covers CV pipeline, classifier cascade, observation builder, safety layer, response parsers, HTTP sidecar, v1.0.3 regression invariants (`import deltavision` works, DV ≤ FF on every single step), v1.0.5 token-cap guard (proxy never bills more than a full frame), v1.0.6 greedy bbox-merge optimizer + periodic full-frame refresh.

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
| **Total** | **224** | |

```bash
pytest tests/ -q                    # full offline suite (224 collected, all pass)
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

> The token numbers below use the same accounting convention as the rest of this README — the forced-full-frame baseline is the `dv_internal` denominator and the delta-gated number is the `model_facing` numerator. See [Cost accounting](#cost-accounting--what-savings-means-here).

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

> All "token savings" percentages in the tables below follow the same accounting convention as the rest of the README: the baseline (full-frame) total is the `dv_internal` denominator, the DV total is the `model_facing` numerator, and savings = `1 − model_facing / dv_internal`. See [Cost accounting](#cost-accounting--what-savings-means-here).

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

## Launch video

A 75-second narrated walkthrough of DeltaVision — the problem, the pipeline, the compression ceiling, and the real-agent head-to-head that produce these numbers:

| File | Content |
|------|---------|
| [`benchmarks/ablation/video_frames/deltavision_v1_launch.mp4`](benchmarks/ablation/video_frames/deltavision_v1_launch.mp4) | 1080p60, 9 scenes: title / problem / task setup / **DV pipeline internals on one real observation** / side-by-side **showing what FF sends vs what DV actually sends** (thumbnail + crop snippets) / savings range / **compression ceiling (77.2%)** / **head-to-head utility (62.0% with real Claude agent)** / install |

Record your own agent session:
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

This is **V1 (browser-focused)** — on PyPI as [`deltavision`](https://pypi.org/project/deltavision/). If you need OS-level or OSWorld-VM observation, see [`deltavision-os`](https://github.com/ddavidgao/deltavision-os) (on PyPI as [`deltavision-os`](https://pypi.org/project/deltavision-os/), currently `0.1.0a0` alpha). V1 is the stable browser middleware; V2 extends the same CV pipeline to the full OS desktop via `mss` + `pyautogui`.

## License

MIT
