# DeltaVision

**Delta-first observation middleware for GUI agents.** Sends models only what changed on screen — not a full screenshot every step.

Standard computer use agents re-process the entire screen from scratch on every action. DeltaVision puts a CV pipeline in front of the model that classifies transitions and routes observations through a tiered system: skip the model entirely for no-ops, send text-only for tiny changes, send cropped regions for moderate changes, and only send full frames on navigation. The model still reasons — it just reasons about less.

## Results

| Benchmark | DeltaVision | Standard Claude CU | Human | Speedup |
|-----------|-------------|---------------------|-------|---------|
| Reaction time (best) | **412ms** | 13,491ms | 273ms | **30x** |
| Detection→click | **6ms** | ~5,000ms | ~50ms | **800x** |
| CV pipeline overhead | **42ms** | — | — | — |

## How It Works

```
Browser Action
    │
    ▼
┌──────────────────────────────────────┐
│  DeltaVision CV Pipeline (no LLM)   │
│                                      │
│  1. Capture screenshot               │
│  2. Compute pixel diff vs anchor     │
│  3. Classify: DELTA or NEW_PAGE      │
│     (URL check → diff ratio →        │
│      perceptual hash → anchor match) │
│  4. Build observation (crops/text)   │
└──────────────────┬───────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
      DELTA path       NEW_PAGE path
      crops + diff     full screenshot
      (cheap)          (expensive, rare)
          │                 │
          ▼                 ▼
┌──────────────────────────────────────┐
│  Any Model Backend                   │
│  Claude / GPT-4o / Ollama / Local    │
└──────────────────────────────────────┘
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/ddavidgao/deltavision.git
cd deltavision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Run tests (no API keys needed)
pytest tests/ -v

# Run reaction time benchmark (pure CV, no model)
python benchmarks/reaction/run_reaction.py --rounds 5
```

### With a model backend

```bash
# Claude
export ANTHROPIC_API_KEY=sk-...
python main.py --task "Search for sedimentary rocks" --url https://en.wikipedia.org --backend claude

# OpenAI
export OPENAI_API_KEY=sk-...
python main.py --task "..." --url ... --backend openai

# Local via Ollama (free, no API key)
ollama pull qwen2.5vl:7b
python main.py --task "..." --url ... --backend ollama --model qwen2.5vl:7b

# GUI-specialized model
ollama pull 0000/ui-tars-1.5-7b
python main.py --task "..." --url ... --backend ollama --model 0000/ui-tars-1.5-7b
```

### Safety modes

```bash
# Block credential entry, URL shorteners, suspicious domains
python main.py --task "..." --url ... --safety strict

# Allowlist educational sites only
python main.py --task "..." --url ... --safety educational
```

## Architecture

```
deltavision/
├── vision/          # CV pipeline: diff, pHash, classifier, capture
├── agent/           # Loop, state, typed actions
├── observation/     # Builds model input (FullFrame or Delta)
├── model/           # Pluggable: claude, openai, ollama, local, scripted
├── safety.py        # Model-agnostic action validation
├── config.py        # All thresholds, site presets
├── results/         # SQLite result store
├── benchmarks/      # Reaction time, site registry
└── tests/           # 49 tests: unit, integration, live browser, real screenshots
```

## Key Design Decisions

1. **The model never decides transition type.** The CV classifier handles it — deterministic, sub-millisecond, testable.
2. **Speed comes from sending less, not skipping the model.** The model still reasons; it just gets cropped regions instead of full screenshots.
3. **Safety is framework-level, not model-level.** Critical for uncensored local models (Hermes, etc.) that won't refuse dangerous actions on their own.
4. **Backend-agnostic.** Same observation format whether Claude, GPT-4o, Qwen, or UI-TARS is reasoning.

## Local VLM Setup (RTX 5080 / 16GB VRAM)

```bash
# Recommended: UI-TARS 1.5 (GUI-specialized, ByteDance, Apache 2.0)
ollama pull 0000/ui-tars-1.5-7b    # ~4.4GB, runs in ~6-8GB VRAM

# Alternative: Qwen2.5-VL (general VLM, strong vision)
ollama pull qwen2.5vl:7b           # ~4.4GB, Q4 quantization

# For faster iteration
pip install -r requirements-local.txt  # torch + transformers
python main.py --backend local --model Qwen/Qwen2.5-VL-3B-Instruct --quantization 4bit
```

## License

MIT
