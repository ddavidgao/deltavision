# DeltaVision

Delta-first computer use agent framework. The model's primary observation is the **delta**, not the full frame.

## Architecture

```
Browser Action → CV Pipeline (diff, phash, anchor) → Classify → DELTA or NEW_PAGE → Model
```

- `vision/` — Pure CV. No LLM. Diff engine, classifier cascade, perceptual hashing.
- `agent/` — Loop, state, typed actions. The loop orchestrates vision → model → action.
- `observation/` — Builds typed observations (FullFrame or Delta) for model consumption.
- `model/` — Pluggable backends. Claude API (`claude.py`) and local VLM (`local.py`, Qwen2.5-VL).
- `config.py` — All tunable constants. Benchmark presets override defaults.
- `benchmarks/` — Task definitions and evaluators.

## Running

```bash
# Claude backend
python main.py --task "Complete the quiz" --url https://example.com --backend claude

# Local VLM (Windows 5080)
python main.py --task "..." --url ... --backend local --model Qwen/Qwen2.5-VL-7B-Instruct

# 4-bit quantized for faster iteration
python main.py --task "..." --url ... --backend local --model Qwen/Qwen2.5-VL-3B-Instruct --quantization 4bit
```

## Tests

```bash
pytest tests/ -v
```

Tests use synthetic images — no browser, no API keys needed.

## Key Invariants

1. The **model never decides** transition type. That's the CV classifier's job.
2. `t0` is always the last anchor frame (reset on NEW_PAGE events).
3. The no_change_streak mechanism forces full-frame refresh after N stuck steps.
4. All thresholds live in `config.py` — no magic numbers in logic code.

## Private/Public Sync

This is the private repo. Run `./sync-public.sh "message"` to mirror to the public repo (excludes `.claude/`, `.env`, credentials).
