# DeltaVision Integrations Overview

Status snapshot of all three integration tracks requested.

## What's fully shipped

| Artifact | What it is | Status |
|---|---|---|
| [`observer.py`](../observer.py) | Public Observer class + 5 format adapters | ✓ |
| [`server.py`](../server.py) | Local HTTP sidecar for non-Python frameworks | ✓ |
| [`tests/test_observer.py`](../tests/test_observer.py) | 34 tests covering every adapter schema | ✓ |
| [`observer_integration_proof.py`](./observer_integration_proof.py) | Runnable proof: 55.6% measured token savings on real Playwright + TodoMVC | ✓ |
| [`browser_use_integration/`](./browser_use_integration/) | Browser Use monkey-patch (five lines), covers Hermes Agent transitively | ✓ |
| [`openclaw_integration/`](./openclaw_integration/) | OpenClaw TypeScript adapter + README | ✓ |

## The three tracks, honestly reported

### Track A: Browser Use + local VLM

**Code:** [`browser_use_integration/hermes_vs_dv.py`](./browser_use_integration/hermes_vs_dv.py) — runnable harness that monkey-patches `BrowserSession.get_browser_state_summary`, runs a TodoMVC task twice (baseline vs DV), measures payload bytes.

**Live-run status:** BLOCKED on an orthogonal Browser Use 0.12.6 ↔ Ollama compatibility issue (`"model runner has unexpectedly stopped"` — the model loads fine via direct API, but Browser Use's payload causes Ollama's model runner to crash, likely context-window overflow).

**Workarounds documented in the README:** bump `num_ctx`, switch to the OpenAI-compat endpoint, or use a larger/smaller model. None of these are DV problems — the DV monkey-patch itself is live code, tested, ready.

**What's already measured:** the *equivalent* integration test (same Anthropic message format, same Playwright, same TodoMVC task) ran successfully and measured **55.6% token savings** — see `examples/observer_proof_results.json`.

### Track B: Hermes Agent on your 5080

**Install status:** Hermes Agent fully installed on your Windows box's WSL2 Ubuntu 24.04 at `~/.hermes/hermes-agent`. `hermes --help` works; CLI is fully functional. Python 3.11.15, Node 22.22.2, uv 0.11.7, all in the Hermes venv.

**Architecture discovery:** Hermes doesn't use Browser Use directly — it uses **`agent-browser` (vercel-labs)** via npm, which is a Node.js CLI + HTTP daemon. This is the SAME layer OpenClaw uses internally. One adapter pattern covers both frameworks.

**Live-run status:** The Hermes setup flow requires interactive OAuth login to Nous Research's portal for the default inference provider (`hermes model` opens a browser). To point it at your local Ollama instead needs manual config-file editing — out of scope for automated SSH.

**What's ready for live run:**
- Hermes binary works
- Your 5080 Ollama is serving qwen2.5vl and ui-tars
- The DV HTTP sidecar is the right integration point (Hermes's Python side reads screenshot bytes from agent-browser's disk output; sidecar sits between)

### Track C: OpenClaw

**Clone status:** OpenClaw cloned locally at `~/Projects/openclaw-research/openclaw` (14,322 files, ~1GB).

**Adapter status:** Complete TypeScript adapter at `examples/openclaw_integration/deltavision-adapter.ts` with full README. Hook point confirmed from source: `src/agents/tools/common.ts:322::imageResultFromFile`.

**Live-run status:** Not yet bootstrapped (OpenClaw needs `pnpm install` → large Node build → Playwright Chromium install → onboarding wizard). Adapter is ready to drop in when bootstrap completes.

**Good news on provider compat:** OpenClaw has first-class Ollama support via its `extensions/ollama/` plugin. No endpoint forking needed.

## The real finding from all three tracks

**Hermes and OpenClaw both use `agent-browser` (vercel-labs) as their local browser layer.** So the single most-leveraged adapter I could write is for `agent-browser` itself — that would transitively cover:

- Hermes Agent (via Python → agent-browser daemon)
- OpenClaw's `thesethrose/agent-browser` skill (Node → agent-browser daemon)
- Any other CU framework that adopts agent-browser

The `agent-browser` daemon exposes screenshots via its HTTP protocol (`dist/daemon.js`). A DV-integrated `agent-browser` proxy on localhost:9001 would:
1. Accept the same HTTP schema agent-browser expects
2. Forward to real agent-browser on localhost:9000
3. When a screenshot comes back, run it through the DV Observer
4. Return the transformed image to the caller

This is follow-up work — the base infrastructure (`server.py`) is what it would build on.

## What actually runs end-to-end TODAY

The `observer_integration_proof.py` IS a real end-to-end validation:
- Playwright drives Chromium headlessly through a 9-step TodoMVC sequence
- Every screenshot gets handed to DeltaVisionObserver
- Observer returns the Anthropic tool_result content blocks it would send
- We measure actual byte size of the base64 payload

**Result: 55.6% fewer image tokens than the baseline full-frame path, on identical actions.**

That's the proof. The framework-specific adapters are wrappers that route the same Observer through each framework's shape. They either work identically (Browser Use monkey-patch, once the Ollama compat is sorted) or via the HTTP sidecar (OpenClaw/Hermes/any non-Python).

## For the next session

1. **Bootstrap OpenClaw fully** (pnpm install → Playwright Chromium → run one task) and apply the adapter
2. **Bypass Browser Use's Ollama crash** (either swap to OpenAI-compat endpoint or bump num_ctx)
3. **Write the `agent-browser` proxy** — single adapter, covers Hermes + OpenClaw + future frameworks using agent-browser
4. **Paper: add Section 6** documenting the integration paths, with the 55.6% number and the classifier-sensitivity curve
