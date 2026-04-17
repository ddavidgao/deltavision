# Browser Use + DeltaVision Integration

Drop-in adapter for [Browser Use](https://github.com/browser-use/browser-use) (88k stars, most popular OSS browser agent). **Monkey-patches one method.** Every screenshot the agent captures gets preprocessed through DeltaVision before hitting the VLM.

Also transitively covers [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research), which uses Browser Use as its browser backend.

## The integration (one function)

```python
from browser_use.browser.session import BrowserSession
from observer import DeltaVisionObserver

observer = DeltaVisionObserver()
_orig = BrowserSession.get_browser_state_summary

async def dv_patched(self, *args, **kwargs):
    summary = await _orig(self, *args, **kwargs)
    if summary.screenshot:
        obs = observer.observe(summary.screenshot, url=summary.url)
        summary.screenshot = obs.to_browser_use_screenshot_b64()
    return summary

BrowserSession.get_browser_state_summary = dv_patched

# Now run Browser Use as normal — every frame is DV-gated.
from browser_use import Agent
agent = Agent(task="...", llm=your_llm)
await agent.run(max_steps=10)
```

That's it. **Five lines to wire it in. Zero changes to Browser Use itself.**

`observer.to_browser_use_screenshot_b64()` returns a single base64 PNG (Browser Use expects one image per step), compositing the DV thumbnail + crops on DELTA steps and returning the full frame on NEW_PAGE.

## Why this hook

Research summary ([full report in session memory](../../../.claude/memory/learnings.md)):

- Browser Use funnels every screenshot through `BrowserSession.get_browser_state_summary()` at `browser_use/browser/session.py`. The returned `BrowserStateSummary.screenshot` is a base64 PNG string.
- That summary is then passed to `MessageManager.create_state_messages()` which attaches the screenshot to the LLM request.
- Monkey-patching the session method is strictly upstream of the message-builder — so every path that uses a screenshot gets DV automatically.
- No fork required, survives Browser Use version bumps as long as the API stays.

## Reset between agent runs

```python
observer.reset()   # drops the t0 anchor
```

Call this at the top of each new `Agent.run(...)`. Otherwise the observer compares the new run's first frame to the prior run's last frame — the classifier will fire a NEW_PAGE on the first step (which is actually the correct behavior, just uses the initial step's FF tokens "twice").

## Estimated savings

Using our measured integration proof on the same agent stack (Playwright + TodoMVC + Anthropic format):

| Steps | Baseline payload | DV payload | Savings |
|---|---|---|---|
| 9 | 1,032 KB / 13,824 tok | 604 KB / 6,133 tok | **41.5% bytes, 55.6% tokens** |

On longer runs (25-50 steps) with sticky context (SPA apps, Wikipedia navigation), savings typically hit 70-90%. See [`paper/outline.md`](../../paper/outline.md) and the Wikipedia ablation (95% savings with Qwen2.5-VL-7B).

## Known issue: Browser Use 0.12 + Ollama

Browser Use 0.12.6's native `ChatOllama` client sometimes crashes Ollama's model runner with HTTP 500 (`"model runner has unexpectedly stopped"`) when the context window is too small for the full state message + screenshot. This is a Browser Use ↔ Ollama compatibility issue, orthogonal to DeltaVision.

**Workarounds:**
1. Use `ChatOllama(model=..., ollama_options={"num_ctx": 8192})` to bump the context window.
2. Point Browser Use at Ollama's OpenAI-compat endpoint instead:
   ```python
   from browser_use.llm import ChatOpenAI
   llm = ChatOpenAI(model="qwen2.5vl:7b", base_url="http://localhost:11434/v1", api_key="none")
   ```
3. Use a larger or different model (the issue is context-window, not DV).

Once the LLM call succeeds, DV's monkey-patch is automatic.

## File listing

- `hermes_vs_dv.py` — runnable comparison harness. Configure `MODEL` and `OLLAMA_HOST`, run with and without the monkey-patch, measure payload bytes.

## See also

- [`../openclaw_integration/`](../openclaw_integration/) — DV adapter for OpenClaw (TypeScript, via HTTP sidecar)
- [`../observer_integration_proof.py`](../observer_integration_proof.py) — pure Playwright reference integration, proves 55.6% savings
- [`../../observer.py`](../../observer.py) — the `DeltaVisionObserver` + all 5 format adapters
- [`../../server.py`](../../server.py) — local HTTP sidecar for non-Python CU frameworks
