# OpenClaw + DeltaVision Integration

Drop-in adapter for [OpenClaw](https://github.com/openclaw/openclaw). Every browser screenshot gets preprocessed through DeltaVision before hitting the model.

## Architecture

OpenClaw is TypeScript; DeltaVision is Python. Integration via HTTP sidecar:

```
OpenClaw agent-browser takes screenshot
    ↓  buffer in Node
OpenClaw writes PNG to disk
    ↓
imageResultFromFile(path) [PATCHED]
    ↓  POST /observe with base64 PNG
DeltaVision HTTP sidecar (localhost:9000)
    ↓  runs CV pipeline (diff + phash + classifier)
    ↓  returns transformed b64
imageResult() → model message block
```

## Installation

**Step 1: Start the DeltaVision sidecar**

```bash
cd /path/to/deltavision
python server.py --port 9000
# Listens on 127.0.0.1:9000 — localhost only, no auth
```

**Step 2: Copy the adapter into OpenClaw**

```bash
cp deltavision-adapter.ts /path/to/openclaw/src/agents/tools/deltavision-adapter.ts
```

**Step 3: Wire it in (one-line change in OpenClaw)**

Edit `openclaw/extensions/browser/src/browser-tool.actions.ts` (search for `imageResultFromFile` import):

```ts
// before:
import { imageResultFromFile } from "@openclaw/agents/tools/common";

// after:
import { imageResultFromFile } from "@openclaw/agents/tools/deltavision-adapter";
```

Or use the test-injection hook that's already exposed (no source change needed):

```ts
import * as browserToolActions from "@openclaw/extensions-browser/browser-tool.actions";
import { imageResultFromFile as dvImageResultFromFile } from "@openclaw/agents/tools/deltavision-adapter";

browserToolActions.browserToolActionDeps.imageResultFromFile = dvImageResultFromFile;
```

**Step 4: Reset between runs**

At the start of each new agent run, call:
```ts
import { dvResetSession } from "@openclaw/agents/tools/deltavision-adapter";
await dvResetSession();
```
This drops the prior t0 anchor inside the sidecar so the new run starts clean.

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `DELTAVISION_ENABLED` | `true` | Set to `false` to bypass DV and use stock behavior |
| `DELTAVISION_URL` | `http://127.0.0.1:9000` | Sidecar endpoint |
| `DELTAVISION_FORMAT` | `raw` | Output format — `raw` returns a single composited image for OpenClaw |

## Graceful fallback

If the DV sidecar is down, unreachable, or returns an error, the adapter silently falls through to OpenClaw's original `imageResultFromFile`. The agent keeps working; it just doesn't get DV's savings for that frame. A warning is logged per failure.

## What DV adds to the agent's context

Every screenshot result now includes extra `details.deltavision` metadata:

```json
{
  "deltavision": {
    "obs_type": "delta",
    "trigger": "none",
    "diff_ratio": 0.033,
    "phash_distance": 10,
    "anchor_score": 0.985,
    "estimated_tokens": 420
  }
}
```

Useful for telemetry, dashboards, A/B testing DV vs no-DV at the agent level.

## What changes in the model's input

**Without DV:** every step sends a full 1200×700 screenshot (~1600 image tokens).

**With DV:**
- Initial step + URL navigation: full screenshot (same as baseline)
- Subsequent DELTA steps: composited thumbnail (320×225 with green change-boxes) + crops (~400 tokens total)

The model still sees everything it needs. It just no longer wastes tokens on the 95% of pixels that didn't change.

## Performance

- CV pipeline overhead: ~40ms per step (measured on a 2024 MacBook Air M-series)
- Network overhead (localhost HTTP): <5ms
- Model inference savings: 1000–1400 tokens per DELTA step (roughly 50-90% cheaper depending on task)

The overhead is negligible next to typical VLM inference times (1–10s per step).

## Testing

```bash
# With the sidecar running:
curl -s http://127.0.0.1:9000/health
# → {"status": "ok", "version": "1.0.0"}

curl -s http://127.0.0.1:9000/state
# → {"step": 0, "last_classification": null, ...}
```

Then run your OpenClaw agent with `DELTAVISION_ENABLED=true` and check the logs for `deltavision` entries in the tool result details.
