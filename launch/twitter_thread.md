# DeltaVision launch — Twitter thread (v1.0.2)

**Status:** Ready to post. Numbers are final.

**Target post time:** Sunday evening ~7-9pm ET or Monday morning 8-10am ET.

**Attachment order:**
1. Tweet 1 → attach `benchmarks/ablation/video_frames/apartment_demo.mp4` (~32s, the multi-tab apartment demo — main hook)
2. Tweet 9 → optionally attach `benchmarks/ablation/video_frames/deltavision_v1_launch.mp4` (~75s, the deep technical walkthrough — supporting)

**Framing discipline:**
- **67% multi-tab apartment demo** is the visceral hook — real workflow, real time, real savings
- 77.2% is the **compression ceiling** on a scripted trajectory
- 62% head-to-head is the **utility proof** — same Claude agent, same task, different observation pipeline, 3/3 both sides
- Three numbers, three different claims. Don't conflate them.

---

## Thread (9 tweets)

### 1/9 — Hook (attach apartment_demo.mp4)
```
I shipped a thing.

DeltaVision: CU agents that send your model only what changed on screen, not a full screenshot every step.

Apartment research → spreadsheet → filter → email.
Same agent, same task.

Full Frame: 39,585 tokens.
DeltaVision: 13,076.

67% fewer tokens. ↓ video ↓

🧵
```

### 2/9 — Problem
```
Every computer-use agent today sends the model a full screenshot on EVERY step.

Even when nothing changed on screen.

A 1280×800 screenshot = ~1,365 image tokens. A 29-step task = 39,585 tokens just on *looking*.

The model doesn't need most of it.
```

### 3/9 — Mechanism
```
DeltaVision sits between browser and model.

Pipeline (all CPU, no LLM, no RNG):
• pixel diff t0 → t1
• perceptual hash distance
• anchor template match
• DOM element + focus extraction

Output: a full frame when the page actually changed, or a thumbnail + tiny crop of what moved.
```

### 4/9 — Compression ceiling
```
On a fully scripted 25-step spreadsheet benchmark, the ceiling is 77.2%.

FF: 34,125 image tokens. DV: 7,780.

No agent, no trajectory variance, no LLM. Byte-reproducible across 3 independent Python processes.

This is the compression ceiling. What's achievable when everything behaves.
```

### 5/9 — Utility with a real Claude agent
```
But scripted isn't real.

Head-to-head on TodoMVC — same claude-sonnet-4, n=3 trials each side:

FF agent: 7 steps, 62,270 tokens, 3/3 success
DV agent: 7 steps, 23,693 tokens, 3/3 success

62% fewer input tokens. Identical success rate. Identical step count.

Token variance on DV: ±66. Deterministic.
```

### 6/9 — The architectural lesson
```
CV alone wasn't enough. Two things the classifier couldn't see:

1. Focus state — cursor blinker is sub-pixel
2. Small UI targets — 20px checkboxes invisible in a 320×225 thumbnail

Fix: one JS eval per step returns DOM clickables + focus. ~300 tokens.

CV + DOM hybrid. The right primitive, not "more pixels."
```

### 7/9 — What's in the box
```
Ships with 4 adapters:
• Anthropic (claude-sonnet-4 verified live)
• OpenAI computer-use (tool_output spec)
• Browser Use (patched state summary)
• Stagehand (typed parts list)

Drop-in replacement for your screenshot step. Tool-result-shaped content — models already speak it.
```

### 8/9 — Reproducibility discipline
```
Every number has a checked-in artifact:
• 67% apartment → examples/multitab_apartment_demo/
• 77.2% scripted → examples/spreadsheet_observation_cost.py
• 62% head-to-head → benchmarks/headtohead/
• 17/17 classifier → benchmarks/generalization/
• 224 tests pass on Ubuntu + macOS × Py 3.11/3.12/3.13

Clone the repo. Re-run anything.
```

### 9/9 — Links
```
pip install deltavision

PyPI: pypi.org/project/deltavision/
Repo: github.com/ddavidgao/deltavision
Deep-dive video (75s, full technical walkthrough): attached

Built this in freshman year at Purdue.
Feedback + PRs welcome.
```

---

## Alt hook (shorter / if you want to lead differently)
```
I built a middleware that cuts your CU agent's vision bill by 67%.

Not a model. Not a fine-tune. A classifier in front of the screenshot.

Video below. Real 3-tab workflow. Same agent on both sides.

pip install deltavision 🧵
```

## Alt hook (pure number lead)
```
Claude agents watching a browser → 39,585 tokens.
Same Claude agent with DeltaVision → 13,076 tokens.
Same task. Same outcome.

67% fewer input tokens. Zero LLM in the pipeline that decides.

🧵
```

## Alt hook (infrastructure angle)
```
Built observation middleware for computer-use agents.

Sits between browser and model. Sends the model only what changed on screen, not a fresh screenshot every step.

32s demo: real agent, 3 tabs, 67% fewer tokens.

pip install deltavision 🧵
```

---

## Pre-post checklist

- [ ] Post apartment_demo.mp4 with tweet 1 (32s, 1080p60, 3.7 MB)
- [ ] Verify the big number at 0:30 reads clean on mobile preview
- [ ] PyPI link resolves (pypi.org/project/deltavision/1.0.2/)
- [ ] GitHub link resolves (github.com/ddavidgao/deltavision)
- [ ] All 9 tweets copied into Twitter's thread composer
- [ ] Pin tweet 1 to profile
- [ ] Optional: cross-post to HN as "Show HN: DeltaVision — observation middleware..."
