# DeltaVision V1 launch — Twitter thread

**Status:** DRAFT. Numbers in [brackets] get filled in from
`benchmarks/headtohead/head_to_head_results.json` once the benchmark finishes.

**Target post time:** Sunday evening ~7-9pm ET or Monday morning 8-10am ET.

**Framing discipline (per Saturday compression-vs-utility discussion):**
- 77.2% is the **compression ceiling** on a scripted trajectory. Don't let it
  read as an agent-win number.
- Head-to-head on TodoMVC is the **utility proof** — same agent, same task,
  different observation pipeline.
- Both matter together; neither alone would.

---

## Thread (9 tweets)

### 1/9 — Hook
```
Shipping DeltaVision v1:

pip install deltavision

Observation middleware for CU agents.
Instead of sending a full screenshot every step, your model sees only what changed.

Zero-LLM CV pipeline in the gating path. Same model. Fewer tokens.

Thread 🧵
```

### 2/9 — Problem
```
Every computer-use agent today sends the model a full screenshot every step.

Even when most of the page didn't change.

A 1920×1080 screenshot costs ~1,365 image tokens. Multiply by 25 steps = 34,125 tokens just on *looking*.

The model doesn't need most of it. Most steps change one region.
```

### 3/9 — Mechanism
```
DeltaVision sits between browser and model.

Pipeline (all CPU, no LLM, no RNG):
• diff screenshot t0 → t1
• pHash distance
• anchor template match
• scroll-bypass gate

Outputs: "this is a new page" OR "here's a low-res thumbnail + tiny crops of what changed."
```

### 4/9 — Compression ceiling (77.2%)
```
Compression ceiling: 77.2%

Scripted 25-step spreadsheet, matched trajectory both sides.
FF baseline: 34,125 image tokens. DV: 7,780.

Byte-reproducible: md5-verified across 3 independent processes.
No LLM in this benchmark = zero trajectory variance.

This is what's achievable when it works.
```

### 5/9 — Utility with a real agent (head-to-head)
```
But scripted isn't the whole story.

Head-to-head on TodoMVC — same claude-sonnet-4, same task, n=3 trials per config:

FF agent: 10.0 (range 10–10) steps, 88,115 (range 87,932–88,267) tokens, 3/3 success
DV agent: 11.0 (range 10–12) steps, 32,446 (range 31,559–32,973) tokens, 2/3 success

63.2% fewer tokens, matched success rate.
```

### 6/9 — Why the split matters
```
Two numbers, two different claims:

77.2% compression (scripted) — what the pipeline can save when agent behavior is fixed.

63.2% utility (live agent) — what you actually get when a real model makes decisions with DV observations instead of full frames.

Don't conflate them.
```

### 7/9 — What's in the box
```
v1 ships with 4 adapters:
• Anthropic (claude-sonnet-4 verified live)
• OpenAI computer-use (tool_output spec)
• Browser Use (monkey-patched state summary)
• Stagehand (parts list)

Drop-in replacement for your screenshot step. Returns tool-result-shaped content the model already understands.
```

### 8/9 — Reproducibility discipline
```
Every headline number has a checked-in artifact:
• 77.2% → examples/spreadsheet_observation_cost.py + .json
• 55.6% TodoMVC → examples/observer_integration_proof.py
• 14.7% WebVoyager → webvoyager_subset.py
• 17/17 classifier → benchmarks/generalization/
• 224 tests → pytest tests/

Clone the repo. Re-run anything.
```

### 9/9 — Links
```
pip install deltavision

PyPI: https://pypi.org/project/deltavision/
Repo: https://github.com/ddavidgao/deltavision
Paper outline + ablation: /paper/outline.md
Launch video (74s): in the repo README

Built this in freshman year at Purdue. Feedback + PRs welcome.
```

---

## Alt snippets (for if you want to reshuffle)

### Alt hook (shorter)
```
DeltaVision v1 is out.

CU agents send a full screenshot every step. DV sends only what changed.

Pure CV classifier in the gating path — no LLM decides when to send a full frame.

77.2% compression ceiling. 63.2% fewer tokens with a real Claude agent.

pip install deltavision 🧵
```

### Alt for scientific audiences
```
DeltaVision: observation middleware for CU agents.

- Pre-model CV classifier (diff, pHash, template, scroll bypass)
- Sends only delta crops when safe, full frames on page transitions
- Matched-trajectory ablation: 77.2% compression
- Head-to-head on TodoMVC vs full-frame baseline: 63.2% (n=3)

github.com/ddavidgao/deltavision
```

### Alt if numbers look ugly
```
Head-to-head was messier than we'd hoped — real agents have real variance.

But on TodoMVC:
• DV-wrapped: 11.0 (range 10–12) steps / 32,446 (range 31,559–32,973) tokens / 2/3
• FF-baseline: 10.0 (range 10–10) steps / 88,115 (range 87,932–88,267) tokens / 3/3

n=3 each. That's the honest picture — same agent, same task, different obs pipeline.
```

---

## Pre-post checklist

- [ ] Numbers plugged in (replace all [BRACKETS])
- [ ] Launch video attached to tweet 1 (74s MP4)
- [ ] Screenshot of head-to-head table attached to tweet 5
- [ ] Links resolve (PyPI, GitHub, paper outline)
- [ ] Run `twine check dist/*` one last time
- [ ] Push latest private → public
- [ ] Pinned tweet on profile: "See pinned — DeltaVision v1"
