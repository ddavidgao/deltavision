# DeltaVision — TODO / Roadmap

Post-V1 improvements. Open a PR or a focused session against these.

## Context / history improvements (priority: high — affects agent reliability)

The head-to-head benchmark on TodoMVC revealed that DV's *history* representation was too sparse: delta history used to be pure-text (`[Step N] After: click(...) | Effect: true | Change: 5.1%`), stripping the visual context that the *current* delta observation already contained.

**Shipped in v1.0.2 (DOM + focus hybrid).** After iterating through three benchmark versions, the successful fix was *not* more pixels but a structured DOM layer:

| | v1 (CV only) | v2 (thumbnail in history) | v3 (DOM clickables alone) | **v4 (DOM + focus)** |
|---|---|---|---|---|
| DV success | 2/3 | 2/3 | 1/3 | **3/3** |
| DV avg tokens | 32,446 | 37,837 (+17%) | 36,017 | **23,693 (−27%)** |
| DV avg steps | 11.0 | 10.3 | 11.0 | **7.0** |
| Savings vs FF | 63.2% | 57.0% | — | **62.0%** |
| Step variance | ±1.0 | ±1.2 | ±1.7 | **±0.0** |

**The failure modes we iterated through:**
1. *v1 failure* — agent clicked input, no visible pixel change (cursor blinker sub-pixel), concluded click failed, pressed Tab, lost focus.
2. *v2 hypothesis* — add thumbnail to history so agent sees prior state visually. Result: thumbnail too low-res to show 20 px checkbox (5 px in 320×225). Changed failure mode but not rate.
3. *v3 hypothesis* — add DOM clickable list. Result: agent had exact checkbox coords but still couldn't tell whether clicking the input succeeded (focus state still invisible). 1/3 — worse than v1.
4. *v4 hypothesis* — DOM clickables **AND** focus state from `document.activeElement`. Agent now knows both "where can I click" and "what already has focus." Result: 3/3, zero step variance, agent deterministic.

**Architectural principle:** CV alone can't answer "what does the page know about its own state?" The DOM can — for free. The v1 architecture was over-indexed on pixels; v4 is a CV + DOM hybrid where each layer handles what it's cheapest at.

Code: [`vision/elements.py`](vision/elements.py) — one `page.evaluate()` returns both.

Artifacts preserved for comparison:
- v1 baseline: `benchmarks/headtohead/head_to_head_results_v1.json`
- v2 (thumbnail-in-history): `head_to_head_results_v2_experimental.json`
- v3 (DOM only): `head_to_head_results_v3_dom_only.json`
- v4 (DOM + focus, canonical): `head_to_head_results.json`

**Still open:**

- [ ] **Smarter OCR per step.** Currently DV emits text_deltas only when `diff_ratio` is tiny enough for reliable OCR. Extend: always run cheap OCR on the delta regions (even when pixel diff is large) and include short snippets in the prompt. Would let the agent reason about *semantic* change ("the word 'submitted' appeared near my click") instead of just pixel change.

- [ ] **Rolling "what's on the page" text state.** Maintain an append-only text log of visible text across steps, pruned to last N turns. Gives the agent a cheap textual ground-truth of page content without sending images. Think: `state.page_text_log`.

- [ ] **Action-contextual OCR.** When `last_action.type == "click"`, OCR a small window around the click coordinates post-action. Include in the observation as e.g. `"You clicked near: 'Submit', 'Cancel', 'Input: empty'"`. Essentially visual-to-textual lift of the click landing zone.

- [ ] **Focus state detection.** Add a layer to the CV pipeline that detects probable text-cursor location (thin vertical blinker, text-field highlight ring). Emit as structured state: `focus: {x, y, kind: "input" | "textarea"}`. Would let the agent know where focus is without seeing the full screen.

- [ ] **Tunable history depth.** Right now `max_history = 5` is hardcoded. Make it a config knob tied to token budget. On long tasks, maybe 10+ is worth the cost; on short tasks, 3 suffices.

- [ ] **Action verification loop.** If `diff_result.action_had_effect` is False, currently we just increment `no_change_streak`. Consider: immediately re-emit a full frame before the next action, not after N stuck steps. Faster recovery.

## Task-shape sensitivity (measured, 2026-04-19)

A sibling-agent A/B dogfood ran a deep-research trajectory (10 steps, URL navigation + 600px scrolls, no idle re-observes) and got **0% savings** — identical bytes transferred, DV arm +4% from DOM header overhead.

| Arm | Observations | Bytes | Delta% |
|---|---|---:|---:|
| Baseline (no DV) | 10 full screenshots | 1.99 MB | — |
| DeltaVision v1.0.3 | 10 full_frame (0 delta) | 2.07 MB | 0% |

**Why:** every observation was either URL change → NEW_PAGE → full_frame, or 600px scroll → `crop_covers_frame` guard (v1.0.3) → full_frame. DV never entered the delta path.

This is **not a bug** — it's DV behaving exactly as designed. But it means the headline "67% / 77%" numbers don't generalize to nav-heavy research workloads.

### Task-shape matrix (to add to README + pitch)

| Shape | DV savings | Why |
|---|---|---|
| Data entry in a form (scripted) | **77.2%** | same page, many small deltas — delta path every step |
| Multi-tab user workflow (apartment demo) | **67.0%** | 3 tab switches → 3 full frames, 26 deltas |
| Real agent on SPA (TodoMVC head-to-head) | **62.0%** | same page, agent types sequentially |
| Matched-trajectory SPA (TodoMVC replay) | **55.6%** | same as above, scripted |
| **Nav-heavy research (new finding)** | **~0%** | URL changes + big scrolls → every step is full_frame |
| Scroll-dominant (WebVoyager subset) | 14.7% | lots of big scrolls, guard fires often |

**Marketing implication:** the headline needs a qualifier. "Up to 77% on sticky-context tasks, ~0% on nav-heavy research." The claim stays honest + defensible.

Follow-up to add in a future pass:
- [ ] Dedicated "When DeltaVision helps (and when it doesn't)" section at the top of README
- [ ] Thread qualifier on tweet 5: "Sweet spot: agents that re-read the same page. Not for hop-page-every-2-steps research."
- [ ] Consider a low-overhead mode for detected nav-heavy sessions (skip DOM extraction when URL changes every step)

## Benchmark infrastructure

- [ ] **n>3 head-to-head trials** for statistical significance. Currently n=3 limited by Anthropic API rate limits. Spread across time + use cache-control to reduce cost.
- [ ] **Cross-machine reproducibility test** of scripted 77.2% benchmark. Run on a different OS / different Chromium build. Report pixel-level and classification-level drift. **PARTIAL: GitHub Actions nightly workflow now runs this on Ubuntu; extend to Windows.**
- [ ] **More tasks in head-to-head**: Wikipedia article lookup (more text-heavy), HackerNews navigation, Gmail compose flow. Different failure modes will surface.

### Known cross-platform classifier drift

**HumanBenchmark reaction-time page, "idle" scenario.** On macOS + Chromium the 17/17 classifier suite is fully green. On Ubuntu headless Chromium (GitHub Actions), one scenario consistently drifts: the reaction-time idle page shows ~24% pixel diff + pHash distance 36 between two captures taken seconds apart, which exceeds the NEW_PAGE pHash threshold and gets classified as `new_page` instead of `delta`.

Cause is almost certainly a subtle animation on HumanBenchmark's page (pulsing loading indicator, cursor, etc.) that renders differently between runner environments. CI workflow tolerates this as a known drift (≥16/17 passes; >1 fails). Fix options when revisited:

1. Widen NEW_PAGE pHash threshold (risky — might miss real page transitions elsewhere)
2. Add per-site classifier tuning (already exists as `MCGRAWHILL_CONFIG` preset — add `HUMANBENCHMARK_CONFIG`)
3. Drop humanbenchmark from the generalization suite and replace with a more stable site
4. Capture ~5 snapshots of the "idle" state and use median pHash instead of pairwise diff

## Model backends

- [ ] **Qwen3-VL-8B** via llama.cpp server (mentioned in `project_model_landscape_2026_04.md`). Beats UI-TARS-1.5 on ScreenSpot-Pro. Worth comparing DV+Qwen3-VL vs DV+Claude on utility.
- [ ] **MAI-UI-8B** alternative.
- [ ] **Claude Opus 4.7** as the backend for claude.py — currently default is Sonnet-4. Compare agent reliability with a stronger model.

## Packaging

- [ ] **`cache-control` blocks** in the Anthropic API messages. System prompt + early history entries are repeated every step; marking them cacheable would shrink *charged* input tokens by another ~70% (not raw tokens, but billed cost).
- [ ] **CI on public repo** — `gh auth refresh -s workflow` still blocks the workflow file from syncing. Need to resolve OAuth scope.
- [ ] **V2 packaging audit** — `deltavision-os==0.1.0a0` has the same `py-modules` pattern that broke V1's v1.0.0. If V2 ever promotes past alpha, verify fresh-install imports work from a non-repo venv.

## Paper

- [ ] Prose in `paper/outline.md`. Keep claims tempered: "compression ablation" and "utility on N tasks", not "DV-agent wins."
- [ ] Figure: side-by-side per-step token trace (FF vs DV vs DV-with-history-thumbnail) from this benchmark — shows where the savings compound.
- [ ] Figure: classifier decision tree visualized, with confusion matrix on the 17-scenario generalization set.
