# DeltaVision: Observation-Level Gating for Efficient GUI Agents

## Abstract
GUI agents that receive full screenshots every step waste tokens on unchanged pixels. We introduce DeltaVision, an observation middleware that uses a zero-LLM computer vision pipeline to classify transitions and gate what the model sees. On a controlled 9-step TodoMVC task with the Anthropic tool_result message format, DeltaVision reduces image tokens by 55.6% at identical task outcomes. On a Wikipedia navigation task with Qwen2.5-VL-7B, the DeltaVision-wrapped agent completes the task in 3 steps while the full-frame baseline hits the 50-step limit without completing (not a direct token comparison, but an outcome comparison). Our 4-layer classifier cascade achieves 100% accuracy across 17 scenarios on 8 diverse websites using default thresholds, with 41.6ms median CV overhead per step measured on commodity laptop hardware.

## 1. Introduction
- Computer use agents send full screenshots every step (1600+ tokens each)
- This is wasteful: most steps change <5% of pixels
- Prior work optimizes the model (GUI-KV, GUIPruner) or uses DOM (Agent-E)
- DeltaVision is the first to gate observations with a CV pipeline BEFORE the model
- Key insight: send less to the model, not skip the model

**Figure 1:** Pipeline diagram. Browser -> [CV Pipeline] -> Observation (delta or full) -> Model -> Action

## 2. Related Work
- **Agent-E** (Zhu et al. 2024): DOM-text extraction, not visual. Requires page structure access.
- **GUIPruner** (Li et al. 2025): Prunes internal model attention on UI regions. Model-specific.
- **GUI-KV** (Wang et al. 2025): KV cache optimization for GUI agents. Internal optimization.
- **WebVoyager** (He et al. 2024): Benchmarks web agents. Full screenshot baseline.
- **CogAgent** (Hong et al. 2024): GUI-specialized VLM. Orthogonal — DeltaVision wraps it.
- DeltaVision differs: operates at the observation layer, model-agnostic, zero LLM classification.

## 3. Method

### 3.1 4-Layer Classifier Cascade
Ordered cheapest to most expensive:
1. **URL change** (free) — catches traditional navigation
2. **Diff ratio** (numpy) — catches SPA full-content swaps
3. **Perceptual hash** (PIL) — catches visual layout changes even with low diff ratio
4. **Anchor template match** (cv2) — catches SPA nav where URL stays same

**Figure 2:** Cascade diagram with example triggers from each layer.
Data: See `benchmarks/generalization/results.json` for per-layer trigger counts.

### 3.2 Scroll-Aware Gating
Scrolling shifts the viewport but not the page state. Without handling, pHash distance hits 28 (threshold: 20) and triggers false NEW_PAGE. We add a scroll bypass gate that skips Layers 2-4 when the last action was scroll.

**Figure 3:** Before/after diff from `frames/wikipedia_scroll_long_article/`
- diff.png showing 33% pixel change from scroll
- Without bypass: classified NEW_PAGE (wrong)
- With bypass: classified DELTA (correct), agent re-anchors

### 3.3 Observation Types
- **Full frame**: Sent on NEW_PAGE events. ~1600 tokens.
- **Delta**: Sent on DELTA events. Diff image + cropped before/after regions. ~400 tokens.

**Figure 4:** Side-by-side from `frames/todomvc_spa_add_items/`
- t0.png (empty todo list)
- t1.png (3 items added)
- diff.png (only list region changed)
- Crop before/after showing exact changed region

## 4. Experiments

### 4.1 Classifier Generalization
- 17 scenarios across 7 websites (Wikipedia, HumanBenchmark, HN, TodoMVC, GitHub, example.com, dynamic SPA)
- 100% accuracy with DEFAULT config (no site-specific tuning)
- Tests all 4 cascade layers + scroll bypass

**Table 1:** Per-site classifier results (from SQLite run #5)
**Figure 5:** Layer trigger distribution across sites

### 4.2 Reaction Time Benchmark (CV-only)
- No model needed — pure CV pipeline speed test
- Best: 65ms, Avg: 79ms (beats human median 273ms)
- 173x faster than Claude standard computer use (13,491ms)
- Detection-to-click: 10ms avg

**Table 2:** Reaction time comparison (from SQLite runs #6, #10)

### 4.3 Ablation: Delta Gating vs Full-Frame
- Same task: Wikipedia search + navigate to article
- Same model: Qwen2.5-VL-7B (Q4_K_M) via Ollama
- DeltaVision: 3 steps, 4,000 tokens, task completed (29s)
- Full-frame: 50 steps (max), 81,600 tokens, task FAILED (267s)
- **Outcome:** DeltaVision completes the task; full-frame does not. Per-step observation cost is ~3-7× cheaper on DELTA steps. Reporting raw outcomes rather than a single headline percentage, since the two runs don't execute the same number of steps.

**Table 3:** Ablation comparison (from SQLite runs #11, #12)
**Figure 6:** Per-step observation type (delta vs full frame) over task progression

### 4.4 Model Comparison
- Qwen2.5-VL-7B: Completed task, 80% delta ratio, ~6.6s/step
- MiniCPM-V-8B: Failed — could not ground UI elements, 0% task completion
- Implication: delta observations require adequate GUI grounding capability

**Table 4:** Model comparison (from SQLite runs #7, #13)

## 5. Discussion
- Delta gating makes smaller models viable by reducing observation complexity
- The CV pipeline adds ~40ms overhead per step (negligible vs model inference)
- Scroll handling is necessary — without it, pHash false-positives at 28/64
- Site-specific presets (McGraw-Hill) can further optimize for known domains
- Token reduction scales with task length — longer tasks save more

## 6. Limitations
- Scroll bypass is a heuristic — SPA nav during scroll could be missed
- Anchor template matching assumes persistent nav bars (fails on full-screen apps)
- Reaction benchmark limited to 5 clean rounds (site overlay state issue)
- Only tested on 2 VLMs — larger model comparison needed
- No DOM integration — purely visual, misses non-visible state changes

## 7. Conclusion
DeltaVision demonstrates that observation-level gating with a zero-LLM CV pipeline reduces GUI agent token consumption on the DELTA path by a factor of 3-7× per step, and enables task completion for 7B parameter models that hit step limits without completing under full-frame observation. The approach is model-agnostic, adds ~40ms overhead per step, and generalizes across diverse websites without site-specific tuning. On a controlled TodoMVC comparison at the Anthropic tool_result format, DeltaVision reduces total image tokens by 55.6%.

---

## Data Sources (all in SQLite: results/deltavision.db)

| Figure/Table | SQLite Run | Also in |
|---|---|---|
| Table 1 | Run #5 | benchmarks/generalization/results.json |
| Table 2 | Runs #6, #10 | - |
| Table 3 | Runs #11, #12 | benchmarks/ablation/results.json |
| Table 4 | Runs #7, #13 | - |
| Figure 3 | - | benchmarks/generalization/frames/wikipedia_scroll_long_article/ |
| Figure 4 | - | benchmarks/generalization/frames/todomvc_spa_add_items/ |
| Figure 6 | Run #11 transition_log | - |
