# DeltaVision Test Coverage

Visual map of what every test verifies. 190 tests total across 11 files.
Run `pytest tests/ -q` (183 pass; 7 live tests skipped without network / Playwright).

```
Total: 190 tests
├── CV pipeline         34 tests  (diff + phash + classifier + mcgrawhill real)
├── Model response      33 tests  (JSON extraction, VLM quirks)
├── Safety layer        37 tests  (URL, credentials, action limits, presets)
├── Config validation   45 tests  (every field, every bound)
├── Results store       19 tests  (SQLite, schema, persistence)
├── Integration         15 tests  (builder, action parse, state, pipeline)
├── Live (CI-skipped)    7 tests  (browser E2E, live capture)
```

## 1. CV pipeline — 34 tests

The zero-LLM classification pipeline. Everything here runs on synthetic or real pixel data; no network, no models.

### `tests/test_diff.py` — 8 tests

Verifies `vision/diff.py`: bbox extraction, thresholding, morphological dilation.

| Test class | Covers |
|---|---|
| `TestComputeDiff::test_identical_frames` | diff_ratio = 0 when frames identical |
| `TestComputeDiff::test_completely_different` | diff_ratio near 1 on full swap |
| `TestComputeDiff::test_small_change_detected` | sub-5% change produces correct bboxes |
| `TestComputeDiff::test_large_change_has_effect` | `action_had_effect=True` above threshold |
| `TestComputeDiff::test_subthreshold_noise_filtered` | JPEG noise doesn't spuriously trigger |
| `TestComputeDiff::test_max_regions_cap` | `MAX_REGIONS` enforced, largest kept |
| `TestExtractCrops::test_basic_crop` | crop_before / crop_after returned with padding |
| `TestExtractCrops::test_crop_clamped_to_image_bounds` | negative coords don't crash |

### `tests/test_phash.py` — 4 tests

Perceptual hash Hamming distance on 8×8 hashes (max 64).

| Test | Covers |
|---|---|
| `test_identical_images_zero_distance` | same image → 0 |
| `test_completely_different_high_distance` | wildly different → > 25 |
| `test_similar_images_low_distance` | small crops / brightness shifts → < 10 |
| `test_hash_size` | returns 64-bit hash |

### `tests/test_classifier.py` — 14 tests

The 4-layer cascade: URL → diff ratio → pHash → anchor template match, plus scroll bypass and animation guard.

| Test class | Layer(s) tested |
|---|---|
| `TestURLChange` (2 tests) | L1: URL delta triggers NEW_PAGE, same URL falls through |
| `TestDiffRatio` (1 test) | L2: high diff_ratio triggers NEW_PAGE |
| `TestAnchorMatch` (2 tests) | L4: lost anchor template → NEW_PAGE, persistent anchor → DELTA |
| `TestScrollBypass` (5 tests) | scroll_bypass gate: skips L2-4, URL still wins, non-scroll doesn't bypass |
| `TestAnimationGuard` (2 tests) | pHash floor: low-diff + high-pHash stays DELTA (animations) |
| `TestExtractAnchor` (2 tests) | default top-strip vs custom bbox |

### `tests/test_mcgrawhill_real.py` — 8 tests

Real McGraw-Hill screenshots (not synthetic). The site is the motivating use case.

| Test | Covers |
|---|---|
| `TestRealDiffMetrics` (3 tests) | diff_ratio on actual state transitions (question ↔ reading ↔ feedback) |
| `TestRealClassifier` (2 tests) | both transitions classify as NEW_PAGE correctly |
| `TestRealPHash` (1 test) | pHash distances match expected (18-28 range for new page) |
| `TestRealAnchor` (1 test) | nav bar anchor score ≥ 0.95 across all 3 states |
| `TestRealCropExtraction` (1 test) | crops correctly isolate the changed question stem |

## 2. Model response parsing — 33 tests

`tests/test_response_parser.py` covers `model/_response_parser.py`, the shared JSON
extractor used by Claude and OpenAI backends. Every local-VLM failure mode we've
observed in the wild gets a regression test.

| Test class | What breaks if this fails |
|---|---|
| `TestPureJSON` (3 tests) | Well-formed JSON no longer parses |
| `TestCodeFences` (4 tests) | ```json fences break backends (Qwen-VL habit) |
| `TestBraceExtraction` (3 tests) | Prose preamble / postamble breaks parsing |
| `TestFallback` (5 tests) | Malformed output crashes the loop instead of gracefully stopping |
| `TestNormalizeConfidenceHoisting` (3 tests) | MAI-UI-8B confidence-in-action quirk resurfaces |
| `TestNormalizeAltDoneFields` (5 tests) | `finish` / `finished` / `complete` aren't recognized as "done" |
| `TestNormalizeDefaults` (2 tests) | Missing fields cause KeyError downstream |
| `TestGetConfidence` (8 tests) | Numeric strings, garbage, out-of-range confidence |

## 3. Safety layer — 37 tests

`tests/test_safety.py` exercises every branch in `safety.py`. Critical because
uncensored local VLMs (Hermes, etc.) rely entirely on this layer to prevent
credential entry and suspicious URL navigation.

| Test class | Defense surface |
|---|---|
| `TestURLShorteners` (5 tests) | bit.ly / tinyurl / t.co blocked; flag disables block |
| `TestSuspiciousPatterns` (3 tests) | `.ru`, `password-reset`, `account-verify` URLs blocked |
| `TestDomainAllowlist` (4 tests) | allowlist enforced, None means permissive, empty URL ok |
| `TestCredentialDetection` (8 tests) | SSN (9 digits), credit card (13-19 digits), CVV detection; disable flag |
| `TestSensitiveFieldContext` (3 tests) | typing into `password`/`ssn` fields blocked by page context |
| `TestActionLimits` (5 tests) | oversized type, negative coords, None coords |
| `TestPresets` (4 tests) | PERMISSIVE / STRICT / EDUCATIONAL behave differently |
| `TestNonTypeActions` (3 tests) | click, scroll, key pass through credential check |
| `TestSafetyResult` (2 tests) | dataclass defaults |

**Bug found by these tests:** `block_url_shorteners=False` had no effect because shorteners appeared in both `SUSPICIOUS_PATTERNS` and the dedicated shortener check. Fixed by removing from `SUSPICIOUS_PATTERNS`.

## 4. Config validation — 45 tests

`tests/test_config.py` covers `config.py::DeltaVisionConfig.__post_init__`.

| Test class | Validator |
|---|---|
| `TestDefaults` (2 tests) | default `DeltaVisionConfig` + `MCGRAWHILL_CONFIG` preset construct cleanly |
| `TestFractions` (15 tests) | 7 fraction fields × {above 1 / below 0} + boundary test |
| `TestPHashThresholds` (4 tests) | PHASH_DISTANCE_THRESHOLD in [0,64]; PHASH_ANIMATION_MARGIN ≥ 0 |
| `TestPositiveInts` (14 tests) | 10 int fields must be non-negative; floats rejected |
| `TestQuantization` (8 tests) | only None/"4bit"/"8bit" accepted |
| `TestAnchorBBox` (6 tests) | length=4, x2>x1, y2>y1, non-degenerate |

## 5. Results store — 19 tests

`tests/test_results_store.py` covers SQLite schema and persistence. The DB is
the single source of truth for every paper figure, so regressions here would
silently corrupt published numbers.

| Test class | Covers |
|---|---|
| `TestSave` (9 tests) | ID sequencing, flattened fields, JSON blobs, legacy metric names, NULL handling |
| `TestQuery` (3 tests) | returns list of dicts, param binding, empty result |
| `TestBest` (4 tests) | lowest-time lookup, best_ms preferred over avg_ms, metrics rehydrated |
| `TestSchema` (3 tests) | tables + indexes created, reopen preserves data |

## 6. Integration — 15 tests

`tests/test_integration.py` covers the glue between modules: observation
building, action parsing, agent state, end-to-end simulated pipeline.

| Test class | Covers |
|---|---|
| `TestObservationBuilder` (3 tests) | full_frame + delta + text-delta builders |
| `TestActionParsing` (8 tests) | every `ActionType`: click, type, scroll, key, done, None, invalid, `__str__` |
| `TestAgentState` (3 tests) | initial state, no-change streak, delta ratio |
| `TestFullPipeline::test_simulated_run` (1 test) | synthetic 3-step task runs end-to-end without models |

## 7. Live tests (skipped without env) — 7 tests

Require real Playwright browser and network. Run manually with `pytest tests/test_e2e_live.py`.

| File | Tests | Covers |
|---|---|---|
| `tests/test_e2e_live.py` | 3 | Wikipedia search-and-navigate, HumanBenchmark aim trainer, per-step timing |
| `tests/test_live_capture.py` | 4 | Capture → diff → navigation → SPA content swap against live pages |

---

## How to run subsets

```bash
# Fast unit tests only (default CI target, ~1s)
pytest tests/ -q --ignore=tests/test_e2e_live.py --ignore=tests/test_live_capture.py

# Single module
pytest tests/test_safety.py -v

# By test class
pytest tests/test_config.py::TestFractions -v

# Live tests (needs Playwright browsers installed + network)
pytest tests/test_e2e_live.py tests/test_live_capture.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## What's NOT tested (documented gaps)

- **Model backend end-to-end** — tests cover response parsing in isolation but
  don't mock the actual API clients (anthropic/openai). Integration against
  real APIs is covered by `benchmarks/ablation/` runs that save to SQLite.
- **Ollama backend specifically** — `model/ollama.py` response format parsing
  should be lifted into `_response_parser.py` too (currently has its own path).
- **Local transformers backend** — `model/local.py` requires GPU and large
  downloads; tested only manually.
- **Browser executor retries** — `agent/actions.py::execute_action` has a
  JS-evaluate-then-fallback-to-Playwright path that's only covered by live
  tests.
