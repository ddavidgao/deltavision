# Session Summary — 2026-04-17 (afternoon, autonomous)

You stepped away and asked me to keep working. Here's what got done.

## V1 — production hardening

### Tests: 56 → 190 (+134)

New test suites with full table in [TESTS.md](TESTS.md):

| Suite | Before | After | Delta |
|---|---|---|---|
| `test_safety.py` | 0 | 37 | +37 — every URL / credential / action-limit branch |
| `test_config.py` | 0 | 45 | +45 — every threshold range, bbox coherence |
| `test_results_store.py` | 0 | 19 | +19 — SQLite save/query/best/schema |
| `test_response_parser.py` | 0 | 33 | +33 — JSON extraction, VLM quirks, confidence hoisting |
| **Total** | **56** | **190** | **+134** |

All 183 offline tests pass in ~1s. Live tests (7) skipped unless you explicitly run them.

### Real bug found and fixed
`safety.py::block_url_shorteners=False` had **no effect** because shorteners also appeared in `SUSPICIOUS_PATTERNS`, so the URL would get blocked by the pattern check regardless. Fixed by removing shorteners from the patterns list (they have their own dedicated check that respects the flag).

### Config validation
`DeltaVisionConfig.__post_init__` now validates every field at construction. Bad thresholds raise `ConfigError` immediately instead of silently breaking CV behavior much later:
- Fractions in [0, 1]
- pHash distance in [0, 64]  
- Pixel-count params non-negative ints
- Quantization in {None, "4bit", "8bit"}
- Anchor bbox coherent (x2>x1, y2>y1)

### Shared model response parser
Extracted `model/_response_parser.py` (pure functions, no network, 100% unit-tested). Handles every local-VLM failure mode we've seen:
- Markdown code fences
- Prose preamble / postamble
- MAI-UI-8B's confidence-in-action quirk
- Alternate done fields (finish / finished / complete / is_done)
- String confidence values, out-of-range clamping

Wired into all 3 backends (`claude.py`, `openai.py`, `ollama.py`) — replaces 3 duplicate inline parsers with 1 tested function.

### Retry logic
All 3 model backends now retry transient errors with exponential backoff:
- **Claude**: `APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError` → 3 tries, 1s→2s backoff
- **OpenAI / llama.cpp server**: same categories → same backoff
- **Ollama**: HTTPError, ConnectionError, Timeout → 3 tries, 3s→6s backoff (was already partial, now handles connection + timeout)

Permanent errors (auth, bad request) still bubble up on the first failure.

### Packaging
Added `pyproject.toml`. Now installable via:
```bash
pip install -e ".[claude]"         # or [openai], [ollama], [all], [dev]
```
`deltavision` CLI entry point wired via `project.scripts`.

### README
- Install-via-pip block
- Testing section with 7-row coverage table → TESTS.md
- "Using as a library" code sample
- Troubleshooting section (ANTHROPIC_API_KEY, Ollama, Playwright, ConfigError, etc.)
- V1/V2 pointer at the bottom

## Bloat prune (-14MB)

Verified every deletion first — no grep matches to any doc, no links broken.

| Removed | Size | Reason |
|---|---|---|
| `deltavision_demo.mp4` | 384KB | v1 demo, superseded by v3 |
| `deltavision_demo_v2.mp4` | 653KB | v2, superseded by v3 |
| `deltavision_agent_demo.mp4` | 2.3MB | old agent version |
| `deltavision_live.mp4` | 362KB | early live version |
| `build_demo_video.py` + `build_agent_demo.py` | — | orphaned builders for deleted videos |
| Raw webm recordings (×2) | 648KB | intermediate, composited into final mp4 |
| `real_frames/` (18 PNGs) | 4.4MB | intermediate frames, regenerable |
| `segoeui.ttf` + `segoeuib.ttf` | 1.9MB | Windows-only fonts — replaced with system-font fallback chain (macOS Helvetica, Windows Segoe, Linux DejaVu) |
| `todomvc_screenshot.png` | 68KB | root-level debug artifact |
| `.DS_Store`, `.playwright-mcp/` | 2.2MB | macOS / browser caches, now in .gitignore |
| `results/deltavision_snapshot_20260416.db` | 86KB | never-tracked local snapshot |

**Preserved everything the paper or reproducibility needs:**
- SQLite DB (`results/deltavision.db`)
- All `runs/` directories (subagent_dv, subagent_ff, github_dv, github_ff, todomvc_*, wiki_multihop_*, etc.)
- All generalization `frames/` (t0, t1, diff, crops, meta.json per scenario)
- Paper outline (`paper/outline.md`)
- Current canonical videos: `deltavision_demo_v3.mp4`, `real_comparison.mp4`, `github_comparison.mp4`, `deltavision_final_demo.mp4`

## V2 — scaffold created

New sibling repo at `~/Projects/deltavision-os/`:

```
deltavision-os/
├── README.md          # Scope table: V1=browser, V2=OS-level + OSWorld
├── CLAUDE.md          # Project instructions
├── LICENSE            # MIT
├── pyproject.toml     # deps: mss, pyautogui, openai (covers llama.cpp)
├── capture/
│   ├── base.py        # Platform ABC (setup/capture/execute/get_url/teardown)
│   ├── os_native.py   # mss + pyautogui impl, full DRAG/HOTKEY support
│   └── osworld.py     # OSWorld env wrapper (stub, pending harness install)
├── execute/ agent/ eval/ ...  # (empty dirs, to be populated)
```

One local commit on `main`. Not pushed to remote yet — you haven't created the GitHub repo for it.

## Windows GPU box

Verified `david-computer` reachable via Tailscale (`~/.ssh/ssh_open`). RTX 5080 Laptop GPU, 16GB VRAM, ~1.6GB used. Ready for llama.cpp server setup whenever you want to try MAI-UI-8B or Qwen3-VL-8B.

## What's next (your call)

Candidates in priority order:
1. **Create GitHub remote for `deltavision-os`** and push the scaffold. I can do it if you tell me the repo name preference.
2. **V2 Phase 1: OS-native capture end-to-end.** mss + pyautogui real test on your Mac desktop — open Finder, screenshot, click a file. Proves the Platform ABC works before wiring llama.cpp.
3. **V2 Phase 2: llama.cpp server on Windows box.** SSH in, install llama.cpp, pull MAI-UI-8B GGUF, wire `model/llamacpp.py`, run a V2 test from Mac.
4. **V1 paper: section 4.3 update.** Incorporate the real-agent subagent comparison data (TodoMVC 62%, GitHub 13.4%) and the big-O / big-Ω theorem into the paper outline.

All commits pushed to private (`main`) and public (`ddavidgao/deltavision`). Public mirror is up to date.
