# Multi-Tab Apartment Demo

Real-world 3-tab agent workflow measured with DeltaVision — the source
behind the launch video at `benchmarks/ablation/video_frames/apartment_demo.mp4`.

**The task**: research 5 Brooklyn apartment listings on one tab, fill a
comparison spreadsheet on another, filter to those under $3000, then draft
an email to the broker.

**The result**: **67.0% fewer input tokens with DeltaVision** compared to a
full-frame baseline on the same 29-step trajectory.

| | Full Frame | DeltaVision |
|---|---|---|
| Input tokens | 39,585 | 13,076 |
| Per-step avg | 1,365 | 451 |
| Full-frame obs | 29 (every step) | 3 (initial + 2 tab switches) |
| Delta obs | — | 26 |
| Runtime | 20.8s | 20.8s (same trajectory) |

## Reproducing

```bash
# Terminal 1 — serve the 3 HTML mocks
cd examples/multitab_apartment_demo/mocks
python3 -m http.server 8765

# Terminal 2 — run the scripted trajectory + DV observation pipeline
cd examples/multitab_apartment_demo
python3 run_multitab_demo.py
```

The script drives one Chromium through three "tabs" (via URL navigation to
`listings.html` / `spreadsheet.html` / `email.html`) while capturing:

- `runs_multitab/browser.mp4` — continuous 20s recording of the browser session
- `runs_multitab/step_NN/capture.png` — per-step screenshot (the input to both FF and DV accounting)
- `runs_multitab/step_NN/dv_obs.json` — DV's decision per step: obs type, trigger, diff, pHash, anchor score, est tokens
- `runs_multitab/step_NN/dv_thumbnail.png` + `dv_crop_NN.png` — what DV actually emits for delta observations
- `runs_multitab/metadata.json` — aggregate summary with per-step timestamps + running token totals

## What DV sees on each tab switch

Tabs in this demo are simulated via URL navigation in a single Chromium page
(rather than OS-level browser tabs). DV's CV classifier treats each URL
change as a NEW_PAGE event and re-anchors. That's why the 3 `full_frame`
observations in the trajectory correspond exactly to: initial listings load,
switch to spreadsheet, switch to email. Every other step is a `delta`.

## Why this is a fair comparison

Both FF and DV measurements run on the **same captured screenshots**. The
trajectory is scripted (deterministic). FF cost is Anthropic's image-token
formula `max(75, w*h/750)` applied to the full-resolution screenshot. DV
cost is what the observer pipeline actually emits — thumbnail (~100 tokens)
+ up to 2 crops of the changed region (~100-300 tokens each) for delta
observations, or the full frame for NEW_PAGE events.

No LLM is in the loop for this measurement — we're measuring what the
observation layer sends to the model, independent of whatever the model
does with it.

## Checked-in artifacts

- [`mocks/`](mocks/) — 3 HTML pages (listings, spreadsheet, email) served at a local port
- [`run_multitab_demo.py`](run_multitab_demo.py) — scripted trajectory runner
- [`metadata.json`](metadata.json) — canonical result used in the launch video

The video composition source lives in `~/Projects/dv-video-scratch/code_review_demo/remotion/src/ApartmentDemo.tsx`
(outside the repo — Remotion project scaffolding).
