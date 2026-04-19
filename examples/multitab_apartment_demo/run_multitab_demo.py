"""
run_multitab_demo.py — scripted multi-tab apartment demo runner.

Drives ONE Chromium page through 3 simulated "tabs" (listings → spreadsheet →
email) via URL navigation, with Playwright's built-in video recording on. At
every step we also:
  1. Capture a still screenshot
  2. Compute FF cost = image tokens of the full screenshot
  3. Compute DV cost = what DV's observation pipeline actually emits
  4. Record timestamp + per-step metadata

This gives us BOTH a continuous browser recording (for the hero video) AND
per-step measurements + still frames (for Remotion overlays + reproducibility).

No subagent, no Anthropic API. Just Playwright + DV's CV pipeline running on
real screenshots. Fully deterministic: same action sequence → same frames →
same token counts.

Outputs:
  runs_multitab/
    metadata.json           — aggregate summary + per-step list w/ timestamps
    browser.webm            — continuous screen recording of the whole session
    browser.mp4             — ffmpeg conversion (if ffmpeg available)
    step_NN/
      capture.png           — still screenshot at end of this step
      dv_obs.json           — DV decision: obs_type, trigger, diff, phash, tokens
      dv_thumbnail.png      — DV delta thumbnail (delta steps only)
      dv_crop_NN.png        — DV delta crops (delta steps only)
      action.json           — what we did this step
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

# Add DV repo so we can import the observer directly
# When run from the DV repo root, observer.py is importable directly.
# When run from this subdirectory, add repo root to path.
_here = Path(__file__).resolve().parent
for _candidate in (_here.parent.parent, _here.parent, _here):
    if (_candidate / "observer.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from observer import DeltaVisionObserver  # noqa: E402

# ---- Config ----
MOCK_BASE = "http://localhost:8765"  # launch with: python3 -m http.server 8765 in ./mocks
VIEWPORT = {"width": 1280, "height": 800}
OUT_DIR = Path(__file__).parent / "runs_multitab"

# Timing controls (ms). Each value is how long to pause after the action so
# the viewer can see the effect. These tune the pacing of the final video.
PACING = {
    "initial_load":        800,
    "research_scroll":     1200,
    "tab_switch":          600,
    "fill_cell":           200,  # fast — data entry should feel snappy
    "fill_row_gap":        100,
    "filter_click":        1000, # linger on the filtered view
    "email_type":          350,
    "email_send":          1200, # let the "Sent ✓" feel triumphant
}


def image_tokens_anthropic(path: Path) -> int:
    """Anthropic image-token formula: max(75, w*h/750)."""
    with Image.open(path) as img:
        w, h = img.size
    return max(75, int((w * h) / 750))


# ---- Scripted trajectory ----
TRAJECTORY = [
    # Phase 1: Research on listings
    {"phase": "research",   "tab": "listings",     "url": "listings.html",    "action": "load listings"},
    {"phase": "research",   "tab": "listings",     "url": "listings.html",    "action": "scroll to see all 5"},

    # Phase 2: Fill spreadsheet (tab switch = URL change)
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "switch to spreadsheet"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 1: title"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 1: price"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 1: sqft"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 1: hood"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 2: title"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 2: price"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 2: sqft"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 2: hood"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 3: title"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 3: price"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 3: sqft"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 3: hood"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 4: title"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 4: price"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 4: sqft"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 4: hood"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 5: title"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 5: price"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 5: sqft"},
    {"phase": "fill",       "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "row 5: hood"},

    # Phase 3: Filter
    {"phase": "filter",     "tab": "spreadsheet",  "url": "spreadsheet.html", "action": "click Filter under 3000"},

    # Phase 4: Email
    {"phase": "email",      "tab": "email",        "url": "email.html",       "action": "switch to email"},
    {"phase": "email",      "tab": "email",        "url": "email.html",       "action": "type To"},
    {"phase": "email",      "tab": "email",        "url": "email.html",       "action": "type Subject"},
    {"phase": "email",      "tab": "email",        "url": "email.html",       "action": "type Body"},
    {"phase": "email",      "tab": "email",        "url": "email.html",       "action": "click Send"},
]

APARTMENTS = [
    {"title": "Sunny 1BR - Park Slope",            "price": 2850, "sqft": 650,  "hood": "Park Slope"},
    {"title": "Spacious 2BR Loft - Williamsburg",  "price": 4200, "sqft": 1050, "hood": "Williamsburg"},
    {"title": "Brownstone Garden 1BR - Bed-Stuy",  "price": 2495, "sqft": 720,  "hood": "Bed-Stuy"},
    {"title": "Modern 2BR w Roof - Crown Heights", "price": 3400, "sqft": 900,  "hood": "Crown Heights"},
    {"title": "Cozy Studio - Bushwick",            "price": 1950, "sqft": 420,  "hood": "Bushwick"},
]

EMAIL_TO = "maya@bkrentals.nyc"
EMAIL_SUBJECT = "Viewing request: 2 apartments this weekend"
EMAIL_BODY = (
    "Hi Maya,\n\n"
    "I narrowed down to two listings under $3000:\n"
    "  - Brownstone Garden 1BR, Bed-Stuy ($2,495)\n"
    "  - Cozy Studio, Bushwick ($1,950)\n\n"
    "Could we tour both this Saturday? Morning slots work best for me.\n\n"
    "Thanks,\nDavid"
)


def perform_action(spec: dict, page) -> dict:
    """Execute the action. Returns detail dict for logs."""
    action = spec["action"]
    detail = {"action": action}

    if action == "load listings":
        detail["note"] = "initial navigate already done"

    elif action == "scroll to see all 5":
        page.evaluate("window.scrollTo({top: 280, behavior: 'smooth'})")
        page.wait_for_timeout(PACING["research_scroll"])

    elif action.startswith("switch to"):
        detail["note"] = "tab switch via URL nav already done"
        page.wait_for_timeout(PACING["tab_switch"])

    elif action.startswith("row ") and ":" in action:
        _, rest = action.split(" ", 1)
        row_num, field = rest.split(": ")
        row_num = int(row_num)
        apt = APARTMENTS[row_num - 1]
        selector = f"#r{row_num}_{field}"
        value = str(apt[field])
        page.locator(selector).click()
        page.wait_for_timeout(60)
        page.locator(selector).fill(value)
        detail["selector"] = selector
        detail["value"] = value
        # tiny gap between cells for pacing
        page.wait_for_timeout(PACING["fill_cell"])

    elif action.startswith("click Filter"):
        page.locator("#filterBtn").click()
        detail["clicked"] = "#filterBtn"
        page.wait_for_timeout(PACING["filter_click"])

    elif action == "type To":
        page.locator("#to").fill(EMAIL_TO)
        detail["value"] = EMAIL_TO
        page.wait_for_timeout(PACING["email_type"])

    elif action == "type Subject":
        page.locator("#subject").fill(EMAIL_SUBJECT)
        detail["value"] = EMAIL_SUBJECT
        page.wait_for_timeout(PACING["email_type"])

    elif action == "type Body":
        page.locator("#body").fill(EMAIL_BODY)
        detail["value_len"] = len(EMAIL_BODY)
        page.wait_for_timeout(PACING["email_type"])

    elif action == "click Send":
        page.locator("#sendBtn").click()
        detail["clicked"] = "#sendBtn"
        page.wait_for_timeout(PACING["email_send"])

    else:
        detail["warning"] = f"unknown action: {action}"

    return detail


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    video_dir = OUT_DIR / "_video_raw"
    video_dir.mkdir()

    observer = DeltaVisionObserver()

    ff_total = 0
    dv_total = 0
    per_step = []
    t_start = time.time()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[f"--window-size={VIEWPORT['width']},{VIEWPORT['height']}"],
        )
        context = browser.new_context(
            viewport=VIEWPORT,
            # THIS is the video recording — Playwright writes one WebM per
            # Page in this directory. One Page → one WebM covering the whole session.
            record_video_dir=str(video_dir),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()

        current_url = None
        prev_tab = None

        for step_idx, spec in enumerate(TRAJECTORY):
            step_dir = OUT_DIR / f"step_{step_idx:02d}"
            step_dir.mkdir()

            target_url = f"{MOCK_BASE}/{spec['url']}"
            tab_switched = spec["tab"] != prev_tab

            # Navigate if we changed tabs (or on very first step)
            if target_url != current_url:
                page.goto(target_url, wait_until="networkidle")
                current_url = target_url
                page.wait_for_timeout(PACING["initial_load"])
                # DV should re-anchor on tab switch — CV will likely classify as
                # NEW_PAGE via url_change trigger, which is the correct behavior.
                if tab_switched and step_idx > 0:
                    observer.reset()

            ts_before = time.time() - t_start

            # Execute the action
            detail = perform_action(spec, page)

            ts_after = time.time() - t_start

            # Capture still
            cap_path = step_dir / "capture.png"
            page.screenshot(path=str(cap_path))

            # Costs
            ff_tokens = image_tokens_anthropic(cap_path)
            # observer.observe wants bytes/PIL/base64 — not a Path.
            with open(cap_path, "rb") as f:
                cap_bytes = f.read()
            obs = observer.observe(
                screenshot=cap_bytes,
                url=page.url,
                last_action=spec["action"],
            )
            dv_tokens = obs.estimated_image_tokens()
            crop_bboxes = [list(b) for b in (obs.crop_bboxes or [])]

            # Save DV artifacts
            (step_dir / "action.json").write_text(json.dumps({
                "step": step_idx,
                "phase": spec["phase"],
                "tab": spec["tab"],
                "action": spec["action"],
                "detail": detail,
                "timestamp_before_sec": round(ts_before, 3),
                "timestamp_after_sec": round(ts_after, 3),
            }, indent=2))
            (step_dir / "dv_obs.json").write_text(json.dumps({
                "obs_type": obs.obs_type,
                "trigger": obs.trigger,
                "diff_ratio": obs.diff_ratio,
                "phash_distance": obs.phash_distance,
                "anchor_score": obs.anchor_score,
                "estimated_tokens": dv_tokens,
                "ff_tokens": ff_tokens,
                "crop_bboxes": crop_bboxes,
            }, indent=2))
            if obs.thumbnail is not None:
                obs.thumbnail.save(step_dir / "dv_thumbnail.png")
            for i, crop_img in enumerate(obs.crops or []):
                crop_img.save(step_dir / f"dv_crop_{i:02d}.png")
            if obs.obs_type == "full_frame" and obs.frame is not None:
                obs.frame.save(step_dir / "dv_full_frame.png")

            ff_total += ff_tokens
            dv_total += dv_tokens
            per_step.append({
                "step": step_idx,
                "phase": spec["phase"],
                "tab": spec["tab"],
                "action": spec["action"],
                "ts_before_sec": round(ts_before, 3),
                "ts_after_sec": round(ts_after, 3),
                "ff_tokens": ff_tokens,
                "dv_tokens": dv_tokens,
                "dv_obs_type": obs.obs_type,
                "dv_trigger": obs.trigger,
                "diff_ratio": round(obs.diff_ratio or 0.0, 4),
                "phash_distance": obs.phash_distance,
                "anchor_score": round(obs.anchor_score or 0.0, 4),
                "crop_bboxes": crop_bboxes,
                "ff_running_total": ff_total,
                "dv_running_total": dv_total,
            })
            print(
                f"step {step_idx:02d} t={ts_after:5.1f}s [{spec['tab']:>11}] "
                f"{spec['action']:<30} FF={ff_tokens:>4} DV={dv_tokens:>4} "
                f"obs={obs.obs_type:<10} trig={obs.trigger}"
            )

            prev_tab = spec["tab"]

        # Close context BEFORE browser to flush the WebM to disk
        video_path_raw = None
        try:
            video_path_raw = page.video.path() if page.video else None
        except Exception:
            pass
        context.close()
        browser.close()

    # Playwright wrote the WebM somewhere in video_dir — move to a canonical path
    webms = list(video_dir.glob("*.webm"))
    final_webm = OUT_DIR / "browser.webm"
    if webms:
        shutil.move(str(webms[0]), str(final_webm))
        for leftover in webms[1:]:
            leftover.unlink()
    if video_dir.exists():
        try:
            video_dir.rmdir()
        except OSError:
            pass

    # Convert to MP4 if ffmpeg is available
    final_mp4 = OUT_DIR / "browser.mp4"
    if final_webm.exists() and shutil.which("ffmpeg"):
        subprocess.run([
            "ffmpeg", "-y", "-i", str(final_webm),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(final_mp4),
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_sec = time.time() - t_start
    savings_pct = round((ff_total - dv_total) / ff_total * 100, 1) if ff_total else 0.0

    summary = {
        "task": "Brooklyn apartment deal flow (3 tabs)",
        "n_steps": len(per_step),
        "total_runtime_sec": round(total_sec, 2),
        "ff_total_tokens": ff_total,
        "dv_total_tokens": dv_total,
        "savings_pct": savings_pct,
        "per_step": per_step,
        "apartments": APARTMENTS,
        "email": {"to": EMAIL_TO, "subject": EMAIL_SUBJECT, "body": EMAIL_BODY},
        "video_webm": str(final_webm) if final_webm.exists() else None,
        "video_mp4": str(final_mp4) if final_mp4.exists() else None,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 72)
    print(f"Steps:              {len(per_step)}")
    print(f"Total runtime:      {total_sec:.2f}s")
    print(f"FF total tokens:    {ff_total:>6,}")
    print(f"DV total tokens:    {dv_total:>6,}")
    print(f"Savings:            {savings_pct}%")
    print(f"Video (webm):       {final_webm if final_webm.exists() else '(none)'}")
    print(f"Video (mp4):        {final_mp4 if final_mp4.exists() else '(ffmpeg missing)'}")
    print(f"Metadata:           {OUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
