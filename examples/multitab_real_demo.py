"""
multitab_real_demo.py — scripted real-site 2-tab workflow.

Research apartments on Google Maps, document them in a real Google Sheet.
Both are live sites — no mocks. Scripted Playwright trajectory for
reproducibility. Zero LLM API cost.

Tab 0: Google Maps search for Brooklyn apartments
Tab 1: Shared Google Sheet (anonymous-edit link)

Trajectory (~22 steps):
  Phase RESEARCH (Maps):
   1. Load Maps + search
   2. Read sidebar results (idle)
   3. Open first listing detail
   4. Scroll + read
   5. Back to results
   6. Open second listing detail
   7. Scroll + read
  Phase DOCUMENT (Sheet):
   8. Switch to Sheet tab
   9-12. Type header row (A1..D1): Apartment, Address, Rating, Note
  13-16. Type row 2: 461 Dean Apartments data
  17-20. Type row 3: The Bay NYC Luxury data
   21. Final hold showing the filled sheet

VIDEO RECORDING FIX: Maps and Sheets run in SEPARATE browser contexts,
each with its own record_video_dir. The two recordings are concatenated
with ffmpeg at the end so the viewer sees Maps → Sheets as one clip.
(Single-context recording always showed Sheets for the full duration,
even during Maps steps — this fixes that.)

Every screenshot runs through DeltaVisionObserver live.

Output: runs_multitab_real/
  browser.webm          (Maps recording + Sheets recording, concatenated)
  metadata.json         -- summary + per-step list
  step_NN/capture.png + dv_obs.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image
from playwright.sync_api import sync_playwright

_here = Path(__file__).resolve().parent
for _cand in (_here, _here.parent, Path("C:/Users/david/Projects/deltavision")):
    if (_cand / "observer.py").exists():
        sys.path.insert(0, str(_cand))
        break

from observer import DeltaVisionObserver  # noqa: E402

VIEWPORT = {"width": 1280, "height": 800}
# Output dir: repo root / runs_multitab_real — works on Mac and Windows
OUT = Path(__file__).resolve().parent.parent / "runs_multitab_real"
MAPS_URL = "https://www.google.com/maps/search/apartments+brooklyn+ny"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1_WQ2e9-7CS6NbFZ-3WsdP5Mrfptb1e4_CWIlVOszKzc/edit?usp=sharing"

# Pacing (ms) — hold after each action so viewer can read
PACE_READ     = 2500  # Maps reading beats (scroll, open listing, read)
PACE_SWITCH   = 1500  # tab/url switch
PACE_TYPE     = 1400  # between cell entries
PACE_FINAL    = 2500

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")

CTX_KWARGS = dict(
    viewport=VIEWPORT,
    user_agent=UA,
    locale="en-US",
    timezone_id="America/New_York",
    record_video_size=VIEWPORT,
)


def toks(p: Path) -> int:
    with Image.open(p) as im:
        w, h = im.size
    return max(75, int(w * h / 750))


def dismiss_consent(page):
    for label in ("Accept all", "Reject all", "I agree"):
        try:
            btn = page.locator(f"button:has-text('{label}')").first
            if btn.is_visible(timeout=1200):
                btn.click()
                page.wait_for_timeout(400)
                return
        except Exception:
            pass


def clear_sheet(page):
    """Best-effort clear — Ctrl+A twice + Delete."""
    try:
        page.keyboard.press("Control+Home"); page.wait_for_timeout(300)
        page.keyboard.press("Control+A"); page.wait_for_timeout(300)
        page.keyboard.press("Control+A"); page.wait_for_timeout(300)
        page.keyboard.press("Delete"); page.wait_for_timeout(800)
        page.keyboard.press("Control+Home"); page.wait_for_timeout(300)
    except Exception as e:
        print(f"(clear_sheet warn: {e})")


def type_cell(page, text: str, advance: str = "Tab"):
    """Type into the focused cell, then advance to the next column/row."""
    page.keyboard.type(text, delay=25)
    page.wait_for_timeout(250)
    page.keyboard.press(advance)
    page.wait_for_timeout(350)


def concat_videos(maps_dir: Path, sheet_dir: Path, out_path: Path) -> bool:
    """Concatenate the Maps webm and Sheets webm into one file using ffmpeg."""
    maps_vids = list(maps_dir.glob("*.webm"))
    sheet_vids = list(sheet_dir.glob("*.webm"))
    if not maps_vids or not sheet_vids:
        print(f"WARN: missing videos — maps={maps_vids} sheet={sheet_vids}")
        return False

    concat_list = out_path.parent / "concat.txt"
    with open(concat_list, "w") as f:
        f.write(f"file '{maps_vids[0]}'\n")
        f.write(f"file '{sheet_vids[0]}'\n")

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c", "copy", str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr[-800:]}")
        return False
    print(f"Concatenated: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vid_maps = OUT / "video_maps"
    vid_sheet = OUT / "video_sheet"
    for d in (vid_maps, vid_sheet):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    steps = []
    obs = DeltaVisionObserver()
    t0 = time.time()

    apt1 = {"name": "461 Dean Apartments",
            "addr": "461 Dean St, Brooklyn",
            "rating": "4.7",
            "note": "Strong reviews, pet-friendly"}
    apt2 = {"name": "The Bay NYC Luxury Rentals",
            "addr": "2971 Shell Rd, Brooklyn",
            "rating": "5.0",
            "note": "Waterfront, near Coney Island"}

    def do_step(idx: int, label: str, phase: str, action: str, pause_ms: int,
                current_page):
        current_page.wait_for_timeout(pause_ms)
        sd = OUT / f"step_{idx:02d}"
        sd.mkdir(exist_ok=True)
        cap = sd / "capture.png"
        current_page.screenshot(path=str(cap))
        r = obs.observe(cap.read_bytes(), url=current_page.url, last_action=action)
        ff = toks(cap)
        dv = r.estimated_image_tokens()
        rec = {
            "step": idx, "label": label, "phase": phase, "action": action,
            "url": current_page.url, "t_rel_s": round(time.time() - t0, 2),
            "obs_type": r.obs_type, "trigger": r.trigger,
            "dv_tokens": dv, "ff_tokens": ff,
            "savings_pct": round(100 * (1 - dv / ff), 1) if ff else 0,
        }
        steps.append(rec)
        (sd / "dv_obs.json").write_text(json.dumps(rec, indent=2))
        print(f"  {idx:02d} | {phase:8s} | {label:35s} | {r.obs_type:10s} | "
              f"DV={dv:>5d} FF={ff:>5d} | {r.trigger}")
        return rec

    print("\n=== MULTITAB REAL DEMO (split-context) ===\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )

        # ── PRE-LOAD: open Sheet in its own context so it's warmed up ──────
        print(">> Pre-loading Sheet in background context...")
        ctx_sheet = browser.new_context(
            **CTX_KWARGS,
            record_video_dir=str(vid_sheet),
        )
        tab_sheet = ctx_sheet.new_page()
        tab_sheet.goto(SHEET_URL, wait_until="domcontentloaded", timeout=30000)
        tab_sheet.wait_for_timeout(3500)
        clear_sheet(tab_sheet)
        # Minimise to background — Maps context will be in front
        tab_sheet.wait_for_timeout(300)

        # ── PHASE 1: RESEARCH — Maps in its own context ───────────────────
        ctx_maps = browser.new_context(
            **CTX_KWARGS,
            record_video_dir=str(vid_maps),
        )
        tab_maps = ctx_maps.new_page()
        tab_maps.bring_to_front()
        tab_maps.goto(MAPS_URL, wait_until="domcontentloaded", timeout=30000)
        tab_maps.wait_for_timeout(4000)
        dismiss_consent(tab_maps)
        tab_maps.wait_for_timeout(800)
        active = tab_maps

        print(">> 1. Load Maps w/ search")
        do_step(1, "Load Maps + search", "research", "navigate to maps", 500, active)

        print(">> 2. Read results (idle)")
        do_step(2, "Read sidebar results", "research", "idle read", PACE_READ, active)

        listing_urls = active.evaluate("""() => {
          const cards = document.querySelectorAll('a.hfpxzc, [role="feed"] a[href*="/maps/place/"]');
          const out = [];
          cards.forEach(a => {
              if (a.href && !out.find(x => x.href === a.href)) {
                  out.push({href: a.href, label: a.getAttribute('aria-label') || ''});
              }
          });
          return out.slice(0, 5);
        }""")
        print(f">> Found {len(listing_urls)} listings")
        for r in listing_urls:
            print(f"     {r['label'][:60]}")
        if len(listing_urls) < 2:
            print("WARN: need >=2 listings")
            ctx_maps.close(); ctx_sheet.close(); browser.close()
            return

        print(">> 3. Open first listing")
        active.goto(listing_urls[0]['href'], wait_until="domcontentloaded", timeout=25000)
        active.wait_for_timeout(2500)
        do_step(3, "Open " + listing_urls[0]['label'][:30], "research", "click listing 1", 300, active)

        print(">> 4. Scroll first listing")
        active.mouse.move(300, 400)
        active.mouse.wheel(0, 500)
        do_step(4, "Scroll first listing", "research", "scroll 500px", PACE_READ, active)

        print(">> 5. Back to results")
        active.goto(MAPS_URL, wait_until="domcontentloaded", timeout=25000)
        active.wait_for_timeout(2200)
        do_step(5, "Back to search results", "research", "back to maps", 300, active)

        listing_urls2 = active.evaluate("""() => {
          const cards = document.querySelectorAll('a.hfpxzc, [role="feed"] a[href*="/maps/place/"]');
          const out = [];
          cards.forEach(a => {
              if (a.href && !out.find(x => x.href === a.href)) {
                  out.push({href: a.href, label: a.getAttribute('aria-label') || ''});
              }
          });
          return out;
        }""")
        target2 = None
        for r in listing_urls2:
            if r['href'] != listing_urls[0]['href']:
                target2 = r
                break
        if not target2:
            target2 = listing_urls[1] if len(listing_urls) > 1 else listing_urls[0]

        print(f">> 6. Open second listing: {target2['label'][:40]}")
        active.goto(target2['href'], wait_until="domcontentloaded", timeout=25000)
        active.wait_for_timeout(2500)
        do_step(6, "Open " + target2['label'][:30], "research", "click listing 2", 300, active)

        print(">> 7. Scroll second listing")
        active.mouse.move(300, 400)
        active.mouse.wheel(0, 500)
        do_step(7, "Scroll second listing", "research", "scroll 500px", PACE_READ, active)

        # Finalize Maps recording
        ctx_maps.close()
        print(">> Maps context closed — video finalised.")

        # ── PHASE 2: DOCUMENT — Sheet context ─────────────────────────────
        print(">> 8. Switch to Sheet tab")
        tab_sheet.bring_to_front()
        active = tab_sheet
        active.wait_for_timeout(1500)
        active.keyboard.press("Control+Home")
        active.wait_for_timeout(500)
        do_step(8, "Switch to Sheet", "document", "switch tab", PACE_SWITCH, active)

        # Header row
        print(">> 9-12. Type header row")
        type_cell(active, "Apartment", "Tab")
        do_step(9, "Header A1: Apartment", "document", "type A1", PACE_TYPE, active)
        type_cell(active, "Address", "Tab")
        do_step(10, "Header B1: Address", "document", "type B1", PACE_TYPE, active)
        type_cell(active, "Rating", "Tab")
        do_step(11, "Header C1: Rating", "document", "type C1", PACE_TYPE, active)
        type_cell(active, "Note", "Enter")
        active.keyboard.press("Home")
        active.wait_for_timeout(200)
        do_step(12, "Header D1: Note", "document", "type D1", PACE_TYPE, active)

        # Row 2: apt 1
        print(">> 13-16. Type row 2 (apt 1)")
        type_cell(active, apt1["name"], "Tab")
        do_step(13, f"Row 2 A: {apt1['name']}", "document", "type A2", PACE_TYPE, active)
        type_cell(active, apt1["addr"], "Tab")
        do_step(14, "Row 2 B: address", "document", "type B2", PACE_TYPE, active)
        type_cell(active, apt1["rating"], "Tab")
        do_step(15, "Row 2 C: rating", "document", "type C2", PACE_TYPE, active)
        type_cell(active, apt1["note"], "Enter")
        active.keyboard.press("Home")
        active.wait_for_timeout(200)
        do_step(16, "Row 2 D: note", "document", "type D2", PACE_TYPE, active)

        # Row 3: apt 2
        print(">> 17-20. Type row 3 (apt 2)")
        type_cell(active, apt2["name"], "Tab")
        do_step(17, f"Row 3 A: {apt2['name']}", "document", "type A3", PACE_TYPE, active)
        type_cell(active, apt2["addr"], "Tab")
        do_step(18, "Row 3 B: address", "document", "type B3", PACE_TYPE, active)
        type_cell(active, apt2["rating"], "Tab")
        do_step(19, "Row 3 C: rating", "document", "type C3", PACE_TYPE, active)
        type_cell(active, apt2["note"], "Enter")
        active.wait_for_timeout(400)
        do_step(20, "Row 3 D: note", "document", "type D3", PACE_TYPE, active)

        # Final beat: show the completed sheet
        print(">> 21. Final hold")
        active.keyboard.press("Control+Home")
        active.wait_for_timeout(800)
        do_step(21, "Final sheet", "document", "idle", PACE_FINAL, active)

        # Save final screenshot before closing
        tab_sheet.screenshot(path=str(OUT / "final_sheet.png"))

        # Summary
        tot_dv = sum(s["dv_tokens"] for s in steps)
        tot_ff = sum(s["ff_tokens"] for s in steps)
        sav = 100 * (1 - tot_dv / tot_ff) if tot_ff else 0
        summary = {
            "task": "Apartment research on Google Maps -> document in shared Google Sheet",
            "sites": ["google.com/maps", "docs.google.com/spreadsheets"],
            "sheet_url": SHEET_URL,
            "viewport": VIEWPORT,
            "n_steps": len(steps),
            "total_dv_tokens": tot_dv,
            "total_ff_tokens": tot_ff,
            "savings_pct": round(sav, 1),
            "n_full_frames": sum(1 for s in steps if s["obs_type"] == "full_frame"),
            "n_deltas": sum(1 for s in steps if s["obs_type"] == "delta"),
            "elapsed_s": round(time.time() - t0, 1),
            "phase_breakdown": {
                "research": sum(1 for s in steps if s["phase"] == "research"),
                "document": sum(1 for s in steps if s["phase"] == "document"),
            },
            "steps": steps,
        }
        print("\n=== SUMMARY ===")
        print(f"DV={tot_dv:,}  FF={tot_ff:,}  SAVINGS={sav:.1f}%  "
              f"full={summary['n_full_frames']} deltas={summary['n_deltas']}  "
              f"elapsed={summary['elapsed_s']}s")
        (OUT / "metadata.json").write_text(json.dumps(summary, indent=2))

        ctx_sheet.close()
        browser.close()

    # ── Concatenate Maps + Sheets recordings ──────────────────────────────
    print("\n>> Concatenating Maps + Sheet recordings...")
    ok = concat_videos(vid_maps, vid_sheet, OUT / "browser.webm")
    if not ok:
        # Fallback: just use Sheets recording if concat fails
        sheet_vids = list(vid_sheet.glob("*.webm"))
        if sheet_vids:
            sheet_vids[0].rename(OUT / "browser.webm")
            print("(fallback: using Sheets recording only)")

    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()
