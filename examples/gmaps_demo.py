"""
gmaps_demo.py — REAL CRAWL on Google Maps apartment search.

Clean trajectory (~11 steps) with generous pacing for readability (3-4s per
beat, total ~40s). Records browser.webm + metadata.json + per-step captures.
Bot-tolerant site, realistic agent behavior, DV-friendly (sticky map context).

Trajectory:
  1. Load Maps w/ query "apartments brooklyn ny"  (full_frame — initial)
  2. Idle read beat                               (delta)
  3. Click first sidebar result                   (full_frame — URL change)
  4. Read detail (linger)                         (delta)
  5. Scroll detail panel                          (delta)
  6. Back to results                              (full_frame — URL change)
  7. Zoom map in                                  (delta or full_frame)
  8. Click different result                       (full_frame — URL change)
  9. Read detail                                  (delta)
 10. Scroll detail                                (delta)
 11. Final hold                                   (delta)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

_here = Path(__file__).resolve().parent
for _cand in (_here, _here.parent, Path("C:/Users/david/Projects/deltavision")):
    if (_cand / "observer.py").exists():
        sys.path.insert(0, str(_cand))
        break

from observer import DeltaVisionObserver  # noqa: E402

VIEWPORT = {"width": 1280, "height": 800}
OUT = Path("C:/Users/david/Projects/deltavision/runs_gmaps")
URL_SEARCH = "https://www.google.com/maps/search/apartments+brooklyn+ny"

# Pacing (ms) — HOW LONG we hold after each action for the viewer to read
READ_INITIAL = 3500
READ_DETAIL  = 3500
READ_SCROLL  = 2500
READ_PAN     = 2000
READ_FINAL   = 2500


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "video").mkdir(exist_ok=True)

    meta = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            viewport=VIEWPORT,
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
            locale="en-US",
            timezone_id="America/New_York",
            record_video_dir=str(OUT / "video"),
            record_video_size=VIEWPORT,
        )
        page = ctx.new_page()
        obs = DeltaVisionObserver()
        t0 = time.time()

        def step(idx, label, action, pause_ms):
            page.wait_for_timeout(pause_ms)
            sd = OUT / f"step_{idx:02d}"
            sd.mkdir(exist_ok=True)
            cap = sd / "capture.png"
            page.screenshot(path=str(cap))
            r = obs.observe(cap.read_bytes(), url=page.url, last_action=action)
            ff = toks(cap)
            dv = r.estimated_image_tokens()
            rec = {
                "step": idx, "label": label, "action": action,
                "url": page.url, "t_rel_s": round(time.time() - t0, 2),
                "obs_type": r.obs_type, "trigger": r.trigger,
                "dv_tokens": dv, "ff_tokens": ff,
                "savings_pct": round(100 * (1 - dv / ff), 1) if ff else 0,
            }
            meta.append(rec)
            (sd / "dv_obs.json").write_text(json.dumps(rec, indent=2))
            print(f"  {idx:02d} | {label:30s} | {r.obs_type:10s} | "
                  f"DV={dv:>5d} FF={ff:>5d} | {r.trigger}")
            return rec

        print("\n=== GMAPS REAL-CRAWL DEMO ===\n")

        # 1. Load Maps with search
        print(">> 1. Load Google Maps w/ search")
        page.goto(URL_SEARCH, wait_until="domcontentloaded", timeout=30000)
        time.sleep(4)  # let tiles + sidebar render
        dismiss_consent(page)
        time.sleep(1)
        step(1, "Load Maps w/ search", "goto maps search", READ_INITIAL)

        # 2. Idle read
        print(">> 2. Idle read")
        step(2, "Read sidebar results", "idle read", READ_INITIAL)

        # Grab the visible result cards — Google Maps uses <a class='hfpxzc'>
        def get_results():
            return page.evaluate("""() => {
              const cards = document.querySelectorAll('a.hfpxzc, [role="feed"] a[href*="/maps/place/"]');
              const out = [];
              cards.forEach(a => {
                  if (a.href && !out.find(x => x.href === a.href)) {
                      out.push({href: a.href, label: a.getAttribute('aria-label') || ''});
                  }
              });
              return out.slice(0, 5);
            }""")

        results = get_results()
        print(f">> found {len(results)} result cards:")
        for r in results:
            print(f"     {r['label'][:60]}: {r['href'][:80]}")
        if len(results) < 2:
            print("WARNING: <2 results — will still run but narrative weak")

        # 3. Click first result — use direct navigation to its href (more reliable than clicking)
        if results:
            print(f">> 3. Open first result: {results[0]['label'][:40]}")
            page.goto(results[0]['href'], wait_until="domcontentloaded", timeout=25000)
            time.sleep(3)
        step(3, "Open first result", f"click {results[0]['label'][:30] if results else '?'}", READ_DETAIL)

        # 4. Read detail
        print(">> 4. Read detail (linger)")
        step(4, "Read first result", "idle read", READ_DETAIL)

        # 5. Scroll the detail panel
        print(">> 5. Scroll detail")
        # Put mouse over the detail panel, then wheel
        page.mouse.move(300, 400)
        page.mouse.wheel(0, 400)
        step(5, "Scroll first detail", "scroll 400px", READ_SCROLL)

        # 6. Back to results via URL goto (more reliable than browser back)
        print(">> 6. Back to search results")
        page.goto(URL_SEARCH, wait_until="domcontentloaded", timeout=25000)
        time.sleep(3)
        step(6, "Back to search results", "back to list", READ_INITIAL)

        # 7. Zoom the map in (press +)
        print(">> 7. Zoom map in")
        # Click the + button which has aria-label "Zoom in"
        try:
            zoom_in = page.locator("button[aria-label='Zoom in']").first
            zoom_in.click()
            time.sleep(1.5)
        except Exception:
            # fallback: keyboard +
            page.keyboard.press("+")
            time.sleep(1.5)
        step(7, "Zoom map in", "zoom in", READ_PAN)

        # 8. Open second result
        results2 = get_results()
        target = None
        for r in results2:
            if results and r['href'] != results[0]['href']:
                target = r
                break
        if target:
            print(f">> 8. Open second result: {target['label'][:40]}")
            page.goto(target['href'], wait_until="domcontentloaded", timeout=25000)
            time.sleep(3)
        step(8, "Open second result", f"click {target['label'][:30] if target else '?'}", READ_DETAIL)

        # 9. Read detail
        print(">> 9. Read detail")
        step(9, "Read second result", "idle read", READ_DETAIL)

        # 10. Scroll
        print(">> 10. Scroll detail")
        page.mouse.move(300, 400)
        page.mouse.wheel(0, 400)
        step(10, "Scroll second detail", "scroll 400px", READ_SCROLL)

        # 11. Final
        print(">> 11. Final hold")
        step(11, "Final frame", "idle", READ_FINAL)

        tot_dv = sum(s["dv_tokens"] for s in meta)
        tot_ff = sum(s["ff_tokens"] for s in meta)
        sav = 100 * (1 - tot_dv / tot_ff) if tot_ff else 0
        summary = {
            "site": "google.com/maps",
            "url": URL_SEARCH,
            "viewport": VIEWPORT,
            "n_steps": len(meta),
            "total_dv_tokens": tot_dv,
            "total_ff_tokens": tot_ff,
            "savings_pct": round(sav, 1),
            "n_full_frames": sum(1 for s in meta if s["obs_type"] == "full_frame"),
            "n_deltas": sum(1 for s in meta if s["obs_type"] == "delta"),
            "elapsed_s": round(time.time() - t0, 1),
            "steps": meta,
        }
        print(f"\nDV={tot_dv:,}  FF={tot_ff:,}  SAVINGS={sav:.1f}%  "
              f"full={summary['n_full_frames']}  deltas={summary['n_deltas']}  "
              f"elapsed={summary['elapsed_s']}s")
        (OUT / "metadata.json").write_text(json.dumps(summary, indent=2))

        ctx.close()
        browser.close()

    vids = list((OUT / "video").glob("*.webm"))
    if vids:
        t = OUT / "browser.webm"
        if t.exists():
            t.unlink()
        vids[0].rename(t)
        print(f"Video: {t} ({t.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
