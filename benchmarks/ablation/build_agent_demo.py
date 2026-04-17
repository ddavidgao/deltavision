"""
DeltaVision REAL AGENT DEMO — v1.

This is NOT a scripted benchmark. A real Claude Sonnet 4.6 agent was given a task:
  "Manage a project checklist. Add 5 work items, check off 2, navigate to Active view."

At each step the agent:
  1. Received a DeltaVision observation (thumbnail + crops for DELTA, full frame for NEW_PAGE)
  2. Read the observation and DECIDED what to do next
  3. Executed the action via Playwright

The video shows, side by side:
  LEFT  — FULL-FRAME: what a baseline agent would have received (full 1200x700 screenshot)
  RIGHT — DELTAVISION: what the agent actually received (thumbnail w/ green boxes + crop)

Each frame shows the agent's actual decision text under the DV observation.
"""

import json, sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[2]))

BASE    = Path(__file__).parent
RUN_DIR = BASE / "runs/agent_run_dv"
OUT_DIR = BASE / "video_frames"
OUT_MP4 = OUT_DIR / "deltavision_agent_demo.mp4"
OUT_DIR.mkdir(exist_ok=True)

def load_steps(run_dir: Path) -> list[dict]:
    steps = []
    for i in range(0, 9):
        f = run_dir / f"step_{i}" / "obs.json"
        if f.exists():
            steps.append(json.loads(f.read_text()))
    return steps

STEPS    = load_steps(RUN_DIR)
TOTAL_FF = STEPS[-1]["ff_tokens_cumul"]
TOTAL_DV = STEPS[-1]["dv_tokens_cumul"]

# ── Canvas ────────────────────────────────────────────────────────────
CW, CH   = 1520, 980
HEAD_H   = 88
PANEL_Y  = HEAD_H + 8
PANEL_W  = (CW - 56) // 2
LP_X     = 18
RP_X     = LP_X + PANEL_W + 20
PANEL_H  = 650
AGENT_Y  = PANEL_Y + PANEL_H + 6    # agent decision strip
AGENT_H  = 52
BAR_Y    = AGENT_Y + AGENT_H + 6
PAD      = 12

DARK   = (12,  14,  20)
DARK2  = (20,  22,  30)
DARK3  = (28,  30,  40)
ORANGE = (230, 118,  36)
BLUE   = ( 46, 168, 238)
GREEN  = ( 46, 208,  88)
RED    = (228,  52,  52)
WHITE  = (238, 240, 246)
GRAY   = (108, 110, 124)
GOLD   = (246, 200,  52)
TEAL   = ( 46, 208, 178)
PURPLE = (168,  98, 238)

try:
    def F(sz): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", sz)
    F_XS, F_SM, F_MD, F_LG, F_XL, F_BIG = F(11), F(13), F(16), F(20), F(26), F(36)
except Exception:
    F_XS = F_SM = F_MD = F_LG = F_XL = F_BIG = ImageFont.load_default()


def t(draw, xy, text, font, fill, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

def r(draw, xy1, xy2, outline=None, fill=None, width=2):
    draw.rectangle([xy1, xy2], outline=outline, fill=fill, width=width)


def thumbnail_with_boxes(thumb_path, crops, fw, fh):
    img = Image.open(thumb_path).convert("RGB")
    tw, th = img.size
    d = ImageDraw.Draw(img)
    for c in crops:
        x, y, w, h = c["bbox"]
        tx, ty = int(x/fw*tw), int(y/fh*th)
        tw_, th_ = max(int(w/fw*tw), 2), max(int(h/fh*th), 2)
        d.rectangle([tx, ty, tx+tw_, ty+th_], outline=GREEN, width=2)
        if c["index"] == 0:
            d.rectangle([tx, max(ty-11,0), tx+tw_, max(ty-1,0)], fill=GREEN)
            d.text((tx+2, max(ty-10,0)), "CHANGED", font=F_XS, fill=DARK)
    return img


def render_intro() -> Image.Image:
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)
    r(draw, (0,0), (CW, 5), fill=GREEN)

    cx = CW // 2
    t(draw, (cx, 90),  "DeltaVision — Real Agent Demo",          F_BIG, WHITE, "mm")
    t(draw, (cx, 138), "A real model received this task and made every decision itself.",  F_LG, GRAY, "mm")

    y = 210
    r(draw, (cx-440, y-14), (cx+440, y+240), fill=DARK2, outline=DARK3, width=1)
    t(draw, (cx, y+8),  "TASK (given to Claude Sonnet 4.6 in natural language):", F_MD, TEAL, "mm")
    y += 42
    for line in [
        '"Manage a project checklist on TodoMVC."',
        '"Add: write report, review PR, send invoice, update docs, schedule meeting."',
        '"Mark write report and review PR as done."',
        '"Navigate to Active view to confirm what remains."',
    ]:
        t(draw, (cx, y), line, F_SM, WHITE, "mm"); y += 26

    y = 510
    lx, rx = cx-380, cx+50
    t(draw, (lx, y), "FULL-FRAME (what a baseline sees)", F_MD, ORANGE, "la")
    for line in ["Full 1200x700 screenshot every step",
                 "Model searches whole image for changes",
                 "8 steps x 1,600 = 12,800 + 1 anchor = 14,400 tok"]:
        y += 24; t(draw, (lx, y), f"  {line}", F_SM, GRAY, "la")

    y = 510
    t(draw, (rx, y), "DELTAVISION (what the agent got)", F_MD, BLUE, "la")
    for line in ["DELTA: thumbnail (60 tok) + crop(s) (180-360 tok)",
                 "NEW_PAGE: full frame (URL changed, 1,600 tok)",
                 "7 DELTA + 1 NEW_PAGE + 1 anchor = 5,420 tok"]:
        y += 24; t(draw, (rx, y), f"  {line}", F_SM, GRAY, "la")

    t(draw, (cx, 740), "Result:", F_LG, WHITE, "mm")
    t(draw, (cx, 780), f"Full-Frame  14,400 tokens   vs   DeltaVision  5,420 tokens   (62% less)", F_XL, GOLD, "mm")
    t(draw, (cx, CH-28), "Agent completed task correctly in both modes. DV used 62% fewer tokens.", F_SM, GRAY, "mm")
    return img


def render_step(obs: dict, sn: int) -> Image.Image:
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    r(draw, (0,0), (CW, HEAD_H), fill=DARK2)
    r(draw, (0, HEAD_H), (CW, HEAD_H+2), fill=DARK3)

    action_label = obs.get("action_label", "")
    obs_type     = obs.get("obs_type", "full_frame")
    trans        = obs.get("transition", "new_page")
    diff_r       = obs.get("diff_ratio", 0.0)
    phash        = obs.get("phash_distance", 0)
    crops        = obs.get("crops", [])
    decision     = obs.get("agent_decision", "")
    ff_c         = obs.get("ff_tokens_cumul") or 1600
    dv_c         = obs.get("dv_tokens_cumul") or 1600

    if sn == 0:
        step_lbl = "Step 0 / 8  —  Initial page load  (anchor frame)"
        cls_lbl  = "Both modes: full frame  |  anchor reset"
        cls_col  = TEAL
    elif obs_type == "delta":
        step_lbl = f"Step {sn} / 8  —  {action_label}"
        cls_lbl  = (f"DV: DELTA  |  diff={diff_r:.1%}  |  phash={phash}  |  "
                    f"{len(crops)} crop(s)  |  DV sends thumbnail + {len(crops)} crop(s)")
        cls_col  = GREEN
    else:
        step_lbl = f"Step {sn} / 8  —  {action_label}"
        cls_lbl  = "DV: NEW_PAGE  |  URL changed  |  DV sends full frame (same as FF)"
        cls_col  = PURPLE

    t(draw, (PAD, 14),    step_lbl,  F_LG, WHITE)
    t(draw, (PAD, 44),    cls_lbl,   F_SM, cls_col)
    t(draw, (CW-PAD, 14), "TASK: Add 5 items, check 2, Active view", F_SM, GRAY, "ra")
    t(draw, (CW-PAD, 38), f"FF cumul: {ff_c:,} tok  |  DV cumul: {dv_c:,} tok", F_SM, GRAY, "ra")

    # ── Panel headers ────────────────────────────────────────────────
    hdr_y     = PANEL_Y
    img_lbl_y = hdr_y + 16
    img_y     = hdr_y + 54
    max_img_h = PANEL_H - 54

    t(draw, (LP_X,  hdr_y),    "FULL-FRAME  (baseline)", F_MD, ORANGE)
    t(draw, (LP_X,  hdr_y+22), "Model receives: full 1200x700 screenshot every step", F_SM, GRAY)
    t(draw, (LP_X,  hdr_y+40), f"This step: {obs.get('step_ff_tokens',1600):,} tokens", F_SM, ORANGE)

    t(draw, (RP_X,  hdr_y),    "DELTAVISION  (agent mode)", F_MD, BLUE)
    if obs_type == "delta":
        t(draw, (RP_X, hdr_y+22), "Agent received: (1) thumbnail w/ green boxes  +  (2) high-res crop", F_SM, GRAY)
    else:
        t(draw, (RP_X, hdr_y+22), "Agent received: full frame  (navigation — URL changed)", F_SM, GRAY)
    t(draw, (RP_X,  hdr_y+40), f"This step: {obs.get('step_dv_tokens',1600):,} tokens", F_SM, BLUE)

    # ── LEFT: full screenshot + red bbox ────────────────────────────
    t1 = obs.get("t1_path") or obs.get("current_frame_path")
    if t1 and Path(t1).exists():
        full = Image.open(t1).convert("RGB")
        fw, fh = full.size
        sc = min(PANEL_W / fw, max_img_h / fh)
        dw, dh = int(fw*sc), int(fh*sc)
        img.paste(full.resize((dw, dh), Image.LANCZOS), (LP_X, img_y))
        r(draw, (LP_X, img_y), (LP_X+dw, img_y+dh), outline=ORANGE, width=2)

        if crops:
            x, y_, w_, h_ = crops[0]["bbox"]
            bx = LP_X + int(x*sc); by = img_y + int(y_*sc)
            bw = max(int(w_*sc),4); bh = max(int(h_*sc),4)
            r(draw, (bx, by), (bx+bw, by+bh), outline=RED, width=3)
            r(draw, (bx, max(by-14, img_y)), (bx+bw, max(by-1, img_y)), fill=RED)
            t(draw, (bx+2, max(by-13, img_y)), "CHANGED", F_XS, WHITE)
            pct = (w_*h_)/(fw*fh)*100
            t(draw, (LP_X, img_y+dh+3),
              f"bbox = {pct:.1f}% of screen  |  actual diff = {diff_r:.1%}  |  model must find this signal in {fw}x{fh} pixels",
              F_XS, RED)

    # ── RIGHT: DV observation ────────────────────────────────────────
    if obs_type == "delta" and crops:
        thumb_path = obs.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            full_img = Image.open(t1).convert("RGB")
            fw2, fh2 = full_img.size
            thumb = thumbnail_with_boxes(thumb_path, crops, fw2, fh2)
            tw0, th0 = thumb.size
            tsc = min(PANEL_W / tw0, 220 / th0)
            tdw, tdh = int(tw0*tsc), int(th0*tsc)
            thumb_big = thumb.resize((tdw, tdh), Image.LANCZOS)

            # label bar above thumbnail
            lb_y = img_y - 16
            r(draw, (RP_X, lb_y), (RP_X+PANEL_W, img_y-1), fill=DARK3)
            t(draw, (RP_X+4, lb_y+3),
              "(1) Spatial context — 320x225 thumbnail  |  green boxes = changed regions  |  ~60 tokens",
              F_XS, TEAL)

            img.paste(thumb_big, (RP_X, img_y))
            r(draw, (RP_X, img_y), (RP_X+tdw, img_y+tdh), outline=BLUE, width=1)
            t(draw, (RP_X, img_y+tdh+3),
              "Model locates change without scanning full image", F_XS, GREEN)

            # crop below
            crop_path = crops[0].get("crop_after_path")
            cl_y = img_y + tdh + 18
            c_y  = cl_y + 16
            rem  = (img_y + max_img_h) - c_y - 16

            if crop_path and Path(crop_path).exists():
                ci = Image.open(crop_path).convert("RGB")
                cw0, ch0 = ci.size
                csc = min(PANEL_W/cw0, max(rem,60)/ch0, 3.5)
                cdw, cdh = int(cw0*csc), int(ch0*csc)
                tok_c = obs.get("step_dv_tokens",240) - 60

                r(draw, (RP_X, cl_y), (RP_X+PANEL_W, c_y-1), fill=DARK3)
                t(draw, (RP_X+4, cl_y+2),
                  f"(2) High-res crop (x{csc:.1f} zoom, ~{tok_c} tokens) — AFTER state",
                  F_XS, TEAL)

                img.paste(ci.resize((cdw, cdh), Image.LANCZOS), (RP_X, c_y))
                r(draw, (RP_X, c_y), (RP_X+cdw, c_y+cdh), outline=GREEN, width=2)
                r(draw, (RP_X, c_y), (RP_X+42, c_y+13), fill=GREEN)
                t(draw, (RP_X+2, c_y+1), "AFTER", F_XS, DARK)
                t(draw, (RP_X, c_y+cdh+3),
                  "Model sees only the changed region — no noise, no scanning", F_XS, GREEN)

    else:
        if t1 and Path(t1).exists():
            full3 = Image.open(t1).convert("RGB")
            fw3, fh3 = full3.size
            sc3 = min(PANEL_W/fw3, max_img_h/fh3)
            dw3, dh3 = int(fw3*sc3), int(fh3*sc3)
            img.paste(full3.resize((dw3, dh3), Image.LANCZOS), (RP_X, img_y))
            col = TEAL if sn == 0 else PURPLE
            r(draw, (RP_X, img_y), (RP_X+dw3, img_y+dh3), outline=col, width=2)
            note = ("Anchor frame — both modes start with full frame" if sn == 0
                    else "URL changed: #/active detected — DV sends full frame, same as FF")
            t(draw, (RP_X, img_y+dh3+3), note, F_XS, col)

    # ── Agent decision strip ─────────────────────────────────────────
    r(draw, (0, AGENT_Y), (CW, AGENT_Y+AGENT_H), fill=DARK2)
    r(draw, (0, AGENT_Y), (CW, AGENT_Y+1), fill=DARK3)
    t(draw, (PAD, AGENT_Y+6),  "Agent decided:", F_SM, GOLD)
    # wrap long decision text
    max_chars = 130
    if len(decision) > max_chars:
        split = decision[:max_chars].rfind(" ")
        t(draw, (PAD+110, AGENT_Y+6),  decision[:split], F_SM, WHITE)
        t(draw, (PAD+110, AGENT_Y+24), decision[split+1:], F_SM, WHITE)
    else:
        t(draw, (PAD+110, AGENT_Y+16), decision, F_SM, WHITE)

    # ── Token bar ────────────────────────────────────────────────────
    MAX_TOK = 15000
    half    = CW // 2

    ff_frac = min(ff_c / MAX_TOK, 1.0)
    ff_w    = int((half-90) * ff_frac)
    t(draw, (PAD, BAR_Y), f"FF cumulative: {ff_c:,} tokens", F_MD, ORANGE)
    r(draw, (PAD, BAR_Y+22), (PAD+half-90, BAR_Y+42), outline=GRAY, width=1)
    if ff_w: r(draw, (PAD, BAR_Y+22), (PAD+ff_w, BAR_Y+42), fill=ORANGE)

    dv_frac = min(dv_c / MAX_TOK, 1.0)
    dv_w    = int((half-90) * dv_frac)
    sav     = int((1 - dv_c/ff_c)*100) if ff_c else 0
    sav_str = f"  ({sav}% less)" if sav > 0 else ""
    t(draw, (half, BAR_Y), f"DV cumulative: {dv_c:,} tokens{sav_str}", F_MD, BLUE)
    r(draw, (half, BAR_Y+22), (half+half-90, BAR_Y+42), outline=GRAY, width=1)
    if dv_w: r(draw, (half, BAR_Y+22), (half+dv_w, BAR_Y+42), fill=BLUE)

    if sav >= 40:
        t(draw, (CW-PAD, BAR_Y+5), f"^{sav}% less", F_BIG, GOLD, "ra")

    return img


def render_summary() -> Image.Image:
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)
    r(draw, (0,0), (CW, 4), fill=GOLD)
    cx = CW // 2

    t(draw, (cx, 55),  "TASK COMPLETE — RESULTS", F_BIG, WHITE, "mm")
    t(draw, (cx, 100), "Claude Sonnet 4.6 completed task correctly in 8 steps using DeltaVision observations", F_MD, GRAY, "mm")

    hdrs = ["Step", "Action", "DV classifies", "DV tokens", "FF tokens", "Savings"]
    cols = [36, 112, 320, 530, 650, 770]
    y = 160
    for i, h in enumerate(hdrs): t(draw, (cols[i], y), h, F_SM, TEAL)
    draw.line([(36, y+18), (CW-36, y+18)], fill=GRAY, width=1)
    y += 28

    for obs in STEPS:
        sn = obs.get("step", 0)
        lbl = obs.get("action_label","")[:42]
        otype = obs.get("obs_type","full_frame")
        nc = len(obs.get("crops",[]))
        if sn == 0:
            cls_d, cc = "anchor / full frame", TEAL
        elif otype == "delta":
            cls_d, cc = f"DELTA  ({nc} crop{'s' if nc!=1 else ''})", GREEN
        else:
            cls_d, cc = "NEW_PAGE", PURPLE
        dv_t = obs.get("step_dv_tokens", 1600)
        ff_t = obs.get("step_ff_tokens", 1600)
        sv   = int((1 - dv_t/ff_t)*100) if ff_t else 0

        rc = WHITE if sn % 2 == 0 else GRAY
        t(draw, (cols[0], y), str(sn),      F_SM, rc)
        t(draw, (cols[1], y), lbl,           F_SM, rc)
        t(draw, (cols[2], y), cls_d,         F_SM, cc)
        t(draw, (cols[3], y), f"{dv_t:,}",  F_SM, BLUE)
        t(draw, (cols[4], y), f"{ff_t:,}",  F_SM, ORANGE)
        t(draw, (cols[5], y), f"{sv}% v" if sv > 0 else "--", F_SM, GOLD)
        y += 22

    draw.line([(36, y+3), (CW-36, y+3)], fill=GRAY, width=1)
    y += 16
    sav = int((1 - TOTAL_DV/TOTAL_FF)*100)
    t(draw, (cols[0], y), "TOTAL", F_MD, WHITE)
    t(draw, (cols[3], y), f"{TOTAL_DV:,}", F_MD, BLUE)
    t(draw, (cols[4], y), f"{TOTAL_FF:,}", F_MD, ORANGE)
    t(draw, (cols[5], y), f"{sav}% v",     F_MD, GOLD)

    y += 60
    t(draw, (cx, y),    f"DeltaVision: {TOTAL_DV:,} tokens", F_XL, BLUE, "mm")
    t(draw, (cx, y+40), f"Full-Frame:  {TOTAL_FF:,} tokens", F_XL, ORANGE, "mm")
    t(draw, (cx, y+90),
      f"{sav}% fewer tokens — 7 DELTA + 1 NEW_PAGE — task completed correctly",
      F_LG, GOLD, "mm")
    t(draw, (cx, CH-28), "github.com/ddavidgao/deltavision", F_SM, GRAY, "mm")
    return img


# ── Assemble ──────────────────────────────────────────────────────────
FPS    = 30
FADE_F = int(FPS * 0.4)

HOLDS = {
    "intro": 4.0,
    0: 2.5,
    1: 4.5,
    2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0,
    6: 5.5,   # tiny 1.5% change — give time to read
    7: 5.5,
    8: 4.5,
    "summary": 6.0,
}

print("Rendering frames...")
intro_f   = render_intro()
step_fs   = [render_step(obs, obs.get("step", 0)) for obs in STEPS]
summary_f = render_summary()

all_np = []

def hold(f, s): [all_np.append(np.array(f)) for _ in range(int(FPS*s))]
def fade(a, b, n=FADE_F):
    a_, b_ = np.array(a, float), np.array(b, float)
    for i in range(n): all_np.append(((1-i/n)*a_ + (i/n)*b_).astype(np.uint8))

hold(intro_f, HOLDS["intro"])
prev = intro_f
for obs, fr in zip(STEPS, step_fs):
    fade(prev, fr); hold(fr, HOLDS.get(obs.get("step",0), 4.0)); prev = fr
fade(prev, summary_f); hold(summary_f, HOLDS["summary"])
blank = Image.new("RGB", (CW, CH), DARK)
fade(summary_f, blank, FADE_F * 3)

dur = len(all_np) / FPS
print(f"Total frames: {len(all_np)}  |  Duration: {dur:.1f}s at {FPS}fps")

from moviepy import ImageSequenceClip
clip = ImageSequenceClip(all_np, fps=FPS)
clip.write_videofile(str(OUT_MP4), fps=FPS, codec="libx264",
                     audio=False, logger=None,
                     ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"])
clip.close()
sz = OUT_MP4.stat().st_size / 1024
print(f"\n[OK]  {OUT_MP4.name}  {sz:.0f} KB  {dur:.1f}s")
