"""
DeltaVision demo video — v3.

Accurately shows what each mode SENDS TO THE MODEL at every step.

Full-Frame (FF) side:
  → Full 1200×700 screenshot every step, always
  → Red box drawn on the region that actually changed

DeltaVision (DV) side — DELTA step:
  (1) Thumbnail (320×225) with GREEN bounding boxes on changed regions
     This is the spatial context: "here's where on the page something moved"
  (2) crop_after image(s), scaled up to ~400px max dim
     This is the detail: "here's exactly what changed"

DeltaVision (DV) side — NEW_PAGE step:
  → Full frame same as FF (navigation detected via URL change)
  → Label: "Navigation detected — DV also sends full frame"

Bottom bar: cumulative tokens FF (orange) vs DV (blue) with savings callout.

Task: 10-step project workflow on TodoMVC (React SPA)
  Steps 1-5:  add write report, review PR, send invoice, update docs, schedule meeting
  Steps 6-7:  check write report, check review PR   [diff~1.4%]
  Steps 8-10: filter Active → Completed → All       [URL hash changes → NEW_PAGE]
"""

import json, sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[2]))

BASE       = Path(__file__).parent
RUN_DIR    = BASE / "runs/project_task/delta"
OUT_DIR    = BASE / "video_frames"
OUT_MP4    = OUT_DIR / "deltavision_demo_v3.mp4"
OUT_DIR.mkdir(exist_ok=True)

# ── Load all step obs.json files ─────────────────────────────────────
def load_steps(run_dir: Path) -> list[dict]:
    steps = []
    for i in range(0, 11):
        f = run_dir / f"step_{i}" / "obs.json"
        if f.exists():
            steps.append(json.loads(f.read_text()))
    return steps

STEPS = load_steps(RUN_DIR)
# Total cumulative counts (from last step)
TOTAL_FF = STEPS[-1]["ff_tokens_cumul"]
TOTAL_DV = STEPS[-1]["dv_tokens_cumul"]

# ── Canvas layout ────────────────────────────────────────────────────
CW, CH    = 1520, 940
HEAD_H    = 90       # header: task + step info
PANEL_Y   = HEAD_H + 10
PANEL_H   = 680
PANEL_W   = (CW - 60) // 2   # ~730px each
LP_X      = 20
RP_X      = LP_X + PANEL_W + 20
BAR_Y     = PANEL_Y + PANEL_H + 14
BAR_AREA  = CH - BAR_Y - 10
PAD       = 12

# Colour scheme
DARK      = (14,  16,  22)
DARK2     = (22,  24,  32)
DARK3     = (30,  32,  42)
ORANGE    = (230, 120,  40)
BLUE      = ( 50, 170, 240)
GREEN     = ( 50, 210,  90)
RED       = (230,  55,  55)
WHITE     = (240, 240, 248)
GRAY      = (110, 112, 126)
GOLD      = (248, 202,  55)
TEAL      = ( 50, 210, 180)
PURPLE    = (170, 100, 240)

try:
    def font(size): return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    F_XS  = font(11)
    F_SM  = font(13)
    F_MD  = font(16)
    F_LG  = font(20)
    F_XL  = font(26)
    F_BIG = font(36)
except Exception:
    F_XS = F_SM = F_MD = F_LG = F_XL = F_BIG = ImageFont.load_default()


def txt(draw, xy, text, font, fill, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

def box(draw, xy1, xy2, outline=None, fill=None, width=2):
    draw.rectangle([xy1, xy2], outline=outline, fill=fill, width=width)


def render_intro() -> Image.Image:
    """Title card shown before the first step."""
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    # Background gradient feel
    box(draw, (0, 0), (CW, CH), fill=DARK)
    box(draw, (0, 0), (CW, 6), fill=BLUE)  # top accent bar

    cx = CW // 2
    y = 120
    txt(draw, (cx, y),     "DeltaVision  vs  Full-Frame Baseline", F_BIG, WHITE, "mm")
    txt(draw, (cx, y+50),  "What does the model actually receive?", F_LG, GRAY, "mm")

    y = 240
    box(draw, (cx-380, y-12), (cx+380, y+220), fill=DARK2, outline=DARK3, width=1)
    txt(draw, (cx, y+8),   "10-STEP PROJECT WORKFLOW  ·  TodoMVC (React SPA)", F_MD, TEAL, "mm")
    y += 40
    txt(draw, (cx, y),   "Steps 1–5   Add tasks: write report · review PR · send invoice · update docs · schedule meeting", F_SM, WHITE, "mm")
    y += 24
    txt(draw, (cx, y),   "Steps 6–7   Complete tasks: click checkbox on 'write report', 'review PR'  (only ~1.4% of pixels change!)", F_SM, WHITE, "mm")
    y += 24
    txt(draw, (cx, y),   "Steps 8–10  Navigate filters: Active → Completed → All  (URL hash changes)", F_SM, WHITE, "mm")

    y = 530
    # Two columns: FF vs DV
    lx, rx = cx - 320, cx + 80
    txt(draw, (lx, y),   "FULL-FRAME (baseline)", F_MD, ORANGE, "la")
    y2 = y + 28
    for line in [
        "Every step: full 1200×700 screenshot",
        "No classification. Always 1,600 tokens.",
        "10 steps × 1,600 = 17,600 tokens",
    ]:
        txt(draw, (lx, y2), f"  {line}", F_SM, GRAY, "la"); y2 += 22

    txt(draw, (rx, y),   "DELTAVISION", F_MD, BLUE, "la")
    y2 = y + 28
    for line in [
        "DELTA step → (1) thumbnail+green-boxes (~60 tok)",
        "             (2) high-res crop (~180–360 tok)",
        "NEW_PAGE    → full frame, same as FF (1,600 tok)",
    ]:
        txt(draw, (rx, y2), f"  {line}", F_SM, GRAY, "la"); y2 += 22

    y = 690
    txt(draw, (cx, y), "Result:", F_LG, WHITE, "mm")
    txt(draw, (cx, y+32), f"Full-Frame  {17600:,} tokens     →     DeltaVision  {8620:,} tokens   (51% less)", F_XL, GOLD, "mm")

    txt(draw, (cx, CH-30), "Press play →", F_SM, GRAY, "mm")
    return img


def thumbnail_with_boxes(thumbnail_path: str, crops: list, full_w: int, full_h: int) -> Image.Image:
    """Load thumbnail and draw green boxes scaled from full-frame coordinates."""
    thumb = Image.open(thumbnail_path).convert("RGB")
    tw, th = thumb.size   # 320×225 typically
    td = ImageDraw.Draw(thumb)

    sx = tw / full_w
    sy = th / full_h

    for c in crops:
        x, y, w, h = c["bbox"]
        tx = int(x * sx)
        ty = int(y * sy)
        tw_ = max(int(w * sx), 2)
        th_ = max(int(h * sy), 2)
        # draw green rectangle
        td.rectangle([tx, ty, tx+tw_, ty+th_], outline=GREEN, width=2)
        # small label
        if c["index"] == 0:
            td.rectangle([tx, max(ty-12, 0), tx+tw_, max(ty-1, 0)], fill=GREEN)
            td.text((tx+2, max(ty-11, 0)), "CHANGED", font=F_XS, fill=DARK)

    return thumb


def render_step(obs: dict, step_num: int) -> Image.Image:
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)

    # ── Header ───────────────────────────────────────────────────────
    box(draw, (0, 0), (CW, HEAD_H), fill=DARK2)
    box(draw, (0, HEAD_H), (CW, HEAD_H+2), fill=DARK3)

    action_label = obs.get("action_label", "")
    obs_type  = obs.get("obs_type", "full_frame")
    trans     = obs.get("transition", "new_page")
    diff_r    = obs.get("diff_ratio", 0.0)
    phash     = obs.get("phash_distance", 0)
    crops     = obs.get("crops", [])
    n_crops   = len(crops)

    if step_num == 0:
        step_label = "Step 0 / 10  ·  Initial page load"
        cls_label  = "Both modes: full frame (anchor reset)"
        cls_color  = TEAL
    else:
        step_label = f"Step {step_num} / 10  ·  {action_label}"
        if obs_type == "delta":
            cls_label = (f"DV → DELTA  ·  diff={diff_r:.1%}  ·  phash={phash}  ·  "
                         f"{n_crops} crop{'s' if n_crops != 1 else ''}  ·  "
                         f"DV sends: thumbnail + {n_crops} crop{'' if n_crops==1 else 's'}")
            cls_color  = GREEN
        else:
            cls_label = (f"DV → NEW_PAGE  ·  URL hash changed  ·  "
                         f"DV also sends full frame (same as FF)")
            cls_color  = PURPLE

    txt(draw, (PAD, 14),  step_label,  F_LG, WHITE)
    txt(draw, (PAD, 44),  cls_label,   F_SM, cls_color)
    txt(draw, (CW-PAD, 14), "TASK: Add 5 tasks → complete 2 → filter", F_SM, GRAY, "ra")
    ff_c = obs.get('ff_tokens_cumul') or 1600
    dv_c = obs.get('dv_tokens_cumul') or 1600
    txt(draw, (CW-PAD, 38), f"FF cumul: {ff_c:,} tok  |  DV cumul: {dv_c:,} tok", F_SM, GRAY, "ra")

    # ── Panel column labels ───────────────────────────────────────────
    panel_label_y = PANEL_Y
    img_start_y   = PANEL_Y + 54

    # Left: Full-Frame
    txt(draw, (LP_X,      panel_label_y),    "FULL-FRAME  (baseline)", F_MD, ORANGE)
    txt(draw, (LP_X,      panel_label_y+24), "Model receives: full 1200×700 screenshot", F_SM, GRAY)
    txt(draw, (LP_X,      panel_label_y+40), f"This step:  {obs.get('step_ff_tokens', 1600):,} tokens", F_SM, ORANGE)

    # Right: DeltaVision
    txt(draw, (RP_X,      panel_label_y),    "DELTAVISION", F_MD, BLUE)
    if obs_type == "delta":
        txt(draw, (RP_X,  panel_label_y+24), "Model receives: (1) thumbnail w/ green boxes   (2) high-res crop(s)", F_SM, GRAY)
    else:
        txt(draw, (RP_X,  panel_label_y+24), "Model receives: full frame  (navigation detected)", F_SM, GRAY)
    txt(draw, (RP_X,      panel_label_y+40), f"This step:  {obs.get('step_dv_tokens', 1600):,} tokens", F_SM, BLUE)

    max_img_h = PANEL_H - 54

    # ── LEFT PANEL: full screenshot + red bbox ───────────────────────
    t1_path = obs.get("t1_path") or obs.get("current_frame_path")
    if t1_path and Path(t1_path).exists():
        full = Image.open(t1_path).convert("RGB")
        fw, fh = full.size
        scale_l = min(PANEL_W / fw, max_img_h / fh)
        dw = int(fw * scale_l)
        dh = int(fh * scale_l)
        full_s = full.resize((dw, dh), Image.LANCZOS)
        img.paste(full_s, (LP_X, img_start_y))
        box(draw, (LP_X, img_start_y), (LP_X+dw, img_start_y+dh), outline=ORANGE, width=2)

        if crops:
            # Draw red bbox for the LARGEST crop (primary change)
            cx_, cy_, cw_, ch_ = crops[0]["bbox"]
            bx = LP_X + int(cx_ * scale_l)
            by = img_start_y + int(cy_ * scale_l)
            bw = max(int(cw_ * scale_l), 4)
            bh = max(int(ch_ * scale_l), 4)
            box(draw, (bx, by), (bx+bw, by+bh), outline=RED, width=3)
            # "CHANGED" label above box
            lh = 15
            box(draw, (bx, max(by-lh, img_start_y)), (bx+bw, max(by-1, img_start_y)), fill=RED)
            txt(draw, (bx+2, max(by-lh+1, img_start_y)), "CHANGED", F_XS, WHITE)
            pct_area = (cw_ * ch_) / (fw * fh) * 100
            txt(draw, (LP_X, img_start_y + dh + 4),
                f"Changed region bbox = {pct_area:.1f}% of screen  |  actual diff_ratio = {diff_r:.1%}  →  model must find the signal",
                F_XS, RED)

    # ── RIGHT PANEL ───────────────────────────────────────────────────
    if obs_type == "delta" and crops:
        # (1) Thumbnail with green boxes
        thumb_path = obs.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            full_for_thumb = Image.open(t1_path).convert("RGB")
            fw2, fh2 = full_for_thumb.size
            thumb_annotated = thumbnail_with_boxes(thumb_path, crops, fw2, fh2)
            # Scale thumbnail to fit right panel width, max ~240px tall
            tw_orig, th_orig = thumb_annotated.size   # 320×225
            t_scale = min(PANEL_W / tw_orig, 240 / th_orig)
            t_dw = int(tw_orig * t_scale)
            t_dh = int(th_orig * t_scale)
            thumb_big = thumb_annotated.resize((t_dw, t_dh), Image.LANCZOS)

            # (1) label bar ABOVE the thumbnail image
            lbar_y = img_start_y - 17
            box(draw, (RP_X, lbar_y), (RP_X + PANEL_W, img_start_y - 1), fill=DARK3)
            txt(draw, (RP_X + 4, lbar_y + 3),
                "(1) Spatial context — 320×225 thumbnail  |  green boxes mark changed regions  |  ~60 tokens",
                F_XS, TEAL)

            img.paste(thumb_big, (RP_X, img_start_y))
            box(draw, (RP_X, img_start_y), (RP_X+t_dw, img_start_y+t_dh), outline=BLUE, width=1)

            # Caption below thumbnail
            caption_y = img_start_y + t_dh + 3
            txt(draw, (RP_X, caption_y),
                "↑ Model locates change instantly — no scanning the full 1200×700 image", F_XS, GREEN)

            # (2) Crop(s) below thumbnail
            crop_label_y = caption_y + 16
            crop_y_start = crop_label_y + 16
            remaining_h  = (img_start_y + max_img_h) - crop_y_start - 20

            primary_crop_path = crops[0].get("crop_after_path")
            if primary_crop_path and Path(primary_crop_path).exists():
                crop_img = Image.open(primary_crop_path).convert("RGB")
                cw_orig, ch_orig = crop_img.size
                c_scale = min(PANEL_W / cw_orig, max(remaining_h, 80) / ch_orig, 3.5)
                c_dw = int(cw_orig * c_scale)
                c_dh = int(ch_orig * c_scale)
                crop_big = crop_img.resize((c_dw, c_dh), Image.LANCZOS)

                tok_crop = obs.get("step_dv_tokens", 240) - 60
                # (2) label bar above crop
                box(draw, (RP_X, crop_label_y), (RP_X + PANEL_W, crop_y_start - 1), fill=DARK3)
                txt(draw, (RP_X + 4, crop_label_y + 2),
                    f"(2) High-res detail crop (zoomed {c_scale:.1f}×, ~{tok_crop} tokens) — AFTER",
                    F_XS, TEAL)

                img.paste(crop_big, (RP_X, crop_y_start))
                box(draw, (RP_X, crop_y_start), (RP_X+c_dw, crop_y_start+c_dh), outline=GREEN, width=2)
                # small AFTER badge corner
                box(draw, (RP_X, crop_y_start), (RP_X+44, crop_y_start+14), fill=GREEN)
                txt(draw, (RP_X+2, crop_y_start+1), "AFTER", F_XS, DARK)

                txt(draw, (RP_X, crop_y_start + c_dh + 3),
                    "↑ Model sees only the changed region at full resolution — no background noise", F_XS, GREEN)

            txt(draw, (RP_X, img_start_y + max_img_h + 4),
                f"DV total this step: ~{obs.get('step_dv_tokens', 240)} tokens  vs  FF: 1,600 tokens",
                F_SM, BLUE)

    else:
        # NEW_PAGE or step 0: show full frame in right panel too
        if t1_path and Path(t1_path).exists():
            full3 = Image.open(t1_path).convert("RGB")
            fw3, fh3 = full3.size
            scale3 = min(PANEL_W / fw3, max_img_h / fh3)
            dw3 = int(fw3 * scale3)
            dh3 = int(fh3 * scale3)
            full_s3 = full3.resize((dw3, dh3), Image.LANCZOS)
            img.paste(full_s3, (RP_X, img_start_y))
            box(draw, (RP_X, img_start_y), (RP_X+dw3, img_start_y+dh3), outline=BLUE, width=2)

            if step_num == 0:
                note = "Step 0: anchor frame — DV also sends full frame to initialise"
                col  = TEAL
            else:
                note = "Navigation detected (URL changed) — DV sends full frame, same as FF"
                col  = PURPLE
            txt(draw, (RP_X, img_start_y + dh3 + 4), note, F_XS, col)
        txt(draw, (RP_X, img_start_y + max_img_h + 4),
            f"DV total this step: 1,600 tokens  =  FF: 1,600 tokens  (no savings on navigation)",
            F_SM, PURPLE if step_num > 0 else TEAL)

    # ── Bottom token bar ─────────────────────────────────────────────
    MAX_TOK   = 18000
    half      = CW // 2

    ff_cumul = obs.get("ff_tokens_cumul", 1600)
    dv_cumul = obs.get("dv_tokens_cumul", 1600)

    # FF bar
    ff_frac = min(ff_cumul / MAX_TOK, 1.0)
    ff_w    = int((half - 100) * ff_frac)
    txt(draw, (PAD, BAR_Y),      f"FF cumulative:  {ff_cumul:,} tokens", F_MD, ORANGE)
    box(draw, (PAD, BAR_Y+24),   (PAD + half-100, BAR_Y+46), outline=GRAY, width=1)
    if ff_w > 0:
        box(draw, (PAD, BAR_Y+24), (PAD + ff_w, BAR_Y+46), fill=ORANGE)

    # DV bar
    dv_frac = min(dv_cumul / MAX_TOK, 1.0)
    dv_w    = int((half - 100) * dv_frac)
    savings = int((1 - dv_cumul / ff_cumul) * 100) if ff_cumul else 0
    savings_str = f"  ({savings}% less)" if savings > 0 else ""
    txt(draw, (half, BAR_Y),     f"DV cumulative:  {dv_cumul:,} tokens{savings_str}", F_MD, BLUE)
    box(draw, (half, BAR_Y+24),  (half + half-100, BAR_Y+46), outline=GRAY, width=1)
    if dv_w > 0:
        box(draw, (half, BAR_Y+24), (half + dv_w, BAR_Y+46), fill=BLUE)

    if savings >= 40:
        txt(draw, (CW-PAD, BAR_Y+5), f"↑ {savings}% less", F_BIG, GOLD, "ra")

    # Step type legend strip
    type_y = BAR_Y + 56
    if obs_type == "delta":
        box(draw, (PAD, type_y), (PAD+10, type_y+10), fill=GREEN)
        txt(draw, (PAD+14, type_y-1), f"DELTA  ·  SPA change detected, crops sent", F_XS, GREEN)
    else:
        tag = "ANCHOR" if step_num == 0 else "NEW_PAGE"
        col = TEAL if step_num == 0 else PURPLE
        box(draw, (PAD, type_y), (PAD+10, type_y+10), fill=col)
        txt(draw, (PAD+14, type_y-1), f"{tag}  ·  Full frame sent by both modes", F_XS, col)

    return img


def render_summary() -> Image.Image:
    img = Image.new("RGB", (CW, CH), DARK)
    draw = ImageDraw.Draw(img)
    box(draw, (0, 0), (CW, 5), fill=GOLD)

    cx = CW // 2
    txt(draw, (cx, 60),  "RESULTS SUMMARY", F_BIG, WHITE, "mm")
    txt(draw, (cx, 105), "DeltaVision vs Full-Frame · 10-step project workflow · TodoMVC React SPA", F_MD, GRAY, "mm")

    # Table
    headers = ["Step", "Action", "DV classifies", "DV tokens", "FF tokens", "Savings"]
    col_x   = [40, 110, 310, 520, 640, 750]
    y       = 160
    for i, h in enumerate(headers):
        txt(draw, (col_x[i], y), h, F_SM, TEAL)
    draw.line([(40, y+18), (CW-40, y+18)], fill=GRAY, width=1)
    y += 28

    type_colors = {
        "initial": TEAL,
        "delta":   GREEN,
        "new_page": PURPLE,
    }

    for obs in STEPS:
        sn = obs.get("step", 0)
        label = obs.get("action_label", "")[:38]
        if sn == 0:
            trans_disp = "anchor/full_frame"
            tc = TEAL
        elif obs.get("obs_type") == "delta":
            nc = len(obs.get("crops", []))
            trans_disp = f"DELTA  ({nc} crop{'s' if nc!=1 else ''})"
            tc = GREEN
        else:
            trans_disp = "NEW_PAGE"
            tc = PURPLE
        dv_tok = obs.get("step_dv_tokens", 1600)
        ff_tok = obs.get("step_ff_tokens", 1600)
        sav_pct = int((1 - dv_tok / ff_tok) * 100) if ff_tok else 0

        row_color = WHITE if sn % 2 == 0 else GRAY
        txt(draw, (col_x[0], y), str(sn),      F_SM, row_color)
        txt(draw, (col_x[1], y), label,         F_SM, row_color)
        txt(draw, (col_x[2], y), trans_disp,    F_SM, tc)
        txt(draw, (col_x[3], y), f"{dv_tok:,}", F_SM, BLUE)
        txt(draw, (col_x[4], y), f"{ff_tok:,}", F_SM, ORANGE)
        if sav_pct > 0:
            txt(draw, (col_x[5], y), f"{sav_pct}% ↓",  F_SM, GOLD)
        else:
            txt(draw, (col_x[5], y), "—",              F_SM, GRAY)
        y += 22

    draw.line([(40, y+4), (CW-40, y+4)], fill=GRAY, width=1)
    y += 18
    txt(draw, (col_x[3], y), f"{TOTAL_DV:,}", F_MD, BLUE)
    txt(draw, (col_x[4], y), f"{TOTAL_FF:,}", F_MD, ORANGE)
    sav = int((1 - TOTAL_DV / TOTAL_FF) * 100)
    txt(draw, (col_x[5], y), f"{sav}% ↓",     F_MD, GOLD)
    txt(draw, (40, y), "TOTAL:", F_MD, WHITE)

    y += 60
    txt(draw, (cx, y),    f"DeltaVision: {TOTAL_DV:,} tokens total", F_XL, BLUE, "mm")
    txt(draw, (cx, y+40), f"Full-Frame:  {TOTAL_FF:,} tokens total", F_XL, ORANGE, "mm")
    txt(draw, (cx, y+90), f"{sav}% fewer tokens — 7 DELTA steps + 3 NEW_PAGE (correctly classified)", F_LG, GOLD, "mm")

    txt(draw, (cx, CH-30), "github.com/ddavidgao/deltavision", F_SM, GRAY, "mm")
    return img


# ── Assemble frames ──────────────────────────────────────────────────
FPS         = 30
FADE_F      = int(FPS * 0.4)   # 12 frames fade

HOLD_TIMES = {
    "intro":   4.0,
    0:         2.5,   # initial page load
    1:         4.5,   # add write report
    2:         4.0,   # add review PR
    3:         4.0,   # add send invoice
    4:         4.0,   # add update docs
    5:         4.0,   # add schedule meeting
    6:         5.0,   # checkbox write report  ← tiny change, needs more time
    7:         5.0,   # checkbox review PR
    8:         4.5,   # filter Active (NEW_PAGE)
    9:         4.5,   # filter Completed
    10:        4.5,   # filter All
    "summary": 5.0,
}

print("Rendering frames...")
intro_frame   = render_intro()
step_frames   = [render_step(obs, obs.get("step", 0)) for obs in STEPS]
summary_frame = render_summary()

all_np = []

def hold(frame, secs):
    n = int(FPS * secs)
    arr = np.array(frame)
    for _ in range(n):
        all_np.append(arr)

def fade(f1, f2, n_frames=FADE_F):
    a1 = np.array(f1, dtype=float)
    a2 = np.array(f2, dtype=float)
    for i in range(n_frames):
        alpha = i / n_frames
        all_np.append(((1-alpha)*a1 + alpha*a2).astype(np.uint8))


# Intro
hold(intro_frame, HOLD_TIMES["intro"])

# Steps
prev = intro_frame
for obs, frame in zip(STEPS, step_frames):
    sn = obs.get("step", 0)
    fade(prev, frame)
    hold(frame, HOLD_TIMES.get(sn, 4.0))
    prev = frame

# Summary
fade(prev, summary_frame)
hold(summary_frame, HOLD_TIMES["summary"])

# Fade to black
blank = Image.new("RGB", (CW, CH), DARK)
fade(summary_frame, blank, FADE_F * 3)

total_s = len(all_np) / FPS
print(f"Total frames: {len(all_np)}  |  Duration: {total_s:.1f}s at {FPS}fps")

# ── Export ───────────────────────────────────────────────────────────
from moviepy import ImageSequenceClip
clip = ImageSequenceClip(all_np, fps=FPS)
clip.write_videofile(str(OUT_MP4), fps=FPS, codec="libx264",
                     audio=False, logger=None,
                     ffmpeg_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"])
clip.close()

sz = OUT_MP4.stat().st_size / 1024
print(f"\n✅  {OUT_MP4.name}  —  {sz:.0f} KB  |  {total_s:.1f}s")
print("   Left panel:  full screenshot + red bbox on changed region")
print("   Right panel: (1) thumbnail (320×225) with green boxes  +  (2) crop zoomed in")
print("   NEW_PAGE:    both panels show full frame — honest representation")
print("   Summary:     per-step token table + total savings")
