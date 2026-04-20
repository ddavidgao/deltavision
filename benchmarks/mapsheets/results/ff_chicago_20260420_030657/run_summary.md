# Chicago FF Baseline Run — 2026-04-20

## Setup
- **City:** Chicago, IL (cold start — zero overlap with SF DV run)
- **Tools:** `mcp__playwright__*` only (NO DeltaVision proxy)
- **Task:** Same structure as SF DV run (Maps research → Sheets entry → post-processing)
- **Session:** `b46b2393-923b-4d8e-a1cc-377d04f9ae4d.jsonl`

## Results

| Metric | Value |
|--------|-------|
| Screenshots taken | 26 |
| FF tokens (26 × 1,365) | 35,490 |

## Listings Found
1. **The Grand Central Apartments** — 221 W Harrison St, Chicago IL 60607 — ⭐ 4.7 — $2,659/mo (1BR) — Pet: Yes
2. **The Duncan - West Loop** — 1515 W Monroe St, Chicago IL 60607 — ⭐ 4.6 — ~$1,177/mo (studio) — Pet: Yes
3. **The Lawrence House - Uptown** — 1020 W Lawrence Ave, Chicago IL 60640 — ⭐ 4.5 — $1,375/mo (studio) — Pet: Yes

## Post-Processing
- ✅ Sort by Star Rating descending
- ✅ Bold top-rated row
- ✅ AVERAGE / COUNTIF formulas
- ✅ Conditional formatting rule

## Comparison vs DV Run (SF)

| Run | City | Steps/Screenshots | Tokens | 
|-----|------|-------------------|--------|
| FF baseline (this run) | Chicago | 26 screenshots | 35,490 |
| DV run | SF | 29 steps | 29,465 (DV actual) |
| DV run FF-equivalent | SF | 29 steps | 39,585 |

**DV savings (internal, FF-equiv → DV actual):** 25.6%  
**DV savings vs Chicago FF baseline:** 17.0%  
*(The 17% number is the cross-city comparison — less clean due to different task trajectories)*

## Artifacts
- `screen_recording.mp4` — 80MB, full FF run recording
- `claude_transcript.txt` — final summary
- Session JSONL: `~/.claude/projects/.../b46b2393-923b-4d8e-a1cc-377d04f9ae4d.jsonl`
