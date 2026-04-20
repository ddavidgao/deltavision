#!/usr/bin/env bash
# run_demo.sh — SF apartment benchmark demo runner
# Usage: ./benchmarks/mapsheets/run_demo.sh [run_id]
#
# What it does:
#   1. Clears the Google Sheet via AppleScript
#   2. Starts ffmpeg screen recording (Capture screen 0)
#   3. Waits for you to start the Claude agent trial
#   4. When you press Enter, stops recording and saves to results/

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="$REPO/benchmarks/mapsheets/results/demo_$RUN_ID"
mkdir -p "$RESULTS_DIR"

SHEET_URL="https://docs.google.com/spreadsheets/d/1_WQ2e9-7CS6NbFZ-3WsdP5Mrfptb1e4_CWIlVOszKzc/edit"
RECORDING="$RESULTS_DIR/screen_recording.mp4"

echo "=== DeltaVision Maps→Sheets Demo Runner ==="
echo "Run ID: $RUN_ID"
echo "Output: $RESULTS_DIR"
echo ""

# ── Step 1: Clear the sheet ──────────────────────────────────────────────────
echo "[1/3] Clearing Google Sheet..."
osascript <<EOF
tell application "Google Chrome"
    activate
    if (count of windows) = 0 then
        make new window
    end if
    set newTab to make new tab at end of tabs of front window with properties {URL:"$SHEET_URL"}
    delay 6
    tell front window
        set active tab index to (count of tabs)
    end tell
end tell
tell application "System Events"
    tell process "Google Chrome"
        delay 2
        keystroke "a" using {command down}
        delay 0.5
        key code 51
        delay 1
        keystroke "w" using {command down}
        delay 0.3
    end tell
end tell
EOF
echo "   Sheet cleared."
echo ""

# ── Step 2: Start screen recording ───────────────────────────────────────────
echo "[2/3] Starting screen recording → $RECORDING"
echo "      (Recording Capture screen 0 at 1920x1080, 30fps)"

# Get screen resolution
SCREEN_RES=$(system_profiler SPDisplaysDataType 2>/dev/null | grep Resolution | head -1 | awk '{print $2"x"$4}' || echo "2560x1664")

ffmpeg -y \
  -f avfoundation \
  -framerate 30 \
  -video_size "$SCREEN_RES" \
  -i "3:none" \
  -vf "scale=1920:1200:force_original_aspect_ratio=decrease,pad=1920:1200:(ow-iw)/2:(oh-ih)/2" \
  -vcodec libx264 \
  -preset ultrafast \
  -crf 18 \
  -pix_fmt yuv420p \
  "$RECORDING" \
  2>"$RESULTS_DIR/ffmpeg.log" &

FFMPEG_PID=$!
echo "   ffmpeg PID: $FFMPEG_PID"
echo ""

# ── Step 3: Run the trial ─────────────────────────────────────────────────────
echo "[3/3] NOW RUN THE CLAUDE AGENT TRIAL."
echo "      The agent will use mcp__dv-playwright__* tools."
echo "      DV log will appear in: $REPO/dv_runs/"
echo ""
echo "      Press Enter HERE when the trial is COMPLETE to stop recording."
read -r

# Stop recording
kill $FFMPEG_PID 2>/dev/null || true
wait $FFMPEG_PID 2>/dev/null || true
echo ""
echo "Recording saved → $RECORDING"

# ── Collect DV log ────────────────────────────────────────────────────────────
LATEST_LOG=$(ls -t "$REPO/dv_runs"/dv_proxy_run_*.jsonl 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    cp "$LATEST_LOG" "$RESULTS_DIR/dv_proxy.jsonl"
    echo "DV log copied → $RESULTS_DIR/dv_proxy.jsonl"

    # Print summary
    python3 "$REPO/benchmarks/mapsheets/verify_genuine_dv.py" "$RESULTS_DIR/dv_proxy.jsonl"
fi

echo ""
echo "=== Done. Demo artifacts in: $RESULTS_DIR ==="
