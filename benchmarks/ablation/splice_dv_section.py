#!/usr/bin/env python3
"""
Build a curated DV spotlight clip from selected frames, then splice it into
ffvsdv_comparison_v1.mp4, replacing the old DV live-proxy section (~45s-72s).

Run:  python3 splice_dv_section.py
Output: benchmarks/ablation/video_frames/ffvsdv_comparison_v2.mp4
"""
import os
import subprocess

BASE = os.path.dirname(__file__)
FRAMES_DIR = "/tmp/dv_maps_spotlight"
ORIG = os.path.join(BASE, "video_frames/ffvsdv_comparison_v1.mp4")
OUT  = os.path.join(BASE, "video_frames/ffvsdv_comparison_v2.mp4")
TMP  = "/tmp/dv_splice"

os.makedirs(TMP, exist_ok=True)

# Curated frame indices — Maps browsing deltas, then Sheets data entry deltas
# Avoid: sort dialog (89% change = looks like full frame), big paste transitions
# Use hardcoded-region frames (sidebar always left ~30% of screen)
SELECTED = [0, 1, 2, 3, 4, 5]  # all 6 maps spotlight frames
HOLD_S = 1.8
FPS = 24

# Use the pre-built verified spotlight clip directly
spotlight_clip = "/tmp/dv_maps_spotlight/maps_spotlight_cfr.mp4"
print(f"Spotlight clip: {spotlight_clip}")

# --- 2. Cut original into: part_a (0→45s) and part_b (72s→end) ---
part_a = os.path.join(TMP, "part_a.mp4")
part_b = os.path.join(TMP, "part_b.mp4")

subprocess.run([
    "ffmpeg", "-y", "-i", ORIG, "-t", "43.5",
    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", part_a
], check=True, capture_output=True)
print(f"Part A: {part_a}")

subprocess.run([
    "ffmpeg", "-y", "-i", ORIG, "-ss", "71", "-t", "13.5",
    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", part_b
], check=True, capture_output=True)
print(f"Part B: {part_b}")

# --- 3. Join with filter_complex concat (reliable across format differences) ---
subprocess.run([
    "ffmpeg", "-y",
    "-i", part_a,
    "-i", spotlight_clip,
    "-i", part_b,
    "-filter_complex",
    "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=24,setsar=1[v0];"
    "[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=24,setsar=1[v1];"
    "[2:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=24,setsar=1[v2];"
    "[v0][v1][v2]concat=n=3:v=1:a=0[vout]",
    "-map", "[vout]",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", OUT
], check=True, capture_output=True)

dur = float(subprocess.check_output(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", OUT]
).strip())
print(f"\nDone: {OUT}  ({dur:.1f}s total)")
