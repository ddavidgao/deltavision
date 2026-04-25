#!/usr/bin/env python3
"""
Build ffvsdv_comparison_v3.mp4 by combining:
  1. FF section from v1 (0 -> 43.5s) — unchanged (chicago FF baseline)
  2. New 46-step DV spotlight (~50s) from run_14_sf_fixed (40.5% savings)
  3. New outro (10s still image) showing 40.5% / 62,790 / 37,370

Output: benchmarks/ablation/video_frames/ffvsdv_comparison_v3.mp4
"""
import os
import subprocess

BASE = os.path.dirname(__file__)
ORIG = os.path.join(BASE, "video_frames/ffvsdv_comparison_v1.mp4")
SPOTLIGHT = "/tmp/composite_spotlight_v2/composite_spotlight_v2.mp4"
OUTRO_PNG = "/tmp/outro_v3/outro.png"
OUT = os.path.join(BASE, "video_frames/ffvsdv_comparison_v3.mp4")
TMP = "/tmp/hero_v3"

os.makedirs(TMP, exist_ok=True)

# Part A: FF section (0 -> 43.5s from v1)
part_a = os.path.join(TMP, "part_a.mp4")
print("Building part A (FF section, 0 -> 43.5s)...")
subprocess.run([
    "ffmpeg", "-y", "-i", ORIG, "-t", "43.5",
    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=24,setsar=1",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", part_a,
], check=True, capture_output=True)
print(f"  -> {part_a}")

# Part B: DV spotlight (already built, 50s)
print(f"Spotlight: {SPOTLIGHT}")

# Part C: outro still image for 10s
outro_clip = os.path.join(TMP, "outro.mp4")
print("Building outro clip (10s still)...")
subprocess.run([
    "ffmpeg", "-y", "-loop", "1", "-i", OUTRO_PNG, "-t", "10",
    "-vf", "scale=1920:1080,fps=24,setsar=1",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", outro_clip,
], check=True, capture_output=True)
print(f"  -> {outro_clip}")

# Concat all three via filter_complex
print("Concatenating...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", part_a,
    "-i", SPOTLIGHT,
    "-i", outro_clip,
    "-filter_complex",
    # ffmpeg 8+ requires fps as a separate filter step (not chained onto scale's options)
    "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24[v0];"
    "[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24[v1];"
    "[2:v]scale=1920:1080,setsar=1,fps=24[v2];"
    "[v0][v1][v2]concat=n=3:v=1:a=0[vout]",
    "-map", "[vout]",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", OUT,
], check=True)

dur = float(subprocess.check_output(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", OUT]
).strip())
print(f"\nDone: {OUT}  ({dur:.1f}s total)")
