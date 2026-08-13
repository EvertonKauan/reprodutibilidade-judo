#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data pipeline step 2: regenerates the clips from the manifest
(data/input_data/video_sources.csv) and the downloaded source videos.

This lets anyone rebuild the public dataset (594 clips) WITHOUT having to
contact the authors: download the videos (scripts/01_download_videos.py)
and run this script, which cuts each clip at the exact interval recorded in
the 'cut_interval' column.

Requires: ffmpeg on PATH.

Usage:
    python scripts/02_generate_clips.py
    python scripts/02_generate_clips.py --copy   # fast cut (no re-encode)
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "input_data", "video_sources.csv")
VIDEOS = os.path.join(ROOT, "data", "input_data", "videos_fonte")
CLIPS = os.path.join(ROOT, "data", "analysis_data", "clips")


def to_seconds(t: str) -> float:
    """'0:04:05.55' -> seconds (float)."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerates the clips from the manifest.")
    ap.add_argument("--copy", action="store_true",
                    help="Fast cut via stream copy (less precise; no re-encode).")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg.")
    args = ap.parse_args()

    os.makedirs(CLIPS, exist_ok=True)
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = missing = errors = 0
    for i, r in enumerate(rows, 1):
        src = os.path.join(VIDEOS, f"{r['id']}.mp4")
        out = os.path.join(CLIPS, r["filename"])
        if not os.path.isfile(src):
            missing += 1
            continue
        if os.path.isfile(out):
            done += 1
            continue
        start, end = [x.strip() for x in r["cut_interval"].split("->")]
        ss, to = to_seconds(start), to_seconds(end)
        if args.copy:
            cmd = [args.ffmpeg, "-y", "-loglevel", "error",
                   "-ss", f"{ss:.3f}", "-to", f"{to:.3f}", "-i", src,
                   "-c", "copy", out]
        else:
            cmd = [args.ffmpeg, "-y", "-loglevel", "error",
                   "-i", src, "-ss", f"{ss:.3f}", "-to", f"{to:.3f}",
                   "-c:v", "libx264", "-c:a", "aac", out]
        try:
            subprocess.run(cmd, check=True)
            done += 1
        except Exception as e:
            print(f"  [error] {r['filename']}: {e}")
            errors += 1
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} processed...", flush=True)

    print(f"\nClips generated/ok: {done} | missing source: {missing} | errors: {errors}")
    if missing:
        print("Tip: run 'python scripts/01_download_videos.py' first to download the source videos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
