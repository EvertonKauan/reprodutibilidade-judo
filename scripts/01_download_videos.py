#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data pipeline step 1: downloads the source videos (YouTube) used to
build the clip dataset.

Reads data/input_data/video_sources.csv (columns: id, channel, url) and downloads
each video at 720p to data/input_data/videos_fonte/<id>.mp4.

Requires: yt-dlp and ffmpeg.
    pip install yt-dlp
    (ffmpeg: install via your system package manager, or use imageio-ffmpeg)

Usage:
    python scripts/01_download_videos.py
    python scripts/01_download_videos.py --quality 480    # smaller
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_SOURCES = os.path.join(ROOT, "data", "input_data", "video_sources.csv")
DEST = os.path.join(ROOT, "data", "input_data", "videos_fonte")


def main() -> int:
    ap = argparse.ArgumentParser(description="Downloads the YouTube source videos (yt-dlp).")
    ap.add_argument("--quality", default="720", help="Maximum video height (e.g., 480, 720, 1080).")
    ap.add_argument("--csv", default=CSV_SOURCES, help="CSV with the source list.")
    ap.add_argument("--dest", default=DEST, help="Output folder.")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    # The manifest has 1 row per clip (repeated URLs); each source is downloaded once.
    with open(args.csv, newline="", encoding="utf-8") as f:
        seen, urls = set(), []
        for row in csv.DictReader(f):
            u = (row.get("url") or "").strip()
            if u and u not in seen:
                seen.add(u); urls.append(u)
    print(f"{len(urls)} video(s) to download at {args.quality}p -> {args.dest}")

    fmt = f"bv*[height<={args.quality}]+ba/b[height<={args.quality}]/b"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", fmt, "--merge-output-format", "mp4",
        "-o", os.path.join(args.dest, "%(id)s.%(ext)s"),
        "--download-archive", os.path.join(args.dest, "_archive.txt"),
        "--ignore-errors", "--no-warnings", "--newline",
        *urls,
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
