#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passo 1 do pipeline de dados: baixa os videos-fonte (YouTube) usados para
construir o dataset de clipes.

Le data/input_data/fontes_videos.csv (colunas: id, canal, url) e baixa cada
video em 720p para data/input_data/videos_fonte/<id>.mp4.

Requer: yt-dlp e ffmpeg.
    pip install yt-dlp
    (ffmpeg: instale pelo gerenciador do sistema, ou use imageio-ffmpeg)

Uso:
    python scripts/01_baixar_videos.py
    python scripts/01_baixar_videos.py --quality 480    # menor
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_FONTES = os.path.join(ROOT, "data", "input_data", "fontes_videos.csv")
DEST = os.path.join(ROOT, "data", "input_data", "videos_fonte")


def main() -> int:
    ap = argparse.ArgumentParser(description="Baixa os videos-fonte do YouTube (yt-dlp).")
    ap.add_argument("--quality", default="720", help="Altura maxima do video (ex.: 480, 720, 1080).")
    ap.add_argument("--csv", default=CSV_FONTES, help="CSV com as fontes.")
    ap.add_argument("--dest", default=DEST, help="Pasta de saida.")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    with open(args.csv, newline="", encoding="utf-8") as f:
        urls = [row["url"] for row in csv.DictReader(f) if row.get("url")]
    print(f"{len(urls)} video(s) a baixar em {args.quality}p -> {args.dest}")

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
