#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passo intermediario do pipeline de dados: regenera os clipes a partir do
manifesto (data/input_data/fontes_videos.csv) e dos videos-fonte baixados.

Assim, qualquer pessoa reconstroi o dataset publico (594 clipes) SEM precisar
entrar em contato com os autores: baixa os videos (scripts/01_baixar_videos.py)
e roda este script, que corta cada clipe no intervalo exato registrado na coluna
'momento_corte'.

Requer: ffmpeg no PATH.

Uso:
    python scripts/05_gerar_clipes.py
    python scripts/05_gerar_clipes.py --copy   # corte rapido (sem re-encode)
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "input_data", "fontes_videos.csv")
VIDEOS = os.path.join(ROOT, "data", "input_data", "videos_fonte")
CLIPS = os.path.join(ROOT, "data", "analysis_data", "clips")


def to_seconds(t: str) -> float:
    """'0:04:05.55' -> segundos (float)."""
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenera os clipes a partir do manifesto.")
    ap.add_argument("--copy", action="store_true",
                    help="Corte rapido por copia de stream (menos preciso; sem re-encode).")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Caminho do ffmpeg.")
    args = ap.parse_args()

    os.makedirs(CLIPS, exist_ok=True)
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    feitos = faltando = erros = 0
    for i, r in enumerate(rows, 1):
        src = os.path.join(VIDEOS, f"{r['id']}.mp4")
        out = os.path.join(CLIPS, r["arquivo"])
        if not os.path.isfile(src):
            faltando += 1
            continue
        if os.path.isfile(out):
            feitos += 1
            continue
        start, end = [x.strip() for x in r["momento_corte"].split("->")]
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
            feitos += 1
        except Exception as e:
            print(f"  [erro] {r['arquivo']}: {e}")
            erros += 1
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} processados...", flush=True)

    print(f"\nClipes gerados/ok: {feitos} | fonte ausente: {faltando} | erros: {erros}")
    if faltando:
        print("Dica: rode antes 'python scripts/01_baixar_videos.py' para baixar os videos-fonte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
