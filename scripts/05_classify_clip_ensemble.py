#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifies an already-cut judo clip (~4s, sutemi_waza vs tachi_waza) using the
10-fold cross-validation ensemble of MC3-18 (RGB-only).

Loads the 10 trained checkpoints (results/models/mc3_18_rgb/foldN/multiclass/best_model.pt),
runs a forward pass of each on the clip, and averages the softmax probabilities
across the 10 folds (bagging-style ensemble) to produce the final prediction and
confidence.

The clip is expected to already be cut to the technique window (e.g., ~4s around
the throw); this script does not detect or cut the clip for you. Frame sampling
(64 frames, uniform over the whole clip) uses the exact same code path as training
(src/main.py's VIDEO_LOADER and val transform), so results match what the
training/evaluation pipeline itself would produce.

Usage:
    python scripts/05_classify_clip_ensemble.py path/to/clip.mp4
    python scripts/05_classify_clip_ensemble.py clip1.mp4 clip2.mp4 --json out.json
    python scripts/05_classify_clip_ensemble.py clip.mp4 --models-dir results/models --num-frames 64
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import main as m  # noqa: E402  (src/main.py — training pipeline, reused for inference)

CLASSES = ["sutemi_waza", "tachi_waza"]


def load_ensemble(models_dir, num_folds, device):
    models = []
    for fold in range(num_folds):
        ckpt = os.path.join(models_dir, f"fold{fold}", "multiclass", "best_model.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                f"Expected layout: {models_dir}/fold0..fold{num_folds - 1}/multiclass/best_model.pt"
            )
        model = m.build_model("mc3_18", num_classes=2, pretrained=False, dropout=0.4)
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        models.append(model)
    return models


def classify_clip(path, models, transform, num_frames, device):
    t0 = time.perf_counter()
    frames = m.VIDEO_LOADER.load(path, num_frames, is_train=False, temporal_jitter=False)
    tensor_frames = [transform(f) for f in frames]
    x = torch.stack(tensor_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1,3,T,H,W]

    probs_sum = torch.zeros(1, len(CLASSES), device=device)
    with torch.no_grad():
        for model in models:
            logits = model(x)
            probs_sum += F.softmax(logits, dim=1)
    probs_avg = (probs_sum / len(models)).squeeze(0).cpu().numpy()
    pred_idx = int(probs_avg.argmax())
    elapsed = time.perf_counter() - t0

    return {
        "clip": path,
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(probs_avg[pred_idx]),
        "probabilities": {c: float(probs_avg[i]) for i, c in enumerate(CLASSES)},
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Classifies pre-cut judo clips with the 10-fold MC3-18 RGB-only ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("clips", nargs="+", help="Path(s) to already-cut clip(s) (e.g., .mp4).")
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "results", "models", "mc3_18_rgb"),
                     help="Directory containing fold0..fold9/multiclass/best_model.pt "
                          "(default: results/models/mc3_18_rgb, the released production config). "
                          "Other configs with released weights: results/models/r2plus1d_18_rgb.")
    ap.add_argument("--num-folds", type=int, default=10, help="Number of ensemble folds (default: 10).")
    ap.add_argument("--num-frames", type=int, default=64, help="Frames sampled per clip (must match training: 64).")
    ap.add_argument("--image-size", type=int, default=112, help="Frame resize (must match training: 112).")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available).")
    ap.add_argument("--json", default=None, help="Optional path to also save the results as JSON.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", file=sys.stderr)

    print(f"Loading {args.num_folds} checkpoints from {args.models_dir} ...", file=sys.stderr)
    models = load_ensemble(args.models_dir, args.num_folds, device)
    transform = m.get_val_transform(args.image_size)

    results = []
    for clip in args.clips:
        r = classify_clip(clip, models, transform, args.num_frames, device)
        results.append(r)
        print(f"{r['clip']} | pred={r['predicted_class']} | confidence={r['confidence']:.4f} | "
              f"p(sutemi)={r['probabilities']['sutemi_waza']:.4f} "
              f"p(tachi)={r['probabilities']['tachi_waza']:.4f} | "
              f"time={r['elapsed_seconds']:.3f}s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[ok] saved to {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
