#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarks 10-fold ensemble inference latency on GPU, reproducing the numbers
reported in the paper's "Inference Cost" section (RQ3).

Loads all 10 checkpoints of a given configuration, runs a forward pass of each
sequentially per "clip" (a single random tensor of the correct shape — this
benchmarks the model's compute cost, not I/O/data-loading), and reports the
wall-clock time of one full ensemble pass (all 10 models), averaged over
several timed runs after a warm-up pass. Uses torch.cuda.synchronize() around
each timed section so CUDA's asynchronous execution does not distort the
measurement.

This measures GPU compute only. It does NOT include optical flow extraction
(Farnebäck, CPU-only in this pipeline — see docs/methodology.md and the paper's
Section III-B for that separate, CPU-side measurement) or video I/O/frame
decoding.

Usage:
    python scripts/06_benchmark_inference.py --config-dir results/models/mc3_18_rgb
    python scripts/06_benchmark_inference.py --config-dir results/models/mc3_18_rgb \
        --config-dir results/models/r2plus1d_18_rgb --runs 20

Requires a CUDA GPU. Reported paper numbers (NVIDIA L4, 10 timed runs after 5
warm-up runs, batch size 1, num_frames=64, image_size=112):
    MC3-18 RGB:               226.06 ms
    MC3-18 two-stream:        452.59 ms
    R(2+1)D-18 RGB:           409.40 ms
    R(2+1)D-18 two-stream:    815.86 ms
Two-stream configs are only benchmarkable here if you also have their
checkpoints (not redistributed in this repository by default — see the main
README) placed under results/models/<config>/foldN/multiclass/best_model.pt.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import main as m  # noqa: E402


def load_ensemble(config_dir, num_folds, device):
    with open(os.path.join(config_dir, "fold0", "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = cfg["model"]
    modality = cfg.get("input_modality", "rgb")
    num_frames = cfg.get("num_frames", 64)
    image_size = cfg.get("image_size", 112)
    n_channels = 6 if modality == "two_stream" else 3

    models = []
    for fold in range(num_folds):
        ckpt = os.path.join(config_dir, f"fold{fold}", "multiclass", "best_model.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                f"This configuration's weights may not be redistributed in this "
                f"repository — see the main README for which configs ship checkpoints."
            )
        model = m.build_model_for_modality(
            type("A", (), {"model": model_name, "input_modality": modality})(),
            num_classes=2, dropout=0.4,
        )
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict)
        models.append(model.to(device).eval())

    return models, model_name, modality, n_channels, num_frames, image_size


def bench_ensemble(config_dir, num_folds, runs, warmup, device):
    name = os.path.basename(os.path.normpath(config_dir))
    models, model_name, modality, n_channels, num_frames, image_size = load_ensemble(
        config_dir, num_folds, device
    )
    x = torch.randn(1, n_channels, num_frames, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            for model in models:
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        run_times = []
        with torch.no_grad():
            for _ in range(runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for model in models:
                    _ = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                run_times.append(time.perf_counter() - t0)

    for model in models:
        del model
    del models
    if device == "cuda":
        torch.cuda.empty_cache()

    mean_ms = statistics.mean(run_times) * 1000
    median_ms = statistics.median(run_times) * 1000
    min_ms = min(run_times) * 1000
    max_ms = max(run_times) * 1000
    print(f"{name} ({model_name}, {modality}, {num_folds}-fold ensemble): "
          f"n={runs} | mean={mean_ms:.2f}ms median={median_ms:.2f}ms "
          f"min={min_ms:.2f}ms max={max_ms:.2f}ms")
    return {"config": name, "model": model_name, "modality": modality, "runs": runs,
            "mean_ms": mean_ms, "median_ms": median_ms, "min_ms": min_ms, "max_ms": max_ms}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmarks 10-fold ensemble GPU inference latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config-dir", action="append", required=True,
                     help="Path to a results/models/<config> folder (repeatable). "
                          "Must contain fold0..foldN-1/multiclass/best_model.pt and "
                          "fold0/config.json.")
    ap.add_argument("--num-folds", type=int, default=10)
    ap.add_argument("--runs", type=int, default=10, help="Timed ensemble passes (default: 10, matches the paper).")
    ap.add_argument("--warmup", type=int, default=5, help="Warm-up passes before timing (default: 5, matches the paper).")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda, required to match the paper's numbers).")
    ap.add_argument("--json", default=None, help="Optional path to save results as JSON.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        print("WARNING: running on CPU. The paper's numbers were measured on an NVIDIA L4 "
              "GPU; CPU timings are not comparable and will be much slower.", file=sys.stderr)
    print(f"device={device}", file=sys.stderr)

    results = [bench_ensemble(cd, args.num_folds, args.runs, args.warmup, device) for cd in args.config_dir]

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[ok] saved to {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
