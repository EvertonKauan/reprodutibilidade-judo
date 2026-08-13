#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarizes the cross-validation (source group-split) of MC3-18 RGB-only, 10 folds.

Reads classification_report.json from each fold in
results/models/foldN/multiclass/ and prints + saves the mean +- std
of the macro F1 (the honest number reported in the paper).

Usage:
    python scripts/04_cv_summary.py
"""
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CV_DIR = os.path.join(ROOT, "results", "models")


def main() -> int:
    rows = []
    for rep in sorted(glob.glob(os.path.join(CV_DIR, "fold*", "multiclass", "classification_report.json"))):
        fold = rep.split(os.sep)[-3].replace("fold", "")
        d = json.load(open(rep, encoding="utf-8"))
        macro = d.get("macro avg", {}).get("f1-score")
        acc = d.get("accuracy")
        per = {k: round(v["f1-score"], 3) for k, v in d.items()
               if isinstance(v, dict) and "f1-score" in v and k not in ("macro avg", "weighted avg")}
        rows.append((fold, macro, acc, per))

    if not rows:
        print("No classification_report.json found in", CV_DIR)
        return 1

    f1s = [r[1] for r in rows]
    mean, std = statistics.mean(f1s), statistics.pstdev(f1s)

    lines = []
    lines.append("# Cross-Validation Result — MC3-18 (RGB-only, 10 folds)\n")
    lines.append("Model: **mc3_18** | Mode: **binary sutemi vs tachi** | "
                 "Input: **RGB-only** | "
                 "Split: **StratifiedGroupKFold by source (leakage-free)**\n")
    lines.append("| Fold | Macro F1 | Accuracy | F1 per class |")
    lines.append("|------|----------|----------|----------------|")
    for fold, macro, acc, per in rows:
        lines.append(f"| {fold} | {macro:.4f} | {acc:.3f} | {per} |")
    lines.append("")
    lines.append(f"**Macro F1 (CV) = {mean:.4f} ± {std:.4f}**  "
                 f"(min {min(f1s):.4f} / max {max(f1s):.4f}, n={len(f1s)} folds)\n")
    lines.append("> The number to report in the paper is the **cross-validation mean**, "
                 "not the best fold (that would be cherry-picking).")

    md = "\n".join(lines)
    print(md)
    out = os.path.join(ROOT, "results", "cv_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\n[ok] saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
