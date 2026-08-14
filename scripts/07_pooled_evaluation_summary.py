#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes the POOLED evaluation (test-set predictions concatenated across the
10 cross-validation folds) for a given configuration — this is the exact
table reported in the paper (class-wise Precision/Recall/F1, confusion
matrix, accuracy, macro F1, weighted F1).

This is different from scripts/04_cv_summary.py, which reports the MEAN ± STD
of the macro F1 across folds (the primary metric for comparing
configurations, since it does not conflate variance across folds with a
single aggregate). Pooling instead concatenates every fold's test
predictions into one set covering the whole dataset exactly once (folds are
disjoint under --group_split), which is what the paper's class-wise tables
report.

Usage:
    python scripts/07_pooled_evaluation_summary.py --config-dir results/models/mc3_18_rgb
    python scripts/07_pooled_evaluation_summary.py --config-dir results/models/mc3_18_rgb \
        --config-dir results/models/r2plus1d_18_rgb
"""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ["sutemi_waza", "tachi_waza"]


def pooled_summary(config_dir, num_folds):
    dfs = []
    for fold in range(num_folds):
        path = os.path.join(config_dir, f"fold{fold}", "multiclass", "predictions_test.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing predictions_test.csv: {path}")
        dfs.append(pd.read_csv(path))
    pooled = pd.concat(dfs, ignore_index=True)

    y_true = pooled["true_class"]
    y_pred = pooled["predicted_class"]

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="weighted")

    return {
        "description": "Pooled evaluation (test-set predictions concatenated across the "
                        f"{num_folds} CV folds). Each clip in the dataset is evaluated exactly "
                        "once, since --group_split produces disjoint test folds.",
        "total_evaluated": int(len(pooled)),
        "accuracy": acc,
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
        "correct": int(pooled["correct"].sum()),
        "errors": int(len(pooled) - pooled["correct"].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Computes the pooled (10-fold-concatenated) evaluation summary for a configuration.",
    )
    ap.add_argument("--config-dir", action="append", required=True,
                     help="Path to a results/models/<config> folder (repeatable). Must contain "
                          "fold0..foldN-1/multiclass/predictions_test.csv.")
    ap.add_argument("--num-folds", type=int, default=10)
    ap.add_argument("--out-name", default="pooled_evaluation_summary.json",
                     help="Output filename, written inside each --config-dir.")
    args = ap.parse_args()

    for config_dir in args.config_dir:
        summary = pooled_summary(config_dir, args.num_folds)
        out_path = os.path.join(config_dir, args.out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        name = os.path.basename(os.path.normpath(config_dir))
        print(f"{name}: macro_f1={summary['macro_f1']:.4f} "
              f"sutemi_f1={summary['per_class']['sutemi_waza']['f1']:.4f} "
              f"tachi_f1={summary['per_class']['tachi_waza']['f1']:.4f} "
              f"-> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
