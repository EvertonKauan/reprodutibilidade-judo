#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resume a validacao cruzada (group-split por fonte) do MC3 v2 40 epocas.

Le os classification_report.json de cada fold em
results/cv_mc3_v2_40ep/foldN/multiclass/ e imprime + salva a media +- desvio
do macro F1 (o numero honesto reportado no artigo).

Uso:
    python scripts/04_resumo_cv.py
"""
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CV_DIR = os.path.join(ROOT, "results", "cv_mc3_v2_40ep")


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
        print("Nenhum classification_report.json encontrado em", CV_DIR)
        return 1

    f1s = [r[1] for r in rows]
    mean, std = statistics.mean(f1s), statistics.pstdev(f1s)

    lines = []
    lines.append("# Resultado da Validacao Cruzada — MC3 v2 (40 epocas, two-stream)\n")
    lines.append("Modelo: **mc3_18** | Modo: **binario sutemi vs tachi** | "
                 "Entrada: **two-stream (RGB + fluxo optico)** | "
                 "Split: **StratifiedGroupKFold por fonte (sem vazamento)**\n")
    lines.append("| Fold | Macro F1 | Accuracy | F1 por classe |")
    lines.append("|------|----------|----------|----------------|")
    for fold, macro, acc, per in rows:
        lines.append(f"| {fold} | {macro:.4f} | {acc:.3f} | {per} |")
    lines.append("")
    lines.append(f"**Macro F1 (CV) = {mean:.4f} ± {std:.4f}**  "
                 f"(min {min(f1s):.4f} / max {max(f1s):.4f}, n={len(f1s)} folds)\n")
    lines.append("> O numero a reportar no artigo e a **media da validacao cruzada**, "
                 "nao o melhor fold (isso seria cherry-picking).")

    md = "\n".join(lines)
    print(md)
    out = os.path.join(ROOT, "results", "cv_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\n[ok] salvo em {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
