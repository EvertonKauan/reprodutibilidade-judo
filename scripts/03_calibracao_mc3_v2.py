"""
Calibracao de limiar (temperature scaling + varredura de threshold) para o
mc3_18 v2 (fusion, 40 epocas), nos 7 folds do group_split.

Mesma metodologia do calibracao_limiar.py (10/06/2026), adaptada para CV de 7
folds em vez de 1 split so. Nao retreina nada: le predictions_val.csv /
predictions_test.csv ja salvos. Regra de ouro: temperatura e limiar ajustados
SOMENTE na validacao de cada fold; o teste e usado uma unica vez, no final.

Tambem aplica a correcao de rotulo de 14/07/2026 (5 removidos, 4 invertidos)
de forma consistente com as demais analises.

Uso: python calibracao_mc3_v2.py
Saidas em: calibracao_mc3_v2/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, classification_report

BASE = "resultado_v2_mc3_experiments/v2_40epocas"
OUT_ROOT = "calibracao_mc3_v2"
EPS = 1e-6

REMOVER = [
    "ma_sutemi_waza_pro_44378_13_hl00_0m08.mp4", "sutemi_waza_2M0AufUQqrY_luta05_sub01.mp4",
    "sutemi_waza_2M0AufUQqrY_luta06_sub02.mp4", "sutemi_waza_2M0AufUQqrY_luta10_sub02.mp4",
    "sutemi_waza_3CMuw3ljBlQ_luta10_sub02.mp4",
]
INVERTER = {
    "ma_sutemi_waza_pro_44277_9_hl00_0m06.mp4": "tachi_waza",
    "sutemi_waza_2M0AufUQqrY_luta06_sub03.mp4": "tachi_waza",
    "sutemi_waza_3CMuw3ljBlQ_luta15_sub02.mp4": "tachi_waza",
    "yoko_sutemi_waza_pro_44438_6_hl00_0m08.mp4": "tachi_waza",
}


def corrigir(df):
    df2 = df[~df["filename"].isin(REMOVER)].copy()
    for fn, nl in INVERTER.items():
        df2.loc[df2["filename"] == fn, "true_class"] = nl
    return df2


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(z, y):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def nll(log_t):
        t = np.exp(log_t)
        p = np.clip(sigmoid(z / t), EPS, 1.0 - EPS)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def run_fold(i):
    val = pd.read_csv(f"{BASE}/res_v2_fusion_mc3v2_fold{i}/multiclass/predictions_val.csv")
    test = pd.read_csv(f"{BASE}/res_v2_fusion_mc3v2_fold{i}/multiclass/predictions_test.csv")
    val = corrigir(val)
    test = corrigir(test)

    y_val = (val["true_class"] == "sutemi_waza").astype(int).values
    y_test = (test["true_class"] == "sutemi_waza").astype(int).values
    z_val = logit(val["prob_sutemi_waza"].values)
    z_test = logit(test["prob_sutemi_waza"].values)

    T = fit_temperature(z_val, y_val)
    p_val_raw, p_test_raw = sigmoid(z_val), sigmoid(z_test)
    p_val_cal, p_test_cal = sigmoid(z_val / T), sigmoid(z_test / T)

    ts = np.round(np.arange(0.05, 0.951, 0.005), 3)
    rows = []
    for t in ts:
        pred = (p_val_cal >= t).astype(int)
        rows.append({"threshold": t, "macro_f1": f1_score(y_val, pred, average="macro", zero_division=0)})
    sweep = pd.DataFrame(rows)
    best_f1_val = sweep["macro_f1"].max()
    plateau = sweep[np.isclose(sweep["macro_f1"], best_f1_val)]
    t_star = float(plateau["threshold"].iloc[len(plateau) // 2])

    def metrics_test(pred_pos):
        yp = np.where(pred_pos == 1, "sutemi_waza", "tachi_waza")
        rep = classification_report(test["true_class"], yp, output_dict=True, zero_division=0)
        return rep

    rep_baseline = metrics_test((p_test_raw >= 0.5).astype(int))
    rep_calibrado = metrics_test((p_test_cal >= t_star).astype(int))

    return {
        "fold": i, "T": T, "t_star": t_star, "val_macro_f1_at_t_star": best_f1_val,
        "baseline_test_macro_f1": rep_baseline["macro avg"]["f1-score"],
        "baseline_test_f1_sutemi": rep_baseline["sutemi_waza"]["f1-score"],
        "baseline_test_recall_sutemi": rep_baseline["sutemi_waza"]["recall"],
        "baseline_test_precision_sutemi": rep_baseline["sutemi_waza"]["precision"],
        "calibrado_test_macro_f1": rep_calibrado["macro avg"]["f1-score"],
        "calibrado_test_f1_sutemi": rep_calibrado["sutemi_waza"]["f1-score"],
        "calibrado_test_recall_sutemi": rep_calibrado["sutemi_waza"]["recall"],
        "calibrado_test_precision_sutemi": rep_calibrado["sutemi_waza"]["precision"],
        "sweep": sweep,
    }


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = [run_fold(i) for i in range(7)]

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "sweep"} for r in results])
    df.to_csv(os.path.join(OUT_ROOT, "resultado_por_fold.csv"), index=False)

    print(df.to_string(index=False))
    print()
    for col in ["baseline_test_macro_f1", "calibrado_test_macro_f1"]:
        print(f"{col}: {df[col].mean():.4f} +- {df[col].std(ddof=1):.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(r["sweep"]["threshold"], r["sweep"]["macro_f1"], alpha=0.5, label=f"fold{r['fold']} (t*={r['t_star']:.2f})")
    ax.axvline(0.5, color="gray", ls=":", label="padrao 0.5")
    ax.set_xlabel("limiar p(sutemi), calibrado")
    ax.set_ylabel("macro F1 (validacao)")
    ax.set_title("Varredura de limiar por fold — mc3_18 v2 (fusion)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "threshold_sweep_folds.png"), dpi=130)

    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump({"per_fold": [{k: v for k, v in r.items() if k != "sweep"} for r in results]}, f, indent=2)

    print("\nSaidas em:", OUT_ROOT)
