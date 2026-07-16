# Resultado da Validacao Cruzada — MC3 v2 (40 epocas, two-stream)

Modelo: **mc3_18** | Modo: **binario sutemi vs tachi** | Entrada: **two-stream (RGB + fluxo optico)** | Split: **StratifiedGroupKFold por fonte (sem vazamento)**

| Fold | Macro F1 | Accuracy | F1 por classe |
|------|----------|----------|----------------|
| 0 | 0.6933 | 0.707 | {'sutemi_waza': 0.628, 'tachi_waza': 0.758} |
| 1 | 0.6414 | 0.662 | {'sutemi_waza': 0.556, 'tachi_waza': 0.727} |
| 2 | 0.6906 | 0.717 | {'sutemi_waza': 0.6, 'tachi_waza': 0.781} |
| 3 | 0.6453 | 0.716 | {'sutemi_waza': 0.486, 'tachi_waza': 0.804} |
| 4 | 0.6789 | 0.711 | {'sutemi_waza': 0.578, 'tachi_waza': 0.78} |
| 5 | 0.7149 | 0.724 | {'sutemi_waza': 0.662, 'tachi_waza': 0.767} |
| 6 | 0.6937 | 0.742 | {'sutemi_waza': 0.571, 'tachi_waza': 0.816} |

**Macro F1 (CV) = 0.6797 ± 0.0250**  (min 0.6414 / max 0.7149, n=7 folds)

> O numero a reportar no artigo e a **media da validacao cruzada**, nao o melhor fold (isso seria cherry-picking).
