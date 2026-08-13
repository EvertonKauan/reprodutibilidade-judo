# Cross-Validation Result — MC3-18 (RGB-only, 10 folds)

Model: **mc3_18** | Mode: **binary sutemi vs tachi** | Input: **RGB-only** | Split: **StratifiedGroupKFold by source (leakage-free)**

| Fold | Macro F1 | Accuracy | F1 per class |
|------|----------|----------|----------------|
| 0 | 0.7811 | 0.796 | {'sutemi_waza': 0.724, 'tachi_waza': 0.838} |
| 1 | 0.7622 | 0.778 | {'sutemi_waza': 0.701, 'tachi_waza': 0.823} |
| 2 | 0.8217 | 0.841 | {'sutemi_waza': 0.763, 'tachi_waza': 0.881} |
| 3 | 0.7884 | 0.807 | {'sutemi_waza': 0.726, 'tachi_waza': 0.851} |
| 4 | 0.7293 | 0.754 | {'sutemi_waza': 0.648, 'tachi_waza': 0.81} |
| 5 | 0.7692 | 0.802 | {'sutemi_waza': 0.682, 'tachi_waza': 0.856} |
| 6 | 0.8190 | 0.836 | {'sutemi_waza': 0.764, 'tachi_waza': 0.874} |
| 7 | 0.7837 | 0.817 | {'sutemi_waza': 0.698, 'tachi_waza': 0.869} |
| 8 | 0.8113 | 0.825 | {'sutemi_waza': 0.76, 'tachi_waza': 0.863} |
| 9 | 0.8014 | 0.816 | {'sutemi_waza': 0.747, 'tachi_waza': 0.856} |

**Macro F1 (CV) = 0.7867 ± 0.0271**  (min 0.7293 / max 0.8217, n=10 folds)

> The number to report in the paper is the **cross-validation mean**, not the best fold (that would be cherry-picking).
