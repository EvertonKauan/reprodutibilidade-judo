# Methodology

Reproducible summary of the judo sub-technique classification experiment.

## Task

Binary classification **sutemi_waza vs tachi_waza** (where `tachi_waza` = `ashi_waza` +
`te_waza` merged), from short (~4s) video clips.

## Data

- Clips of ~4s, one per technique, with the class in the filename.
- The **paper's** experiment used the full dataset (2,070 clips, multiple sources,
  part private — see paper). This public release ships a **565-clip subset** from 13
  public YouTube sources (Sutemi 128 / Tachi 437).
- The `source_id` (originating source video) is derived from the filename
  (`<class>_<sourceId>_luta<NN>_sub<NN>`), enabling the source-wise split.
- Source videos are referenced by URL in the manifest (`video_sources.csv`, with
  `cut_interval`); clips are **regenerable** via `scripts/02_generate_clips.py`.

## Models

- **R(2+1)D-18** and **MC3-18** (video action recognition architectures, pre-trained on
  Kinetics-400).
- **Input:** RGB-only (image only) or **two-stream** (RGB + Farnebäck optical flow, fused
  via feature concatenation into a single jointly-trained classification head), to
  emphasize motion. The paper compares both; **RGB-only MC3-18 is the released,
  recommended configuration** — see the main [`README.md`](../README.md) for why.

## Preprocessing

- 64 frames per clip, resized to 112×112.
- `full_frame` (entire frame). Optionally, athlete-region cropping via YOLO (not used by
  the released RGB-only model).

## Training (two-phase fine-tuning)

1. **Head** (5 epochs): backbone frozen, `lr = 1e-4`.
2. **Fine-tuning** (40 epochs): backbone unfrozen, `lr = 1e-5`, early layers kept frozen.
- Regularization/optimization: `weight_decay = 0.01`, `dropout = 0.4`,
  `label_smoothing = 0.1`, cosine scheduler with 2 warmup epochs, automatic mixed
  precision (AMP), gradient accumulation (4 steps), class weights in the loss.
- Seed: 42.

## Evaluation — source-wise cross-validation

- **StratifiedGroupKFold** with 10 folds, stratified by class and grouped by
  `source_id`: no source crosses train/validation/test (prevents near-duplicate
  leakage). Ten folds follows Kohavi (1995), who found stratified 10-fold cross-validation
  to offer the best bias/variance trade-off for this kind of accuracy estimation.
- Primary metric: **macro F1**, reported as **mean ± std** across the 10 folds.

## Result (full paper dataset, 2,070 clips)

| Model | Input | Macro F1 (source-wise CV) |
|---|---|---|
| R(2+1)D-18 | RGB | 0.582 ± 0.083 |
| R(2+1)D-18 | RGB + flow | 0.639 ± 0.073 |
| **MC3-18** | **RGB** | **0.787 ± 0.027** |
| MC3-18 | RGB + flow | 0.781 ± 0.038 |

Exact commands in `scripts/03_train_cv_mc3.*`; per-fold hyperparameters in
`results/models/foldN/config.json`.

> **Note on the released public evaluation.** Running the released ensemble on the
> 565-clip public subset (the only data redistributed here) gives Macro F1 = 0.769, not
> the 0.787 ± 0.027 above — see the main README for why the two numbers differ and are
> both expected/correct.
