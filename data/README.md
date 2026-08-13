# Data

This repository ships a **public subset** (565 clips from 13 source videos) of the
study's dataset — the full dataset used in the paper is larger and partly private (see
paper). It is organized as follows:

1. **13 full source videos** (public, from YouTube) — the raw material, cut into
   technique clips.
2. **565 already-cut, labeled clips** — ready to train the model; each traceable to one
   of the 13 sources by its filename.

Follows the reproducible-compendium pattern:

- **`input_data/`** — **original, untouched** data: the source list (`video_sources.csv`)
  and, once downloaded, the **full videos** in `videos_fonte/`.
- **`analysis_data/`** — **transformed** data: the ~4s **clips** cut from the full videos
  (this is the training input). Obtained from the source-video URLs.

> Nothing in `input_data/` is edited. Transformations write their output to
> `analysis_data/` or `results/`.

## Data flow

```
video_sources.csv ──(scripts/01_download_videos.py)──> input_data/videos_fonte/*.mp4
                                                              │
                              (each clip is cut at its manifest interval)
                                                              ▼
                                              analysis_data/clips/*.mp4   (labeled)
                                                              │
                          (main.py: StratifiedGroupKFold by source, during training)
                                                              ▼
                                                  training / cross-validation
```

## Approximate reproduction (not identical to the paper)

Only the **public videos** are released here. Part of the data used in the original
experiment was private and is **not** redistributed. Because of that, evaluating this
public subset with the released ensemble gives **Macro F1 = 0.769** (see
[`../results/public_subset_evaluation_summary.json`](../results/public_subset_evaluation_summary.json)),
not the paper's official **0.787 ± 0.027** (full private+public dataset, 10-fold
cross-validation mean). The trained model weights themselves are the exact ones used for
the paper — see [`../results/models/`](../results/models/) and the main
[`README.md`](../README.md) for the full explanation of why the two numbers differ.
