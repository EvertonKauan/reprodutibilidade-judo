# analysis_data — data transformed by the scripts

## `clips/` (the training dataset — 565 public clips)

The **565 public clips**, ~4 seconds each, one per technique, with the **class in the
filename**:
```
ashi_waza_2M0AufUQqrY_luta03_sub02.mp4     -> source: 2M0AufUQqrY
sutemi_waza_3CMuw3ljBlQ_luta07_sub01.mp4   -> source: 3CMuw3ljBlQ
te_waza_JTTP_kAAX7k_luta03_sub03.mp4       -> source: JTTP_kAAX7k
```

These clips are **not** versioned in the repository (source video copyright), but are
**automatically regenerated** from the manifest:

```bash
python scripts/01_download_videos.py   # downloads the 13 source videos (manifest URLs)
python scripts/02_generate_clips.py    # cuts the 565 clips at the manifest intervals
```

- The **class** is inferred from the filename (`sutemi_waza`, `ashi_waza`, `te_waza`).
- The **`source_id`** (originating source video) comes from the filename, enabling the
  **source-wise** split during training (`--group_split`), which prevents near-duplicate
  leakage (clips from the same match never fall into train and test at the same time).

Point training at this folder:
```bash
bash scripts/03_train_cv_mc3.sh data/analysis_data/clips
```
