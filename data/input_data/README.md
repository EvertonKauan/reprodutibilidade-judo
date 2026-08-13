# input_data — original, untouched data

## `video_sources.csv` (the manifest)

Manifest of the public subset: **one row per clip** (565 total), letting anyone
regenerate the whole public dataset from the public source videos. Columns:

| Column | Description |
|---|---|
| `id` | Source video ID on YouTube |
| `channel` | Source channel (CBJ TV, Ochiru, etc.) |
| `url` | Source video URL |
| `cut_interval` | Clip interval within the source video (`H:MM:SS.ss -> H:MM:SS.ss`) |
| `class` | `sutemi_waza`, `ashi_waza`, or `te_waza` |
| `filename` | Generated clip filename |

The 565 clips come from **13 distinct public source videos** (channels CBJ TV, Ochiru,
Judo Highlights, and Judo Spirit). Binary composition: **Sutemi 128 / Tachi 437**
(ashi 258 + te 179).

> **Note on this manifest's history.** The original public sample released with this
> repository had 594 rows. Since then, the underlying (private+public) dataset used for
> the paper went through a manual label review that corrected 37 of these clips' labels,
> removed 17 as duplicates/low quality, and excluded 7 more from the official
> cross-validation evaluation. This manifest reflects those corrections (565 rows,
> current labels) so that anyone regenerating the dataset gets clips whose labels match
> what the released models were actually trained/evaluated on. See
> `../../results/public_subset_evaluation_summary.json` for the exact reconciliation.

## `videos_fonte/` (the full source videos)

The raw material. Not redistributed here (copyright); download from the manifest's URLs:
```bash
python scripts/01_download_videos.py        # 720p (downloads each source once)
```
Saved as `videos_fonte/<id>.mp4`.

## How the clips are regenerated

With the source videos downloaded, `scripts/02_generate_clips.py` cuts each clip at the
exact `cut_interval`, writing to `../analysis_data/clips/`. This reconstructs the dataset
**without contacting the authors**.

> **Traceability:** the clip filename embeds the source `id`
> (`<class>_<sourceId>_luta<NN>_sub<NN>.mp4`), which also backs the **source-wise** split
> used in training (no source crosses train/test).
