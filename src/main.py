"""
Judo Technique Video Classification
Classifies judo videos into: sutemi_waza, ashi_waza, te_waza
Using temporal video models: r2plus1d_18, r3d_18, mc3_18
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
try:
    from sklearn.model_selection import StratifiedGroupKFold
    STRATIFIED_GROUP_KFOLD_AVAILABLE = True
except ImportError:  # sklearn < 0.24
    STRATIFIED_GROUP_KFOLD_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision.models.video as video_models
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- Optional imports ---
try:
    from decord import VideoReader, cpu as decord_cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# --- Constants ---
CLASSES = ["sutemi_waza", "ashi_waza", "te_waza"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Binary mode: sutemi_waza vs tachi_waza (ashi_waza + te_waza merged)
TACHI_CLASSES = ["sutemi_waza", "tachi_waza"]
TACHI_CLASS_TO_IDX = {c: i for i, c in enumerate(TACHI_CLASSES)}

# Binary mode: ashi_waza vs te_waza (sutemi_waza dropped entirely)
ASHI_TE_CLASSES = ["ashi_waza", "te_waza"]
ASHI_TE_CLASS_TO_IDX = {c: i for i, c in enumerate(ASHI_TE_CLASSES)}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145, 0.216989]

# Reduce CUDA memory fragmentation — must be set before first CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def free_gpu_memory():
    """Release cached GPU memory so the next run starts clean."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def suggest_memory_settings(device):
    """Log recommended settings based on GPU VRAM."""
    if device.type != "cuda":
        return
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    if total_mb < 2500:     # e.g. MX450, GTX 1650 Ti
        logger.warning(
            f"GPU has only {total_mb:.0f} MiB VRAM. "
            "Default settings (batch=4, frames=32, size=224) WILL cause OOM.\n"
            "  Recommended: --batch_size 2 --num_frames 16 --image_size 112\n"
            "  Conservative: --batch_size 1 --num_frames 8  --image_size 112"
        )
    elif total_mb < 6000:   # e.g. RTX 3050, GTX 1660
        logger.warning(
            f"GPU has {total_mb:.0f} MiB VRAM. "
            "Reduce if you hit OOM: --batch_size 2 --num_frames 16 --image_size 112"
        )

# AMP helpers: prefer torch.amp (PyTorch >= 2.4), fall back to torch.cuda.amp
def _make_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()

def _autocast_ctx(device_type):
    try:
        return torch.amp.autocast(device_type)
    except Exception:
        return torch.cuda.amp.autocast()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================
# Argument Parser
# ==============================================================

def get_args():
    parser = argparse.ArgumentParser(description="Judo technique video classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with video files")
    parser.add_argument("--output_dir", type=str, default="resultado", help="Output directory")
    parser.add_argument("--model", type=str, default="r2plus1d_18",
                        choices=["r2plus1d_18", "r3d_18", "mc3_18"])
    parser.add_argument("--mode", type=str, default="multiclass",
                        choices=["multiclass", "ovr", "ovo"])
    parser.add_argument("--decision_strategy", type=str, default="max_prob",
                        choices=["max_prob", "voting", "hierarchical"])
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs_head", type=int, default=10)
    parser.add_argument("--epochs_finetune", type=int, default=40)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance_strategy", type=str, default="weighted_sampler",
                        choices=["none", "class_weights", "weighted_sampler", "undersampling"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temporal_jitter", type=str, default="true")
    parser.add_argument("--use_amp", type=str, default="true")
    # Training stability / regularization (defaults preserve previous behavior)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * steps)")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="LR schedule per phase: ReduceLROnPlateau (default) or warmup+cosine")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Warmup epochs at phase start (cosine scheduler only)")
    parser.add_argument("--freeze_early", type=str, default="false",
                        help="Keep stem+layer1 frozen during the finetune phase")
    parser.add_argument("--val_smooth_window", type=int, default=1,
                        help="Moving-average window (epochs) of val F1 used for best-model selection")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for the train loss (CE direct; BCE via target smoothing)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout before the final fc layer")
    parser.add_argument("--pos_weight_mode", type=str, default="full",
                        choices=["full", "sqrt", "none"],
                        help="pos_weight in binary losses (OVR/OVO): n_neg/n_pos, sqrt of it, or disabled")
    # YOLO / spatial
    parser.add_argument("--spatial_mode", type=str, default="full_frame",
                        choices=["full_frame", "yolo_union_crop", "yolo_masked", "edge_mask"])
    # Static edge masking (removes scoreboard/sponsor overlays that sit at fixed screen
    # margins, without needing any detector). Fraction of height/width blanked from each
    # edge, applied identically to every frame (train and eval), before resize.
    parser.add_argument("--mask_top", type=float, default=0.0)
    parser.add_argument("--mask_bottom", type=float, default=0.0)
    parser.add_argument("--mask_left", type=float, default=0.0)
    parser.add_argument("--mask_right", type=float, default=0.0)
    parser.add_argument("--yolo_model_path", type=str, default="judo-ai-2c-yolo11-small.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.45)
    parser.add_argument("--yolo_margin", type=float, default=0.30)
    parser.add_argument("--yolo_every_n_frames", type=int, default=4)
    parser.add_argument("--yolo_min_detection_rate", type=float, default=0.30)
    parser.add_argument("--yolo_smoothing", type=str, default="true")
    parser.add_argument("--yolo_smoothing_alpha", type=float, default=0.70)
    parser.add_argument("--yolo_box_mode", type=str, default="per_frame",
                        choices=["per_frame", "static"],
                        help="per_frame: box follows the athletes on every frame (original behavior). "
                             "static: a single box per video (temporal union of the detections) — "
                             "removes the background without injecting false camera motion into the clip.")
    parser.add_argument("--save_cropped_debug", type=str, default="false")
    parser.add_argument("--max_debug_videos", type=int, default=20)
    parser.add_argument("--mask_background_mode", type=str, default="darken",
                        choices=["darken", "blur", "zero"])
    # Run all modes in sequence
    parser.add_argument("--all", action="store_true",
                        help="Run multiclass, ovr, ovo and multiclass+yolo sequentially")
    parser.add_argument("--skip_yolo_if_missing", action="store_true",
                        help="Skip the yolo run if yolo model file is not found (used with --all)")
    # Compare spatial modes
    parser.add_argument("--compare_spatial", action="store_true",
                        help="Run full_frame and yolo_union_crop sequentially with the same splits and save a comparison")
    # Binary tachi_waza mode
    parser.add_argument("--binary_tachi", action="store_true",
                        help="Binary mode: sutemi_waza vs tachi_waza (ashi_waza+te_waza merged). Forces multiclass.")
    # Binary ashi_waza vs te_waza mode (sutemi_waza dropped)
    parser.add_argument("--binary_ashi_te", action="store_true",
                        help="Binary mode: ashi_waza vs te_waza (sutemi_waza dropped entirely). Forces multiclass.")
    # Manual exclusion list (label review)
    parser.add_argument("--exclude_videos", type=str, default=None,
                        help="Path to a text file with one filename per line (with or without "
                             "extension; # for comments). Matched videos are removed BEFORE the "
                             "split and recorded in excluded_videos.csv.")
    # Dataset-level balancing (experimental)
    parser.add_argument("--balance_dataset", action="store_true",
                        help="Experimental: undersample every class to the minority class size BEFORE "
                             "the split (random, seeded). With --binary_tachi balances sutemi vs tachi; "
                             "otherwise balances the 3 original classes. Dropped videos are recorded "
                             "in balance_dropped.csv.")
    # Grouped (leakage-free) split by source video
    parser.add_argument("--group_split", action="store_true",
                        help="Split by SOURCE instead of by video: uses StratifiedGroupKFold grouping "
                             "clips from the same source video (source_id derived from the filename). "
                             "Prevents context leakage (near-duplicates from the same match in train AND "
                             "test). Keeps class stratification and the ~train/val/test fractions. "
                             "Generates source_groups.csv (audit) and verifies that no group crosses splits.")
    parser.add_argument("--group_fold", type=int, default=0,
                        help="Which fold becomes the TEST set in --group_split (0..k-1, k=round(1/test_split)). "
                             "Rotating 0,1,2,... with the same seed = cross-validation by source.")
    # Two-stream / optical flow (item 4)
    parser.add_argument("--input_modality", type=str, default="rgb",
                        choices=["rgb", "flow", "two_stream"],
                        help="rgb (default) | flow (Farneback optical flow) | two_stream (RGB+flow, "
                             "late feature fusion). flow/two_stream only apply with mode=multiclass "
                             "(includes --binary_tachi) and use full_frame (YOLO ignored in v1).")
    parser.add_argument("--flow_cache_dir", type=str, default=None,
                        help="Optical flow cache directory (.npy per video, uint8-quantized). "
                             "Default: <parent of data_dir>/flow_cache_nf<N>_sz<S>.")
    parser.add_argument("--flow_bound", type=float, default=20.0,
                        help="Flow clipping bound in pixels for quantization/normalization (u,v in [-bound,bound]).")
    parser.add_argument("--precompute_flow", action="store_true",
                        help="Only precomputes the flow cache for every video in the split and exits (no training).")
    parser.add_argument("--recursive_scan", action="store_true",
                        help="Scans data_dir recursively (subfolders). Useful for nested datasets "
                             "(e.g., mega_dataset/<sub>/*.mp4) without having to flatten them into a single dir. "
                             "Basenames must be unique (cache/split are keyed by basename); "
                             "duplicates are skipped with a warning.")
    return parser.parse_args()


def parse_bool(s):
    return str(s).lower() in ("true", "1", "yes")


# ==============================================================
# Reproducibility
# ==============================================================

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================
# Dataset Discovery
# ==============================================================

def infer_class(filename):
    name = filename.lower()
    for cls in CLASSES:
        if cls in name:
            return cls
    return None


def extract_source_id(filename):
    """Extracts the SOURCE identifier (source video/match) from the filename.

    Several clips come from the same source video (same match/competition) — they
    are near-duplicates that leak context between splits if they fall into train AND test.
    Grouping by source (StratifiedGroupKFold) prevents this leakage.

    Patterns observed in the dataset (all matched, 0 fallbacks):
      - `pro_37615_1_hl00_0m11`         -> source `pro_37615`  (highlights database)
      - `2M0AufUQqrY_luta03_sub02`      -> source `2m0aufuqqry` (YouTube video id)
      - `9kw7Wa16CCM_luta01`            -> source `9kw7wa16ccm` (no _sub)
      - `KQ42cd2XwEY_luta05_06_sub01`   -> source `kq42cd2xwey`
      - `youtube_2026-06-02_08-34-47`   -> source `youtube_2026-06-02_08-34-47` (single recording)
    Fallback (no pattern matches): the stem itself becomes the group (isolated clip, no leakage).
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    low = stem.lower()
    m = re.search(r'pro_(\d+)', low)
    if m:
        return f"pro_{m.group(1)}"
    m = re.search(r'youtube_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', low)
    if m:
        return f"youtube_{m.group(1)}"
    m = re.search(r'([A-Za-z0-9]{6,})_luta\d+', stem)
    if m:
        return m.group(1).lower()
    return low


def get_video_meta(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps if fps > 0 else None
        cap.release()
        return duration, n_frames
    except Exception:
        return None, None


def scan_dataset(data_dir, recursive=False):
    records = []
    data_dir = Path(data_dir)
    paths = sorted(data_dir.rglob("*")) if recursive else sorted(data_dir.iterdir())
    seen = set()
    for p in paths:
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        cls = infer_class(p.name)
        if cls is None:
            logger.debug(f"Skipping {p.name}: no class pattern found")
            continue
        # cache/split are keyed by basename -> basenames must be unique
        if p.name in seen:
            logger.warning(f"Duplicate basename skipped (scan): {p.name} <- {p}")
            continue
        seen.add(p.name)
        label = CLASS_TO_IDX[cls]
        duration, n_frames = get_video_meta(str(p))
        status = "ok" if duration is not None else "error"
        records.append({
            "video_path": str(p),
            "filename": p.name,
            "class_name": cls,
            "label": label,
            "source_id": extract_source_id(p.name),
            "duration": round(duration, 3) if duration else None,
            "num_frames": n_frames,
            "status": status,
        })
    return records


# ==============================================================
# Split
# ==============================================================

def apply_exclude_list(df, list_path, output_dir):
    """Removes from the dataset the videos listed in a text file (manual label review).
    Matching is done by filename without extension (robust to path/extension).
    Never silent: removed videos go to excluded_videos.csv; names not found raise a warning."""
    if not os.path.exists(list_path):
        logger.error(f"--exclude_videos: file not found: {list_path}")
        sys.exit(1)
    wanted = set()
    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wanted.add(os.path.splitext(os.path.basename(line))[0].lower())
    stems = df["filename"].apply(lambda x: os.path.splitext(str(x))[0].lower())
    mask = stems.isin(wanted)

    excluded = df[mask]
    excluded_path = Path(output_dir) / "excluded_videos.csv"
    excluded.to_csv(excluded_path, index=False)
    logger.info(f"exclude_videos: {int(mask.sum())} of {len(wanted)} listed were removed "
                f"(recorded in {excluded_path})")
    for cls, n in excluded["class_name"].value_counts().items():
        logger.info(f"  excluded: {cls}: {n}")
    missing = wanted - set(stems)
    if missing:
        logger.warning(f"exclude_videos: {len(missing)} names from the list NOT found in the dataset: "
                       f"{sorted(missing)}")
    return df[~mask].reset_index(drop=True)


def balance_dataset_df(df, seed, output_dir):
    """Deterministic undersampling at the dataset level (before the split):
    reduces every class to the minority class size, by seeded random draw
    (neutral: does not pick videos by any criterion other than randomness).
    Dropped videos never disappear silently: they go into balance_dropped.csv."""
    counts = df["class_name"].value_counts()
    n_min = int(counts.min())
    kept_parts, dropped_parts = [], []
    for cls, grp in df.groupby("class_name", sort=False):
        shuffled = grp.sample(frac=1.0, random_state=seed)
        kept_parts.append(shuffled.iloc[:n_min])
        dropped_parts.append(shuffled.iloc[n_min:])
    kept = pd.concat(kept_parts).sort_index().reset_index(drop=True)
    dropped = pd.concat(dropped_parts).sort_index().reset_index(drop=True)

    dropped_path = Path(output_dir) / "balance_dropped.csv"
    dropped.to_csv(dropped_path, index=False)
    for cls in counts.index:
        logger.info(f"  balance_dataset: {cls}: {int(counts[cls])} -> {n_min}")
    logger.info(f"balance_dataset: {len(dropped)} videos dropped (recorded in {dropped_path})")
    return kept


def _finalize_splits(df, idx_val, idx_test, output_dir, classes):
    """Writes the split column (from positional val/test indices; the rest = train),
    saves splits.csv and class_distribution.csv, and returns (train_df, val_df, test_df, dist_df).
    Shared by make_splits (by video) and make_group_splits (by source)."""
    df = df.copy()
    df["split"] = "train"
    df.iloc[idx_val, df.columns.get_loc("split")] = "val"
    df.iloc[idx_test, df.columns.get_loc("split")] = "test"

    cols = ["video_path", "filename", "class_name", "label"]
    if "source_id" in df.columns:
        cols.append("source_id")
    cols.append("split")
    splits_path = Path(output_dir) / "splits.csv"
    df[cols].to_csv(splits_path, index=False)
    logger.info(f"Splits saved to {splits_path}")

    # class distribution
    rows = []
    for cls in classes:
        total = (df["class_name"] == cls).sum()
        tr = ((df["class_name"] == cls) & (df["split"] == "train")).sum()
        va = ((df["class_name"] == cls) & (df["split"] == "val")).sum()
        te = ((df["class_name"] == cls) & (df["split"] == "test")).sum()
        rows.append({"class": cls, "total": total, "train": tr, "val": va, "test": te})
    dist_df = pd.DataFrame(rows)
    dist_path = Path(output_dir) / "class_distribution.csv"
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"Class distribution saved to {dist_path}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, dist_df


def make_splits(df, train_split, val_split, test_split, seed, output_dir, classes=None):
    if classes is None:
        classes = CLASSES
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1"

    labels = df["label"].values
    idx = np.arange(len(df))

    # First split: train vs (val + test)
    val_test_frac = val_split + test_split
    idx_train, idx_valtest, _, labels_valtest = train_test_split(
        idx, labels, test_size=val_test_frac, random_state=seed, stratify=labels
    )
    # Second split: val vs test from valtest
    val_frac_of_valtest = val_split / val_test_frac
    idx_val, idx_test = train_test_split(
        idx_valtest, test_size=(1 - val_frac_of_valtest),
        random_state=seed, stratify=labels_valtest
    )

    return _finalize_splits(df, idx_val, idx_test, output_dir, classes)


def make_group_splits(df, train_split, val_split, test_split, seed, output_dir,
                      classes=None, fold=0):
    """Split by SOURCE (leakage-free) using StratifiedGroupKFold.

    Groups by `source_id` (clips from the same source video stay together), stratifying
    by class. Two nested cuts: (1) separates the TEST set as one fold of k=round(1/test_split);
    (2) from the remainder, separates VAL as one fold of k'=round(1/(val_split/(1-test_split))).
    Deterministic given `seed`. `fold` selects which of the k folds becomes the test set (rotating =
    cross-validation by source). No group crosses splits (verified via assert)."""
    if classes is None:
        classes = CLASSES
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1"
    if not STRATIFIED_GROUP_KFOLD_AVAILABLE:
        logger.error("--group_split requires StratifiedGroupKFold (scikit-learn >= 0.24). "
                     "Please upgrade scikit-learn.")
        sys.exit(1)
    if "source_id" not in df.columns:
        logger.error("--group_split: column 'source_id' missing (scan_dataset did not generate it).")
        sys.exit(1)

    labels = df["label"].values
    groups = df["source_id"].values
    idx = np.arange(len(df))

    n_groups = len(set(groups))
    logger.info(f"group_split: {len(df)} videos em {n_groups} fontes (grupos)")

    # (1) TESTE = 1 fold de k_test
    k_test = max(2, round(1.0 / test_split))
    if fold < 0 or fold >= k_test:
        logger.error(f"--group_fold={fold} fora do intervalo valido [0, {k_test - 1}] "
                     f"(k=round(1/test_split)={k_test})")
        sys.exit(1)
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    folds = list(sgkf.split(idx, labels, groups))
    trainval_pos, test_pos = folds[fold]
    trainval_idx = idx[trainval_pos]

    # (2) VAL = 1 fold de k_val, sobre o restante (train+val)
    val_frac_of_tv = val_split / (1.0 - test_split)
    k_val = max(2, round(1.0 / val_frac_of_tv))
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed)
    tv_labels = labels[trainval_idx]
    tv_groups = groups[trainval_idx]
    inner = list(sgkf2.split(trainval_idx, tv_labels, tv_groups))
    tr_pos, val_pos = inner[0]
    idx_val = trainval_idx[val_pos]
    idx_test = idx[test_pos]

    # --- Verificacao: nenhum grupo pode cruzar splits (o proposito do metodo) ---
    g_val = set(groups[idx_val])
    g_test = set(groups[idx_test])
    idx_train = np.setdiff1d(idx, np.concatenate([idx_val, idx_test]))
    g_train = set(groups[idx_train])
    leaks = (g_train & g_val) | (g_train & g_test) | (g_val & g_test)
    if leaks:
        logger.error(f"group_split: VAZAMENTO detectado — {len(leaks)} fontes em >1 split: "
                     f"{sorted(leaks)[:10]}")
        sys.exit(1)
    logger.info(f"group_split: OK — 0 sources crossing splits "
                f"(train={len(g_train)} val={len(g_val)} test={len(g_test)} sources; "
                f"k_test={k_test} fold={fold})")

    train_df, val_df, test_df, dist_df = _finalize_splits(df, idx_val, idx_test, output_dir, classes)

    # Per-source audit: source, split, n, breakdown by class
    split_col = pd.Series(
        np.where(np.isin(idx, idx_test), "test",
        np.where(np.isin(idx, idx_val), "val", "train")),
        index=df.index)  # positional alignment -> real df labels (robust to index)
    audit = (df.assign(split=split_col)
               .groupby(["source_id", "split", "class_name"]).size()
               .reset_index(name="n")
               .sort_values(["split", "source_id", "class_name"]))
    audit_path = Path(output_dir) / "source_groups.csv"
    audit.to_csv(audit_path, index=False)
    logger.info(f"group_split: per-source audit saved to {audit_path}")

    return train_df, val_df, test_df, dist_df


# ==============================================================
# Video Loading
# ==============================================================

class VideoLoader:
    """Loads frames from a video file using decord or cv2."""

    def load_frames_cv2(self, path, num_frames, is_train, temporal_jitter):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)

        if total < num_frames:
            # Read all frames and repeat
            frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(frames) == 0:
                raise IOError(f"No frames read from: {path}")
            while len(frames) < num_frames:
                frames = frames + frames
            frames = frames[:num_frames]
            return np.array(frames, dtype=np.uint8)

        # Temporal sampling
        if is_train and temporal_jitter:
            max_start = total - num_frames
            start = random.randint(0, max(0, max_start))
            indices = list(range(start, start + num_frames))
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

        frames = []
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                prev_idx = idx
            ret, f = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.array(frames, dtype=np.uint8)

    def load_frames_decord(self, path, num_frames, is_train, temporal_jitter):
        vr = VideoReader(path, ctx=decord_cpu(0))
        total = len(vr)
        if total < num_frames:
            indices = list(range(total))
            while len(indices) < num_frames:
                indices = indices + indices
            indices = indices[:num_frames]
        elif is_train and temporal_jitter:
            max_start = total - num_frames
            start = random.randint(0, max(0, max_start))
            indices = list(range(start, start + num_frames))
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

        frames_batch = vr.get_batch(indices).asnumpy()  # [T, H, W, C] RGB
        return frames_batch.astype(np.uint8)

    def load(self, path, num_frames, is_train=False, temporal_jitter=True):
        if DECORD_AVAILABLE:
            try:
                return self.load_frames_decord(path, num_frames, is_train, temporal_jitter)
            except Exception:
                pass
        return self.load_frames_cv2(path, num_frames, is_train, temporal_jitter)


VIDEO_LOADER = VideoLoader()


# ==============================================================
# YOLO Processor
# ==============================================================

class YOLOProcessor:
    def __init__(self, model_path, conf, iou, margin, every_n_frames,
                 smoothing, smoothing_alpha, mask_mode, spatial_mode,
                 box_mode="per_frame"):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.margin = margin
        self.every_n_frames = every_n_frames
        self.smoothing = smoothing
        self.smoothing_alpha = smoothing_alpha
        self.mask_mode = mask_mode
        self.spatial_mode = spatial_mode
        self.box_mode = box_mode
        self.model = None

    def load_model(self):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is not installed. pip install ultralytics")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.model = UltralyticsYOLO(self.model_path)
        logger.info(f"YOLO model loaded from {self.model_path}")

    def detect_athletes(self, frame_bgr):
        """Returns list of [x1, y1, x2, y2, conf] for detected athletes (class 'athlete' only)."""
        if self.model is None:
            return []
        try:
            results = self.model(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)
            # Resolve athlete class id from model names (fallback to 1 if not found)
            names = self.model.names  # {0: 'referee', 1: 'athlete'} or similar
            athlete_ids = {k for k, v in names.items() if v.lower() == "athlete"}
            detections = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                    for b, c, cls_id in zip(boxes, confs, cls_ids):
                        if cls_id in athlete_ids:
                            detections.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)])
            return detections
        except Exception as e:
            logger.debug(f"YOLO detection error: {e}")
            return []

    def smooth_box(self, current_box, previous_box):
        if previous_box is None:
            return current_box
        alpha = self.smoothing_alpha
        return [
            alpha * previous_box[i] + (1 - alpha) * current_box[i]
            for i in range(4)
        ]

    def compute_union_box(self, detections, frame_h, frame_w):
        """Compute union bounding box from top-2 detections by confidence, with margin."""
        if not detections:
            return None
        # Sort by confidence descending, take top 2
        dets = sorted(detections, key=lambda d: d[4], reverse=True)[:2]
        x1 = min(d[0] for d in dets)
        y1 = min(d[1] for d in dets)
        x2 = max(d[2] for d in dets)
        y2 = max(d[3] for d in dets)
        # Apply margin
        bw = x2 - x1
        bh = y2 - y1
        mx = bw * self.margin
        my = bh * self.margin
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(frame_w, x2 + mx)
        y2 = min(frame_h, y2 + my)
        return [x1, y1, x2, y2]

    def apply_crop(self, frame_rgb, box):
        """Crop frame to box [x1,y1,x2,y2]."""
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return frame_rgb[y1:y2, x1:x2]

    def apply_mask(self, frame_rgb, box):
        """Apply background mask outside box."""
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        result = frame_rgb.copy()
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True

        if self.mask_mode == "zero":
            result[~mask] = 0
        elif self.mask_mode == "darken":
            result[~mask] = (result[~mask] * 0.2).astype(np.uint8)
        elif self.mask_mode == "blur":
            blurred = cv2.GaussianBlur(frame_rgb, (31, 31), 0)
            result[~mask] = blurred[~mask]
        return result

    def process_video_frames(self, frames_rgb, video_path):
        """
        Process frames using YOLO.
        Returns (processed_frames, log_dict).
        """
        n = len(frames_rgb)
        h, w = frames_rgb[0].shape[:2]

        log = {
            "video_path": video_path,
            "filename": os.path.basename(video_path),
            "spatial_mode": self.spatial_mode,
            "used_yolo": True,
            "used_yolo_crop": self.spatial_mode == "yolo_union_crop",
            "used_yolo_mask": self.spatial_mode == "yolo_masked",
            "num_frames_total": n,
            "num_frames_processed_by_yolo": 0,
            "num_frames_with_detection": 0,
            "detection_rate": 0.0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "num_frames_with_one_detection": 0,
            "num_frames_with_two_or_more_detections": 0,
            "fallback_frame_count": 0,
            "fallback_video_level": False,
            "mean_box_area_ratio": 0.0,
            "notes": "ok",
        }

        boxes_per_frame = [None] * n
        confs_list = []
        frames_with_det = 0
        frames_1det = 0
        frames_2det = 0
        frames_processed = 0

        # Run YOLO on every_n_frames
        for i in range(n):
            if i % self.every_n_frames != 0:
                continue
            frames_processed += 1
            frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
            dets = self.detect_athletes(frame_bgr)
            if dets:
                frames_with_det += 1
                if len(dets) == 1:
                    frames_1det += 1
                else:
                    frames_2det += 1
                for d in dets:
                    confs_list.append(d[4])
                box = self.compute_union_box(dets, h, w)
                boxes_per_frame[i] = box

        log["num_frames_processed_by_yolo"] = frames_processed
        log["num_frames_with_detection"] = frames_with_det
        log["num_frames_with_one_detection"] = frames_1det
        log["num_frames_with_two_or_more_detections"] = frames_2det
        log["detection_rate"] = frames_with_det / frames_processed if frames_processed > 0 else 0.0
        if confs_list:
            log["mean_confidence"] = float(np.mean(confs_list))
            log["min_confidence"] = float(np.min(confs_list))
            log["max_confidence"] = float(np.max(confs_list))

        # Check if we have any detection at all
        any_det = any(b is not None for b in boxes_per_frame)
        if not any_det:
            log["fallback_video_level"] = True
            log["notes"] = "no_detection|fallback_to_full_frame"
            log["used_yolo"] = False
            return list(frames_rgb), log

        if self.box_mode == "static":
            # A single box for the entire video: temporal union of the detected boxes.
            # Fixed geometry across all frames — does not inject false zoom/pan into the clip.
            detected = [b for b in boxes_per_frame if b is not None]
            static_box = [
                min(b[0] for b in detected),
                min(b[1] for b in detected),
                max(b[2] for b in detected),
                max(b[3] for b in detected),
            ]
            boxes_per_frame = [static_box] * n
            log["notes"] = "static_box"
        else:
            # Fill gaps: forward-fill, backward-fill
            last_box = None
            for i in range(n):
                if boxes_per_frame[i] is not None:
                    last_box = boxes_per_frame[i]
                else:
                    boxes_per_frame[i] = last_box

            # backward fill
            last_box = None
            for i in range(n - 1, -1, -1):
                if boxes_per_frame[i] is not None:
                    last_box = boxes_per_frame[i]
                else:
                    boxes_per_frame[i] = last_box

            # Apply smoothing
            if self.smoothing:
                prev_box = None
                for i in range(n):
                    if boxes_per_frame[i] is not None:
                        boxes_per_frame[i] = self.smooth_box(boxes_per_frame[i], prev_box)
                        prev_box = boxes_per_frame[i]

        # Process frames
        processed = []
        fallback_count = 0
        box_area_ratios = []
        frame_area = h * w

        for i, frame in enumerate(frames_rgb):
            box = boxes_per_frame[i]
            if box is None:
                processed.append(frame)
                fallback_count += 1
                continue
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            ratio = (bw * bh) / frame_area if frame_area > 0 else 0
            box_area_ratios.append(ratio)

            if self.spatial_mode == "yolo_union_crop":
                cropped = self.apply_crop(frame, box)
                processed.append(cropped)
            elif self.spatial_mode == "yolo_masked":
                masked = self.apply_mask(frame, box)
                processed.append(masked)
            else:
                processed.append(frame)

        log["fallback_frame_count"] = fallback_count
        if box_area_ratios:
            log["mean_box_area_ratio"] = float(np.mean(box_area_ratios))

        if frames_1det > 0 and frames_2det == 0:
            note = "single_athlete_detected"
        elif frames_with_det < frames_processed * 0.5:
            note = "partial_detection"
        else:
            note = None
        if note:
            log["notes"] = note if log["notes"] == "ok" else f"{log['notes']}|{note}"

        return processed, log


# ==============================================================
# PyTorch Dataset
# ==============================================================

def get_train_transform(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])


def get_val_transform(image_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])


def apply_edge_mask(frames, top=0.0, bottom=0.0, left=0.0, right=0.0, fill=0):
    """Blackens fixed margins (fraction of H/W) on every frame — removes static overlays
    (scoreboard, sponsor banners) that sit at a fixed screen position, without any
    detector. Same margins applied identically train/val/test (deterministic)."""
    out = []
    for f in frames:
        f = np.array(f, dtype=np.uint8).copy()
        h, w = f.shape[:2]
        t, b = int(round(h * top)), int(round(h * bottom))
        l, r = int(round(w * left)), int(round(w * right))
        if t > 0:
            f[:t, :] = fill
        if b > 0:
            f[h - b:, :] = fill
        if l > 0:
            f[:, :l] = fill
        if r > 0:
            f[:, w - r:] = fill
        out.append(f)
    return out


class JudoVideoDataset(Dataset):
    def __init__(self, df, num_frames, image_size, is_train, temporal_jitter,
                 spatial_mode, yolo_processor, bad_videos_list, binary_labels=None,
                 modality="rgb", flow_cache_dir=None, flow_bound=20.0,
                 mask_top=0.0, mask_bottom=0.0, mask_left=0.0, mask_right=0.0):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_train = is_train
        self.temporal_jitter = temporal_jitter
        self.spatial_mode = spatial_mode
        self.yolo_processor = yolo_processor
        self.bad_videos = bad_videos_list
        # binary_labels: optional override for OVR/OVO
        self.binary_labels = binary_labels
        # Two-stream / fluxo optico (item 4)
        self.modality = modality
        self.flow_cache_dir = flow_cache_dir
        self.flow_bound = flow_bound
        self.mask_top = mask_top
        self.mask_bottom = mask_bottom
        self.mask_left = mask_left
        self.mask_right = mask_right
        self.transform = get_train_transform(image_size) if is_train else get_val_transform(image_size)
        self.yolo_logs = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["video_path"]
        label = row["label"] if self.binary_labels is None else self.binary_labels[idx]

        need_rgb = self.modality in ("rgb", "two_stream")
        need_flow = self.modality in ("flow", "two_stream")
        use_yolo = self.spatial_mode != "full_frame" and self.spatial_mode != "edge_mask" and self.yolo_processor is not None
        use_edge_mask = self.spatial_mode == "edge_mask"
        # flow only reuses the crop/mask (YOLO or edge_mask) when frames are loaded once and
        # shared between the two branches (see load_or_compute_flow_from_frames); otherwise
        # flow follows the old path (reopens the video, always full_frame, untagged cache)
        need_shared_frames = need_rgb or (need_flow and (use_yolo or use_edge_mask))

        processed_frames = None  # post-YOLO frames (crop/mask), reused by the flow branch
        rgb_tensor = None
        if need_shared_frames:
            try:
                frames = VIDEO_LOADER.load(
                    path, self.num_frames,
                    is_train=self.is_train,
                    # flow/two_stream: deterministic sampling to align RGB with the flow cache
                    temporal_jitter=self.temporal_jitter and self.is_train and self.modality == "rgb",
                )
            except Exception as e:
                logger.warning(f"Failed to load video {path}: {e}")
                self.bad_videos.append({"video_path": path, "filename": row["filename"], "error": str(e)})
                return None

            # YOLO preprocessing — computed once, shared between RGB and flow (same box)
            if use_yolo:
                try:
                    frames_list = list(frames)
                    processed_frames, yolo_log = self.yolo_processor.process_video_frames(frames_list, path)
                    self.yolo_logs.append(yolo_log)
                except Exception as e:
                    logger.warning(f"YOLO processing failed for {path}: {e}")
                    processed_frames = None  # falls back to full_frame on both branches
            elif use_edge_mask:
                # static mask (same margins on every frame/video) — removes scoreboard/sponsor
                # overlay without needing a detector, shared between RGB and flow
                processed_frames = apply_edge_mask(
                    frames, top=self.mask_top, bottom=self.mask_bottom,
                    left=self.mask_left, right=self.mask_right,
                )

            if need_rgb:
                src = processed_frames if processed_frames is not None else frames
                # Resize all frames to same size (they may differ after crop)
                resized = []
                for f in src:
                    if isinstance(f, np.ndarray):
                        f = cv2.resize(f, (self.image_size, self.image_size))
                    else:
                        f = np.array(f, dtype=np.uint8)
                    resized.append(f)
                rgb_frames = np.array(resized, dtype=np.uint8)
                # Apply transforms frame by frame, then stack: [T, C, H, W] -> [C, T, H, W]
                tensor_frames = [self.transform(f) for f in rgb_frames]
                rgb_tensor = torch.stack(tensor_frames, dim=0).permute(1, 0, 2, 3)

        flow_tensor = None
        if need_flow:
            try:
                if (use_yolo or use_edge_mask) and processed_frames is not None:
                    if use_edge_mask:
                        tag = "_emask"
                    else:
                        tag = "_ycrop" if self.spatial_mode == "yolo_union_crop" else "_ymask"
                        if self.yolo_processor.box_mode == "static":
                            tag += "_static"
                    uv = load_or_compute_flow_from_frames(
                        processed_frames, row["filename"], self.num_frames, self.image_size,
                        self.flow_bound, self.flow_cache_dir, spatial_tag=tag, save=True,
                    )
                else:
                    uv = load_or_compute_flow(
                        path, row["filename"], self.num_frames, self.image_size,
                        self.flow_bound, self.flow_cache_dir, save=True,
                    )
                flow_tensor = flow_uv_to_tensor(uv, self.flow_bound)  # [3,T,H,W]
            except Exception as e:
                logger.warning(f"Flow failed for {path}: {e}")
                self.bad_videos.append({"video_path": path, "filename": row["filename"], "error": f"flow: {e}"})
                return None

        if self.modality == "rgb":
            out = rgb_tensor
        elif self.modality == "flow":
            out = flow_tensor
        else:  # two_stream: stacks along channels -> [6, T, H, W]
            out = torch.cat([rgb_tensor, flow_tensor], dim=0)
        return out, label, path, row["filename"]


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    tensors, labels, paths, filenames = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels), list(paths), list(filenames)


# ==============================================================
# Optical flow (two-stream) — item 4
# ==============================================================

def _flow_default_cache_dir(args):
    if getattr(args, "flow_cache_dir", None):
        return args.flow_cache_dir
    base = os.path.dirname(os.path.normpath(args.data_dir)) or "."
    return os.path.join(base, f"flow_cache_nf{args.num_frames}_sz{args.image_size}")


def _flow_cache_path(cache_dir, filename, num_frames, image_size, spatial_tag=""):
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    return os.path.join(cache_dir, f"{stem}__nf{num_frames}_sz{image_size}{spatial_tag}.npy")


def compute_flow_uv(frames_rgb, image_size):
    """frames_rgb: iterable of [H,W,3] uint8 (RGB). Returns [T,H,W,2] float32 (u,v in px),
    Farneback between consecutive frames resized to image_size. Last flow repeated
    so T = num_frames (aligns 1:1 with the RGB clip)."""
    grays = []
    for f in frames_rgb:
        g = cv2.resize(np.asarray(f, dtype=np.uint8), (image_size, image_size))
        grays.append(cv2.cvtColor(g, cv2.COLOR_RGB2GRAY))
    flows = []
    for a, b in zip(grays[:-1], grays[1:]):
        flows.append(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0))
    if flows:
        flows.append(flows[-1])
    else:
        flows = [np.zeros((image_size, image_size, 2), np.float32)]
    return np.asarray(flows, dtype=np.float32)


def _quantize_flow(uv, bound):
    q = np.clip(uv, -bound, bound)
    return ((q + bound) / (2.0 * bound) * 255.0).astype(np.uint8)


def _dequantize_flow(q, bound):
    return q.astype(np.float32) / 255.0 * (2.0 * bound) - bound


def load_or_compute_flow(path, filename, num_frames, image_size, bound, cache_dir, save=True, spatial_tag=""):
    """Loads flow from the cache (.npy uint8) or computes it on-the-fly (and saves it). ALWAYS
    uses deterministic sampling (linspace, is_train=False) to match the cache and the RGB stream.
    Always operates on the ORIGINAL video (full_frame); use load_or_compute_flow_from_frames
    when flow needs to see the same YOLO crop/mask as the RGB branch."""
    cpath = _flow_cache_path(cache_dir, filename, num_frames, image_size, spatial_tag) if cache_dir else None
    if cpath and os.path.exists(cpath):
        return _dequantize_flow(np.load(cpath), bound)
    frames = VIDEO_LOADER.load(path, num_frames, is_train=False, temporal_jitter=False)
    uv = compute_flow_uv(frames, image_size)
    if cpath and save:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.save(cpath, _quantize_flow(uv, bound))
    return uv


def load_or_compute_flow_from_frames(frames, filename, num_frames, image_size, bound, cache_dir,
                                      spatial_tag, save=True):
    """Like load_or_compute_flow, but receives frames ALREADY loaded (and already processed by
    YOLO, when applicable) instead of reopening the video from scratch. Used when spatial_mode
    != full_frame, so optical flow sees exactly the same region (same box) as the RGB branch —
    avoids computing flow on the raw video while RGB only sees the athlete crop (which would
    inject false pan/zoom between the two branches). spatial_tag differentiates this cache from
    the full_frame flow cache."""
    cpath = _flow_cache_path(cache_dir, filename, num_frames, image_size, spatial_tag) if cache_dir else None
    if cpath and os.path.exists(cpath):
        return _dequantize_flow(np.load(cpath), bound)
    uv = compute_flow_uv(frames, image_size)
    if cpath and save:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.save(cpath, _quantize_flow(uv, bound))
    return uv


def flow_uv_to_tensor(uv, bound):
    """uv [T,H,W,2] float (px) -> tensor [3,T,H,W] with (u, v, magnitude) normalized to ~[-1,1]."""
    u = uv[..., 0] / bound
    v = uv[..., 1] / bound
    mag = np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2) / bound
    stk = np.clip(np.stack([u, v, mag], axis=0), -1.0, 1.0).astype(np.float32)  # [3,T,H,W]
    return torch.from_numpy(stk)


def precompute_flow_dataset(df, args):
    """Precomputes and caches optical flow for every video in df (local use, CPU). Idempotent."""
    cache_dir = _flow_default_cache_dir(args)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"[precompute_flow] {len(df)} videos -> {cache_dir} "
                f"(nf={args.num_frames}, sz={args.image_size}, bound={args.flow_bound})")
    t0 = time.time(); done = ok = bad = 0
    for _, row in df.iterrows():
        done += 1
        cpath = _flow_cache_path(cache_dir, row["filename"], args.num_frames, args.image_size)
        if os.path.exists(cpath):
            ok += 1
        else:
            try:
                load_or_compute_flow(row["video_path"], row["filename"], args.num_frames,
                                     args.image_size, args.flow_bound, cache_dir, save=True)
                ok += 1
            except Exception as e:
                bad += 1
                logger.warning(f"[precompute_flow] falhou {row['filename']}: {e}")
        if done % 100 == 0:
            logger.info(f"  {done}/{len(df)} ok={ok} bad={bad} {time.time()-t0:.0f}s")
    logger.info(f"[precompute_flow] done ok={ok} bad={bad} em {time.time()-t0:.0f}s -> {cache_dir}")


# ==============================================================
# Model Building
# ==============================================================

def _load_video_backbone(model_name, pretrained=True):
    """Loads a torchvision video model with Kinetics-400 weights when available (fc original)."""

    def _load(model_fn, weights_cls, weights_name):
        if not pretrained:
            return model_fn(weights=None)
        # New API (torchvision >= 0.12): Weights enum
        try:
            w = getattr(weights_cls, weights_name)
            return model_fn(weights=w)
        except Exception:
            pass
        # Old API fallback (torchvision < 0.12)
        try:
            return model_fn(pretrained=True)
        except Exception:
            logger.warning("Could not load pretrained weights; initializing randomly")
            return model_fn(weights=None)

    if model_name == "r2plus1d_18":
        return _load(video_models.r2plus1d_18,
                     getattr(video_models, "R2Plus1D_18_Weights", None), "KINETICS400_V1")
    elif model_name == "r3d_18":
        return _load(video_models.r3d_18,
                     getattr(video_models, "R3D_18_Weights", None), "KINETICS400_V1")
    elif model_name == "mc3_18":
        return _load(video_models.mc3_18,
                     getattr(video_models, "MC3_18_Weights", None), "KINETICS400_V1")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _make_head(in_features, num_classes, dropout):
    head = nn.Linear(in_features, num_classes)
    # Dropout before fc (state_dict keys become fc.0/fc.1 when active)
    return nn.Sequential(nn.Dropout(p=dropout), head) if dropout > 0 else head


def build_model(model_name, num_classes, pretrained=True, dropout=0.0):
    model = _load_video_backbone(model_name, pretrained)
    model.fc = _make_head(model.fc.in_features, num_classes, dropout)
    return model


class TwoStreamVideoModel(nn.Module):
    """RGB + optical flow fusion via feature concatenation. Input: [B,6,T,H,W] (channels 0:3=RGB, 3:6=flow).
    Two independent backbones (feature extractors, no weight sharing) + a single fusion head called
    `.fc` — this way freezing (`"fc" not in name`) and the optimizer parameter groups (`model.fc`)
    work without any change. flow-only uses build_model (single 3-channel network)."""

    def __init__(self, model_name, num_classes, pretrained=True, dropout=0.0):
        super().__init__()
        self.rgb_net = _load_video_backbone(model_name, pretrained)
        self.flow_net = _load_video_backbone(model_name, pretrained)
        feat = self.rgb_net.fc.in_features + self.flow_net.fc.in_features
        self.rgb_net.fc = nn.Identity()
        self.flow_net.fc = nn.Identity()
        self.fc = _make_head(feat, num_classes, dropout)

    def forward(self, x):
        rgb = x[:, 0:3]
        flow = x[:, 3:6]
        feats = torch.cat([self.rgb_net(rgb), self.flow_net(flow)], dim=1)
        return self.fc(feats)


def build_model_for_modality(args, num_classes, dropout=0.0):
    if getattr(args, "input_modality", "rgb") == "two_stream":
        return TwoStreamVideoModel(args.model, num_classes, pretrained=True, dropout=dropout)
    # rgb and flow use the same 3-channel network (flow -> (u, v, magnitude))
    return build_model(args.model, num_classes, pretrained=True, dropout=dropout)


# ==============================================================
# Class Weights & Sampler
# ==============================================================

def get_class_weights(labels, num_classes, device):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ==============================================================
# Training Loop
# ==============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, is_binary):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    all_filenames = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        if batch is None:
            continue
        videos, labels, paths, filenames = batch
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        if is_binary:
            loss = criterion(outputs.squeeze(1), labels.float())
            probs = torch.sigmoid(outputs.squeeze(1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        else:
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

        total_loss += loss.item() * videos.size(0)
        all_preds.extend(preds.tolist() if hasattr(preds, "tolist") else list(preds))
        all_labels.extend(labels.cpu().numpy().tolist())
        if is_binary:
            all_probs.extend(probs.tolist())
        else:
            all_probs.extend(probs.tolist())
        all_paths.extend(paths)
        all_filenames.extend(filenames)

    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0
    acc = accuracy_score(all_labels, all_preds) if n > 0 else 0
    avg_arg = "binary" if is_binary else "macro"
    f1 = f1_score(all_labels, all_preds, average=avg_arg, zero_division=0) if n > 0 else 0

    return avg_loss, acc, f1, all_preds, all_labels, all_probs, all_paths, all_filenames


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def freeze_early_layers(model):
    """Keeps stem and layer1 frozen (used in the finetune phase with --freeze_early)."""
    n_frozen = 0
    for name, param in unwrap_model(model).named_parameters():
        parts = name.split(".")
        # casa tanto o modelo unico (stem.*, layer1.*) quanto o two-stream (rgb_net.stem.*, ...)
        if "stem" in parts or "layer1" in parts:
            param.requires_grad = False
            n_frozen += 1
    logger.info(f"freeze_early: {n_frozen} params de stem+layer1 congelados na fase finetune")


class SmoothedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss com label smoothing nos alvos: y -> y*(1-s) + 0.5*s."""

    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def make_binary_train_criterion(n_pos, n_neg, args, device):
    """Criterio de treino binario respeitando --pos_weight_mode e --label_smoothing."""
    mode = getattr(args, "pos_weight_mode", "full")
    smoothing = getattr(args, "label_smoothing", 0.0)
    if mode == "none":
        pos_weight = None
    else:
        ratio = n_neg / max(n_pos, 1)
        if mode == "sqrt":
            ratio = math.sqrt(ratio)
        pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)
        logger.info(f"pos_weight ({mode}): {float(pos_weight[0]):.3f}")
    return SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, smoothing=smoothing)


def make_phase_scheduler(optimizer, args, n_epochs):
    """Per-phase scheduler: plateau (original behavior) or warmup+cosine,
    which decays deterministically and does not react to validation noise."""
    if getattr(args, "scheduler", "plateau") == "cosine":
        warmup = max(0, min(getattr(args, "warmup_epochs", 2), max(n_epochs - 1, 0)))

        def lr_lambda(epoch):  # epoch = numero de step() ja feitos (0-indexado)
            if epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            t = (epoch - warmup) / max(1, n_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)


def step_scheduler(scheduler, val_loss):
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


# ==============================================================
# Plotting
# ==============================================================

def save_training_curves(history, out_dir):
    # Use a global sequential index so the x-axis never resets between phases
    global_epochs = list(range(1, len(history["epoch"]) + 1))

    # Find where head phase ends (for the vertical separator line)
    phases = history.get("phase", [])
    phase_boundary = None
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_boundary = global_epochs[i - 1] + 0.5
            break

    def _add_phase_line(ax):
        if phase_boundary is not None:
            ax.axvline(phase_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="head→finetune")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(global_epochs, history["train_loss"], label="Train")
    axes[0].plot(global_epochs, history["val_loss"], label="Val")
    _add_phase_line(axes[0])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Global Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(global_epochs, history["train_acc"], label="Train")
    axes[1].plot(global_epochs, history["val_acc"], label="Val")
    _add_phase_line(axes[1])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Global Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(global_epochs, history["train_f1"], label="Train")
    axes[2].plot(global_epochs, history["val_f1"], label="Val")
    _add_phase_line(axes[2])
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Global Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=100)
    plt.close()

    # Individual curves
    for metric, fname in [
        ("loss", "loss_curve.png"),
        ("acc", "accuracy_curve.png"),
        ("f1", "macro_f1_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(global_epochs, history[f"train_{metric}"], label="Train")
        ax.plot(global_epochs, history[f"val_{metric}"], label="Val")
        _add_phase_line(ax)
        ax.set_title(metric.upper())
        ax.set_xlabel("Global Epoch")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=100)
        plt.close()


def save_confusion_matrix(cm, classes, out_dir, title="Confusion Matrix", normalize=False, fname=None):
    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_plot = cm_plot / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    thresh = cm_plot.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = f"{cm_plot[i, j]:{fmt}}"
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    if fname is None:
        fname = "confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=100)
    plt.close()


# ==============================================================
# Prediction Saving
# ==============================================================

def softmax_probs_to_dict(probs, classes):
    """Given [N, C] or [N] probs array, return list of per-class prob dicts."""
    result = []
    probs = np.array(probs)
    if probs.ndim == 1:
        # Binary sigmoid output: convert to 3-class using index mapping
        return None  # handled separately for binary
    for row in probs:
        d = {f"prob_{cls}": float(row[i]) for i, cls in enumerate(classes)}
        result.append(d)
    return result


def save_multiclass_predictions(preds, labels, probs, paths, filenames, classes, out_dir, split_name):
    probs = np.array(probs)
    rows = []
    for i in range(len(preds)):
        sorted_idx = np.argsort(probs[i])[::-1]
        margin = float(probs[i][sorted_idx[0]] - probs[i][sorted_idx[1]]) if len(sorted_idx) > 1 else 1.0
        row = {
            "video_path": paths[i],
            "filename": filenames[i],
            "true_class": classes[labels[i]],
            "predicted_class": classes[preds[i]],
            "correct": int(preds[i] == labels[i]),
            "confidence": float(probs[i][preds[i]]),
            "margin_top1_top2": margin,
        }
        for j, cls in enumerate(classes):
            row[f"prob_{cls}"] = float(probs[i][j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"predictions_{split_name}.csv"), index=False)

    if split_name == "test":
        mis = df[df["correct"] == 0].sort_values("confidence", ascending=True)
        mis.to_csv(os.path.join(out_dir, "misclassified_test.csv"), index=False)

        unc = df.sort_values("margin_top1_top2", ascending=True).head(max(1, len(df) // 5))
        unc.to_csv(os.path.join(out_dir, "uncertain_test.csv"), index=False)

    return df


def save_classification_report_files(labels, preds, classes, out_dir, prefix=""):
    report_str = classification_report(labels, preds, target_names=classes, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)

    fname_txt = f"{prefix}classification_report.txt" if prefix else "classification_report.txt"
    fname_json = f"{prefix}classification_report.json" if prefix else "classification_report.json"

    with open(os.path.join(out_dir, fname_txt), "w") as f:
        f.write(report_str)
    with open(os.path.join(out_dir, fname_json), "w") as f:
        json.dump(report_dict, f, indent=2)

    return report_str, report_dict


# ==============================================================
# DataLoader Factory
# ==============================================================

def make_dataloaders(train_df, val_df, test_df, args, device,
                     yolo_processor, bad_videos, binary_labels_train=None,
                     binary_labels_val=None, binary_labels_test=None,
                     balance_strategy=None, num_classes=None):
    if balance_strategy is None:
        balance_strategy = args.balance_strategy
    if num_classes is None:
        num_classes = len(CLASSES)

    is_binary = binary_labels_train is not None

    modality = getattr(args, "input_modality", "rgb")
    flow_cache_dir = _flow_default_cache_dir(args) if modality != "rgb" else None
    flow_bound = getattr(args, "flow_bound", 20.0)
    flow_kw = dict(modality=modality, flow_cache_dir=flow_cache_dir, flow_bound=flow_bound)
    mask_kw = dict(
        mask_top=getattr(args, "mask_top", 0.0), mask_bottom=getattr(args, "mask_bottom", 0.0),
        mask_left=getattr(args, "mask_left", 0.0), mask_right=getattr(args, "mask_right", 0.0),
    )

    train_ds = JudoVideoDataset(
        train_df, args.num_frames, args.image_size, is_train=True,
        temporal_jitter=parse_bool(args.temporal_jitter),
        spatial_mode=args.spatial_mode, yolo_processor=yolo_processor,
        bad_videos_list=bad_videos, binary_labels=binary_labels_train, **flow_kw, **mask_kw,
    )
    val_ds = JudoVideoDataset(
        val_df, args.num_frames, args.image_size, is_train=False,
        temporal_jitter=False, spatial_mode=args.spatial_mode,
        yolo_processor=yolo_processor, bad_videos_list=bad_videos,
        binary_labels=binary_labels_val, **flow_kw, **mask_kw,
    )
    test_ds = JudoVideoDataset(
        test_df, args.num_frames, args.image_size, is_train=False,
        temporal_jitter=False, spatial_mode=args.spatial_mode,
        yolo_processor=yolo_processor, bad_videos_list=bad_videos,
        binary_labels=binary_labels_test, **flow_kw, **mask_kw,
    )

    # Sampler
    if balance_strategy == "weighted_sampler" and not is_binary:
        labels_for_sampler = train_df["label"].values
        sampler = get_weighted_sampler(labels_for_sampler, num_classes)
        shuffle = False
    elif balance_strategy == "weighted_sampler" and is_binary:
        labels_for_sampler = np.array(binary_labels_train)
        sampler = get_weighted_sampler(labels_for_sampler, 2)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


# ==============================================================
# Compute metrics helper
# ==============================================================

def compute_multiclass_metrics(labels, preds, probs, classes, out_dir, split_name):
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(len(classes))))
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))

    logger.info(f"[{split_name}] Acc={acc:.4f} MacroF1={macro_f1:.4f} WeightedF1={weighted_f1:.4f}")
    for i, cls in enumerate(classes):
        logger.info(f"  {cls}: F1={per_class_f1[i]:.4f}")

    save_confusion_matrix(cm, classes, out_dir, title=f"Confusion Matrix ({split_name})", normalize=False)
    save_confusion_matrix(cm, classes, out_dir, title=f"Normalized ({split_name})", normalize=True)
    report_str, report_dict = save_classification_report_files(labels, preds, classes, out_dir)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {cls: float(per_class_f1[i]) for i, cls in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }


# ==============================================================
# MULTICLASS MODE
# ==============================================================

def run_multiclass(args, train_df, val_df, test_df, device, yolo_processor, bad_videos, classes=None):
    if classes is None:
        classes = CLASSES
    out_dir = os.path.join(args.output_dir, "multiclass")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"=== MULTICLASS MODE ({len(classes)} classes: {classes}) ===")

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(
        train_df, val_df, test_df, args, device, yolo_processor, bad_videos,
        num_classes=len(classes),
    )

    model = build_model_for_modality(args, num_classes=len(classes),
                                     dropout=getattr(args, "dropout", 0.0))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")

    # Override criterion with class weights if needed. Deliberately ONLY for
    # "class_weights" (not also "weighted_sampler") -- balance_strategy=weighted_sampler
    # should correct the imbalance only via the sampler, without also reweighting the loss
    # (double correction), the same issue already documented in CLAUDE.md for OVR.
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    if args.balance_strategy == "class_weights":
        cw = get_class_weights(train_df["label"].values, len(classes), device)
        logger.info(f"Class weights: {cw.cpu().numpy()}")
        criterion_train_with_weights = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    else:
        criterion_train_with_weights = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    model, history, best_val_f1 = _train_model_with_criterion(
        model, train_loader, val_loader, device, args, out_dir,
        is_binary=False,
        head_epochs=args.epochs_head,
        finetune_epochs=args.epochs_finetune,
        criterion_train=criterion_train_with_weights,
        criterion_eval=nn.CrossEntropyLoss(),
    )

    # Evaluate on val and test
    logger.info("Evaluating on val set...")
    _, va_acc, va_f1, va_preds, va_labels, va_probs, va_paths, va_fnames = evaluate(
        model, val_loader, nn.CrossEntropyLoss(), device, is_binary=False
    )
    save_multiclass_predictions(va_preds, va_labels, va_probs, va_paths, va_fnames, classes, out_dir, "val")

    logger.info("Evaluating on test set...")
    _, te_acc, te_f1, te_preds, te_labels, te_probs, te_paths, te_fnames = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device, is_binary=False
    )
    save_multiclass_predictions(te_preds, te_labels, te_probs, te_paths, te_fnames, classes, out_dir, "test")

    metrics = compute_multiclass_metrics(te_labels, te_preds, te_probs, classes, out_dir, "test")
    metrics["best_val_f1"] = best_val_f1

    # Collect yolo logs from datasets
    all_yolo_logs = train_ds.yolo_logs + val_ds.yolo_logs + test_ds.yolo_logs
    return metrics, all_yolo_logs


def _train_model_with_criterion(model, train_loader, val_loader, device, args, out_dir,
                                 is_binary, head_epochs, finetune_epochs,
                                 criterion_train, criterion_eval,
                                 skip_finetune_if_stopped=False):
    """Loop de treino unificado (multiclass, binary_tachi, OVR, OVO)."""
    os.makedirs(out_dir, exist_ok=True)
    use_amp = parse_bool(args.use_amp) and device.type == "cuda"
    scaler = _make_scaler() if use_amp else None
    accum_steps = max(1, getattr(args, "grad_accum_steps", 1))
    smooth_window = max(1, getattr(args, "val_smooth_window", 1))

    history = {
        "epoch": [], "phase": [],
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "lr": [],
    }

    best_val_f1 = -1.0  # melhor val F1 SUAVIZADO (janela = val_smooth_window)
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    recent_val_f1 = deque(maxlen=smooth_window)  # persiste entre fases

    def run_epoch_train(optimizer):
        model.train()
        total_loss = 0.0
        all_preds, all_labels_ep = [], []
        optimizer.zero_grad()
        pending = False  # accumulated gradients without a step

        def opt_step():
            if use_amp and device.type == "cuda":
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc="Train", leave=False)):
            if batch is None:
                continue
            videos, labels, _, _ = batch
            videos = videos.to(device)
            labels = labels.to(device)
            if use_amp and device.type == "cuda":
                with _autocast_ctx("cuda"):
                    outputs = model(videos)
                    if is_binary:
                        loss = criterion_train(outputs.squeeze(1), labels.float())
                    else:
                        loss = criterion_train(outputs, labels)
                scaler.scale(loss / accum_steps).backward()
            else:
                outputs = model(videos)
                if is_binary:
                    loss = criterion_train(outputs.squeeze(1), labels.float())
                else:
                    loss = criterion_train(outputs, labels)
                (loss / accum_steps).backward()
            pending = True
            if (i + 1) % accum_steps == 0:
                opt_step()
                pending = False
            total_loss += loss.item() * videos.size(0)
            if is_binary:
                preds = (torch.sigmoid(outputs.squeeze(1)) >= 0.5).long().cpu().numpy()
            else:
                preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels_ep.extend(labels.cpu().numpy().tolist())
        if pending:  # flush dos gradientes restantes
            opt_step()
        n = len(all_labels_ep)
        avg_loss = total_loss / n if n > 0 else 0
        acc = accuracy_score(all_labels_ep, all_preds) if n > 0 else 0
        avg_arg = "binary" if is_binary else "macro"
        f1 = f1_score(all_labels_ep, all_preds, average=avg_arg, zero_division=0) if n > 0 else 0
        return avg_loss, acc, f1

    def do_phase(phase_name, n_epochs, optimizer, scheduler):
        nonlocal best_val_f1, best_val_loss, best_model_state, patience_counter
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1 = run_epoch_train(optimizer)
            va_loss, va_acc, va_f1, _, _, _, _, _ = evaluate(
                model, val_loader, criterion_eval, device, is_binary
            )
            step_scheduler(scheduler, va_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Selecao por media movel do val F1 (janela=1 reproduz o comportamento original)
            recent_val_f1.append(va_f1)
            sm_f1 = sum(recent_val_f1) / len(recent_val_f1)
            sm_txt = f" VaF1sm={sm_f1:.3f}" if smooth_window > 1 else ""
            logger.info(
                f"[{phase_name}] Epoch {epoch}/{n_epochs} | "
                f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.3f} TrF1={tr_f1:.3f} | "
                f"VaLoss={va_loss:.4f} VaAcc={va_acc:.3f} VaF1={va_f1:.3f}{sm_txt} | "
                f"LR={current_lr:.2e} | {elapsed:.1f}s"
            )
            history["epoch"].append(epoch)
            history["phase"].append(phase_name)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["train_f1"].append(tr_f1)
            history["val_f1"].append(va_f1)
            history["lr"].append(current_lr)
            improved = (sm_f1 > best_val_f1) or (abs(sm_f1 - best_val_f1) < 1e-6 and va_loss < best_val_loss)
            if improved:
                best_val_f1 = sm_f1
                best_val_loss = va_loss
                best_model_state = {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()}
                torch.save(best_model_state, os.path.join(out_dir, "best_model.pt"))
                patience_counter = 0
                logger.info(f"  -> New best model saved (val_f1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch} of phase {phase_name}")
                    return True
        return False

    # Phase 1
    logger.info("=== Phase 1: Training head only ===")
    freeze_backbone(model)
    fc_params = [p for p in model.parameters() if p.requires_grad]
    opt_head = optim.AdamW(fc_params, lr=args.lr_head, weight_decay=args.weight_decay)
    sched_head = make_phase_scheduler(opt_head, args, head_epochs)
    patience_counter = 0
    stopped = do_phase("head", head_epochs, opt_head, sched_head)

    # Phase 2
    if not (skip_finetune_if_stopped and stopped and head_epochs > 0):
        logger.info("=== Phase 2: Full fine-tuning ===")
        unfreeze_all(model)
        if parse_bool(getattr(args, "freeze_early", "false")):
            freeze_early_layers(model)
        opt_full = optim.AdamW([
            {"params": [p for n, p in model.named_parameters()
                        if "fc" not in n and p.requires_grad], "lr": args.lr_backbone},
            {"params": unwrap_model(model).fc.parameters(), "lr": args.lr_head},
        ], weight_decay=args.weight_decay)
        sched_full = make_phase_scheduler(opt_full, args, finetune_epochs)
        patience_counter = 0
        do_phase("finetune", finetune_epochs, opt_full, sched_full)

    torch.save(unwrap_model(model).state_dict(), os.path.join(out_dir, "last_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    save_training_curves(history, out_dir)

    if best_model_state is not None:
        unwrap_model(model).load_state_dict(best_model_state)

    return model, history, best_val_f1


# ==============================================================
# OVR MODE
# ==============================================================

def run_ovr(args, train_df, val_df, test_df, device, yolo_processor, bad_videos):
    out_dir = os.path.join(args.output_dir, "ovr")
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== OVR MODE ===")

    all_yolo_logs = []
    # Store test predictions per class model: {class_name: (probs, preds, labels, paths, fnames)}
    ovr_test_probs = {}
    ovr_val_probs = {}

    for target_cls in CLASSES:
        sub_dir = os.path.join(out_dir, f"{target_cls}_vs_rest")
        os.makedirs(sub_dir, exist_ok=True)
        logger.info(f"--- OVR: {target_cls} vs rest ---")

        # Binary labels
        bin_train = [1 if row["class_name"] == target_cls else 0 for _, row in train_df.iterrows()]
        bin_val   = [1 if row["class_name"] == target_cls else 0 for _, row in val_df.iterrows()]
        bin_test  = [1 if row["class_name"] == target_cls else 0 for _, row in test_df.iterrows()]

        # pos_weight (modo controlado por --pos_weight_mode) + label smoothing opcional
        n_pos = sum(bin_train)
        n_neg = len(bin_train) - n_pos
        criterion_train = make_binary_train_criterion(n_pos, n_neg, args, device)
        criterion_eval  = nn.BCEWithLogitsLoss()

        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(
            train_df, val_df, test_df, args, device, yolo_processor, bad_videos,
            binary_labels_train=bin_train,
            binary_labels_val=bin_val,
            binary_labels_test=bin_test,
        )

        model = build_model(args.model, num_classes=1, pretrained=True,
                            dropout=getattr(args, "dropout", 0.0))
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")

        model, history, best_val_f1 = _train_model_with_criterion(
            model, train_loader, val_loader, device, args, sub_dir,
            is_binary=True,
            head_epochs=args.epochs_head,
            finetune_epochs=args.epochs_finetune,
            criterion_train=criterion_train,
            criterion_eval=criterion_eval,
        )

        # Val predictions
        _, _, _, va_preds, va_labels, va_probs, va_paths, va_fnames = evaluate(
            model, val_loader, criterion_eval, device, is_binary=True
        )
        ovr_val_probs[target_cls] = (np.array(va_probs), np.array(va_preds), np.array(va_labels), va_paths, va_fnames)

        # Test predictions
        _, _, _, te_preds, te_labels_bin, te_probs, te_paths, te_fnames = evaluate(
            model, test_loader, criterion_eval, device, is_binary=True
        )
        ovr_test_probs[target_cls] = (np.array(te_probs), np.array(te_preds), np.array(te_labels_bin), te_paths, te_fnames)

        # Save individual binary predictions
        bin_rows_val = []
        for i in range(len(va_preds)):
            bin_rows_val.append({
                "video_path": va_paths[i], "filename": va_fnames[i],
                "true_binary": va_labels[i], "pred_binary": va_preds[i],
                "prob_positive": float(va_probs[i]), "correct": int(va_preds[i] == va_labels[i]),
            })
        pd.DataFrame(bin_rows_val).to_csv(os.path.join(sub_dir, "predictions_val.csv"), index=False)

        bin_rows_test = []
        for i in range(len(te_preds)):
            bin_rows_test.append({
                "video_path": te_paths[i], "filename": te_fnames[i],
                "true_binary": te_labels_bin[i], "pred_binary": te_preds[i],
                "prob_positive": float(te_probs[i]), "correct": int(te_preds[i] == te_labels_bin[i]),
            })
        pd.DataFrame(bin_rows_test).to_csv(os.path.join(sub_dir, "predictions_test.csv"), index=False)

        # Sub confusion matrix and report
        cm_bin = confusion_matrix(te_labels_bin, te_preds, labels=[0, 1])
        save_confusion_matrix(cm_bin, [f"not_{target_cls}", target_cls], sub_dir,
                              title=f"CM: {target_cls} vs rest")
        save_classification_report_files(te_labels_bin, te_preds, [f"not_{target_cls}", target_cls], sub_dir)

        all_yolo_logs.extend(train_ds.yolo_logs + val_ds.yolo_logs + test_ds.yolo_logs)

    # Ensemble: align all predictions to test_df order via path lookup
    te_paths_ref   = test_df["video_path"].tolist()
    te_fnames_ref  = test_df["filename"].tolist()
    te_labels_true = test_df["label"].values
    n_test = len(te_paths_ref)

    # Build prob matrix [N, 3]: col j = sigmoid prob from OVR model for class j
    prob_matrix = np.full((n_test, len(CLASSES)), 0.5)
    for j, cls in enumerate(CLASSES):
        probs_j, _, _, paths_j, _ = ovr_test_probs[cls]
        path_to_prob = dict(zip(paths_j, probs_j.tolist()))
        for i, p in enumerate(te_paths_ref):
            if p in path_to_prob:
                prob_matrix[i, j] = path_to_prob[p]

    # Decision strategy
    if args.decision_strategy == "hierarchical":
        ensemble_preds = _ovr_hierarchical_decision(prob_matrix)
    else:
        ensemble_preds = prob_matrix.argmax(axis=1)

    # Save ensemble predictions
    rows = []
    for i in range(n_test):
        sorted_idx = np.argsort(prob_matrix[i])[::-1]
        margin = float(prob_matrix[i][sorted_idx[0]] - prob_matrix[i][sorted_idx[1]]) if len(sorted_idx) > 1 else 1.0
        rows.append({
            "video_path": te_paths_ref[i],
            "filename": te_fnames_ref[i],
            "true_class": IDX_TO_CLASS.get(int(te_labels_true[i]), "unknown"),
            "predicted_class": CLASSES[ensemble_preds[i]],
            "confidence": float(prob_matrix[i][ensemble_preds[i]]),
            "margin_top1_top2": margin,
            "prob_sutemi_waza": float(prob_matrix[i][CLASS_TO_IDX["sutemi_waza"]]),
            "prob_ashi_waza": float(prob_matrix[i][CLASS_TO_IDX["ashi_waza"]]),
            "prob_te_waza": float(prob_matrix[i][CLASS_TO_IDX["te_waza"]]),
            "correct": int(ensemble_preds[i] == te_labels_true[i]),
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(out_dir, "predictions_test_ovr.csv"), index=False)

    mis = pred_df[pred_df["correct"] == 0].sort_values("confidence", ascending=True)
    mis.to_csv(os.path.join(out_dir, "misclassified_test_ovr.csv"), index=False)
    unc = pred_df.sort_values("margin_top1_top2", ascending=True).head(max(1, len(pred_df) // 5))
    unc.to_csv(os.path.join(out_dir, "uncertain_test_ovr.csv"), index=False)

    # Ensemble metrics
    cm_ovr = confusion_matrix(te_labels_true, ensemble_preds, labels=list(range(len(CLASSES))))
    save_confusion_matrix(cm_ovr, CLASSES, out_dir, title="OVR Ensemble", normalize=False, fname="confusion_matrix_ovr.png")
    save_confusion_matrix(cm_ovr, CLASSES, out_dir, title="OVR Ensemble (Norm)", normalize=True, fname="confusion_matrix_ovr_normalized.png")
    _save_report_fixed(te_labels_true, ensemble_preds, CLASSES, out_dir, "classification_report_ovr")

    macro_f1 = f1_score(te_labels_true, ensemble_preds, average="macro", zero_division=0)
    acc = accuracy_score(te_labels_true, ensemble_preds)
    logger.info(f"OVR Ensemble Test: Acc={acc:.4f} MacroF1={macro_f1:.4f}")

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {cls: float(f1_score(te_labels_true, ensemble_preds, average=None, zero_division=0, labels=list(range(len(CLASSES))))[i]) for i, cls in enumerate(CLASSES)},
    }
    return metrics, all_yolo_logs


def _ovr_hierarchical_decision(prob_matrix):
    """Hierarchical: first decide sutemi_waza vs rest, then ashi_waza vs te_waza."""
    preds = []
    idx_sutemi = CLASS_TO_IDX["sutemi_waza"]
    idx_ashi = CLASS_TO_IDX["ashi_waza"]
    idx_te = CLASS_TO_IDX["te_waza"]
    for row in prob_matrix:
        if row[idx_sutemi] >= 0.5:
            preds.append(idx_sutemi)
        else:
            if row[idx_ashi] >= row[idx_te]:
                preds.append(idx_ashi)
            else:
                preds.append(idx_te)
    return np.array(preds)


# ==============================================================
# OVO MODE
# ==============================================================

def run_ovo(args, train_df, val_df, test_df, device, yolo_processor, bad_videos):
    out_dir = os.path.join(args.output_dir, "ovo")
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== OVO MODE ===")

    all_yolo_logs = []
    pairs = [
        ("sutemi_waza", "ashi_waza"),
        ("sutemi_waza", "te_waza"),
        ("ashi_waza", "te_waza"),
    ]
    pair_test_probs = {}  # (cls_a, cls_b) -> array [N_test] of prob(cls_b wins)

    for cls_a, cls_b in pairs:
        pair_name = f"{cls_a}_vs_{cls_b}"
        sub_dir = os.path.join(out_dir, pair_name)
        os.makedirs(sub_dir, exist_ok=True)
        logger.info(f"--- OVO: {cls_a} vs {cls_b} ---")

        # Filter to only these two classes FOR TRAINING
        train_pair = train_df[train_df["class_name"].isin([cls_a, cls_b])].reset_index(drop=True)
        val_pair   = val_df[val_df["class_name"].isin([cls_a, cls_b])].reset_index(drop=True)

        # Binary labels: 0 = cls_a, 1 = cls_b
        bin_train_pair = [0 if r["class_name"] == cls_a else 1 for _, r in train_pair.iterrows()]
        bin_val_pair   = [0 if r["class_name"] == cls_a else 1 for _, r in val_pair.iterrows()]
        # For inference, run on ALL test videos
        bin_test_all   = [0 if r["class_name"] == cls_a else 1 for _, r in test_df.iterrows()]

        # Balance (positivo = cls_b; modo controlado por --pos_weight_mode)
        n_a = sum(1 for l in bin_train_pair if l == 0)
        n_b = sum(1 for l in bin_train_pair if l == 1)
        criterion_train = make_binary_train_criterion(n_pos=n_b, n_neg=n_a, args=args, device=device)
        criterion_eval  = nn.BCEWithLogitsLoss()

        # Train loader uses only the pair subset
        _mask_kw = dict(
            mask_top=getattr(args, "mask_top", 0.0), mask_bottom=getattr(args, "mask_bottom", 0.0),
            mask_left=getattr(args, "mask_left", 0.0), mask_right=getattr(args, "mask_right", 0.0),
        )
        train_ds_pair = JudoVideoDataset(
            train_pair, args.num_frames, args.image_size, is_train=True,
            temporal_jitter=parse_bool(args.temporal_jitter),
            spatial_mode=args.spatial_mode, yolo_processor=yolo_processor,
            bad_videos_list=bad_videos, binary_labels=bin_train_pair, **_mask_kw,
        )
        val_ds_pair = JudoVideoDataset(
            val_pair, args.num_frames, args.image_size, is_train=False,
            temporal_jitter=False, spatial_mode=args.spatial_mode,
            yolo_processor=yolo_processor, bad_videos_list=bad_videos,
            binary_labels=bin_val_pair, **_mask_kw,
        )
        # Test uses ALL test videos
        test_ds_all = JudoVideoDataset(
            test_df, args.num_frames, args.image_size, is_train=False,
            temporal_jitter=False, spatial_mode=args.spatial_mode,
            yolo_processor=yolo_processor, bad_videos_list=bad_videos,
            binary_labels=bin_test_all, **_mask_kw,
        )

        bin_train_sampler = get_weighted_sampler(np.array(bin_train_pair), 2) if args.balance_strategy == "weighted_sampler" else None
        train_loader_pair = DataLoader(
            train_ds_pair, batch_size=args.batch_size,
            shuffle=(bin_train_sampler is None), sampler=bin_train_sampler,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        val_loader_pair = DataLoader(
            val_ds_pair, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        test_loader_all = DataLoader(
            test_ds_all, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(args.model, num_classes=1, pretrained=True,
                            dropout=getattr(args, "dropout", 0.0))
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")

        model, history, best_val_f1 = _train_model_with_criterion(
            model, train_loader_pair, val_loader_pair, device, args, sub_dir,
            is_binary=True,
            head_epochs=args.epochs_head,
            finetune_epochs=args.epochs_finetune,
            criterion_train=criterion_train,
            criterion_eval=criterion_eval,
        )

        # Val predictions (on pair subset)
        _, _, _, va_preds, va_labels, va_probs, va_paths, va_fnames = evaluate(
            model, val_loader_pair, criterion_eval, device, is_binary=True
        )
        pd.DataFrame([{
            "video_path": va_paths[i], "filename": va_fnames[i],
            "true_binary": va_labels[i], "pred_binary": va_preds[i],
            "prob_class_b": float(va_probs[i]), "correct": int(va_preds[i] == va_labels[i]),
        } for i in range(len(va_preds))]).to_csv(os.path.join(sub_dir, "predictions_val.csv"), index=False)

        # Test predictions (ALL test videos)
        _, _, _, te_preds, te_labels_bin, te_probs, te_paths, te_fnames = evaluate(
            model, test_loader_all, criterion_eval, device, is_binary=True
        )
        pd.DataFrame([{
            "video_path": te_paths[i], "filename": te_fnames[i],
            "true_binary": te_labels_bin[i], "pred_binary": te_preds[i],
            "prob_class_b": float(te_probs[i]), "correct": int(te_preds[i] == te_labels_bin[i]),
        } for i in range(len(te_preds))]).to_csv(os.path.join(sub_dir, "predictions_test.csv"), index=False)

        # Sub CM (only on pair members in test)
        pair_mask = test_df["class_name"].isin([cls_a, cls_b]).values
        pair_te_labels = np.array(te_labels_bin)[pair_mask]
        pair_te_preds  = np.array(te_preds)[pair_mask]
        if len(pair_te_labels) > 0:
            cm_pair = confusion_matrix(pair_te_labels, pair_te_preds, labels=[0, 1])
            save_confusion_matrix(cm_pair, [cls_a, cls_b], sub_dir, title=f"CM: {cls_a} vs {cls_b}")
            save_classification_report_files(pair_te_labels, pair_te_preds, [cls_a, cls_b], sub_dir)

        # Store probs aligned to test_df order
        path_to_prob = dict(zip(te_paths, te_probs))
        pair_test_probs[(cls_a, cls_b)] = {
            "probs": np.array([path_to_prob.get(p, 0.5) for p in test_df["video_path"].values]),
            "cls_a": cls_a,
            "cls_b": cls_b,
        }
        all_yolo_logs.extend(train_ds_pair.yolo_logs + val_ds_pair.yolo_logs + test_ds_all.yolo_logs)

    # OVO Ensemble: voting
    n_test = len(test_df)
    votes = np.zeros((n_test, len(CLASSES)), dtype=float)
    pair_prob_store = {}  # for saving detailed predictions

    for (cls_a, cls_b), info in pair_test_probs.items():
        probs_b = info["probs"]  # prob that cls_b wins
        ia = CLASS_TO_IDX[cls_a]
        ib = CLASS_TO_IDX[cls_b]
        for i in range(n_test):
            if probs_b[i] >= 0.5:
                votes[i, ib] += 1
            else:
                votes[i, ia] += 1
        pair_prob_store[f"prob_{cls_a}_vs_{cls_b}"] = probs_b

    # Decide: argmax votes; ties broken by sum of probs (treating probs as confidence of being that class)
    ensemble_preds = []
    for i in range(n_test):
        max_votes = votes[i].max()
        candidates = np.where(votes[i] == max_votes)[0]
        if len(candidates) == 1:
            ensemble_preds.append(int(candidates[0]))
        else:
            # Use raw probs to break tie: sum of "I won" probabilities for each candidate
            scores = np.zeros(len(CLASSES))
            for (cls_a, cls_b), info in pair_test_probs.items():
                ia = CLASS_TO_IDX[cls_a]
                ib = CLASS_TO_IDX[cls_b]
                scores[ia] += (1 - info["probs"][i])
                scores[ib] += info["probs"][i]
            best_candidate = candidates[np.argmax(scores[candidates])]
            ensemble_preds.append(int(best_candidate))

    ensemble_preds = np.array(ensemble_preds)
    te_labels_true = test_df["label"].values
    te_paths_ref = test_df["video_path"].tolist()
    te_fnames_ref = test_df["filename"].tolist()

    # Save OVO ensemble predictions
    rows = []
    for i in range(n_test):
        row = {
            "video_path": te_paths_ref[i],
            "filename": te_fnames_ref[i],
            "true_class": IDX_TO_CLASS.get(int(te_labels_true[i]), "unknown"),
            "predicted_class": CLASSES[ensemble_preds[i]],
            "confidence": float(votes[i, ensemble_preds[i]] / len(pairs)),
            "votes_sutemi_waza": int(votes[i, CLASS_TO_IDX["sutemi_waza"]]),
            "votes_ashi_waza": int(votes[i, CLASS_TO_IDX["ashi_waza"]]),
            "votes_te_waza": int(votes[i, CLASS_TO_IDX["te_waza"]]),
            "correct": int(ensemble_preds[i] == te_labels_true[i]),
        }
        for pair_key, pv in pair_prob_store.items():
            row[pair_key] = float(pv[i])
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(out_dir, "predictions_test_ovo.csv"), index=False)

    mis = pred_df[pred_df["correct"] == 0].sort_values("confidence", ascending=True)
    mis.to_csv(os.path.join(out_dir, "misclassified_test_ovo.csv"), index=False)
    unc = pred_df.sort_values("confidence", ascending=True).head(max(1, len(pred_df) // 5))
    unc.to_csv(os.path.join(out_dir, "uncertain_test_ovo.csv"), index=False)

    cm_ovo = confusion_matrix(te_labels_true, ensemble_preds, labels=list(range(len(CLASSES))))
    save_confusion_matrix(cm_ovo, CLASSES, out_dir, normalize=False, fname="confusion_matrix_ovo.png")
    save_confusion_matrix(cm_ovo, CLASSES, out_dir, normalize=True, fname="confusion_matrix_ovo_normalized.png")

    _save_report_fixed(te_labels_true, ensemble_preds, CLASSES, out_dir, "classification_report_ovo")

    macro_f1 = f1_score(te_labels_true, ensemble_preds, average="macro", zero_division=0)
    acc = accuracy_score(te_labels_true, ensemble_preds)
    logger.info(f"OVO Ensemble Test: Acc={acc:.4f} MacroF1={macro_f1:.4f}")

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {cls: float(f1_score(te_labels_true, ensemble_preds, average=None, zero_division=0, labels=list(range(len(CLASSES))))[i]) for i, cls in enumerate(CLASSES)},
    }
    return metrics, all_yolo_logs


def _save_report_fixed(labels, preds, classes, out_dir, base_name):
    report_str = classification_report(labels, preds, target_names=classes, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    with open(os.path.join(out_dir, f"{base_name}.txt"), "w") as f:
        f.write(report_str)
    with open(os.path.join(out_dir, f"{base_name}.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    return report_str, report_dict


# ==============================================================
# YOLO Log & Debug
# ==============================================================

def save_yolo_log(log_entries, out_dir):
    if not log_entries:
        empty_log = pd.DataFrame(columns=[
            "video_path", "filename", "spatial_mode", "used_yolo",
            "used_yolo_crop", "used_yolo_mask", "num_frames_total",
            "num_frames_processed_by_yolo", "num_frames_with_detection",
            "detection_rate", "mean_confidence", "min_confidence", "max_confidence",
            "num_frames_with_one_detection", "num_frames_with_two_or_more_detections",
            "fallback_frame_count", "fallback_video_level", "mean_box_area_ratio", "notes",
        ])
        empty_log.to_csv(os.path.join(out_dir, "yolo_crop_log.csv"), index=False)
        return
    df = pd.DataFrame(log_entries)
    df.to_csv(os.path.join(out_dir, "yolo_crop_log.csv"), index=False)
    logger.info(f"YOLO crop log saved: {len(df)} entries")


def save_debug_crops(debug_info_list, out_dir, max_videos):
    if not debug_info_list:
        return
    debug_dir = os.path.join(out_dir, "debug_yolo_crops")
    os.makedirs(debug_dir, exist_ok=True)
    for item in debug_info_list[:max_videos]:
        video_stem = Path(item["video_path"]).stem[:40]
        for frame_idx, frames_dict in item.get("frames", {}).items():
            for frame_type, frame_img in frames_dict.items():
                if isinstance(frame_img, np.ndarray):
                    fname = f"{video_stem}_frame_{frame_idx:03d}_{frame_type}.png"
                    fpath = os.path.join(debug_dir, fname)
                    if frame_img.ndim == 3 and frame_img.shape[2] == 3:
                        cv2.imwrite(fpath, cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(fpath, frame_img)


# ==============================================================
# Summary
# ==============================================================

def print_and_save_summary(args, dist_df, metrics, out_dir):
    lines = []
    lines.append("=" * 60)
    lines.append("JUDO CLASSIFICATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Data dir:  {args.data_dir}")
    lines.append(f"Output:    {out_dir}")
    lines.append(f"Model:     {args.model}")
    lines.append(f"Mode:      {args.mode}")
    lines.append(f"Spatial:   {args.spatial_mode}")
    lines.append("")
    lines.append("Dataset distribution:")
    for _, row in dist_df.iterrows():
        lines.append(f"  {row['class']}: total={row['total']} train={row['train']} val={row['val']} test={row['test']}")
    lines.append("")
    lines.append(f"Test Accuracy:  {metrics.get('accuracy', 0):.4f}")
    lines.append(f"Test Macro F1:  {metrics.get('macro_f1', 0):.4f}")
    lines.append(f"Best Val F1:    {metrics.get('best_val_f1', 'N/A')}")
    lines.append("")
    lines.append("F1 per class:")
    for cls, f1 in metrics.get("per_class_f1", {}).items():
        lines.append(f"  {cls}: {f1:.4f}")
    lines.append("")
    lines.append(f"Best model: {out_dir}/{args.mode}/best_model.pt")
    lines.append("=" * 60)

    summary_str = "\n".join(lines)
    print(summary_str)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_str)


# ==============================================================
# Config
# ==============================================================

def save_config(args, out_dir):
    cfg = vars(args).copy()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ==============================================================
# RUN ALL MODES
# ==============================================================

def _make_args_copy(args, overrides):
    """Return a shallow copy of args with specific fields overridden."""
    import copy
    a = copy.copy(args)
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _run_single(args, train_df, val_df, test_df, dist_df, device, active_classes=None):
    """Run one mode/spatial combination and return metrics."""
    free_gpu_memory()  # start each run with a clean GPU cache

    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args, args.output_dir)

    yolo_processor = None
    if args.spatial_mode != "full_frame":
        yolo_processor = YOLOProcessor(
            model_path=args.yolo_model_path,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            margin=args.yolo_margin,
            every_n_frames=args.yolo_every_n_frames,
            smoothing=parse_bool(args.yolo_smoothing),
            smoothing_alpha=args.yolo_smoothing_alpha,
            mask_mode=args.mask_background_mode,
            spatial_mode=args.spatial_mode,
            box_mode=getattr(args, "yolo_box_mode", "per_frame"),
        )
        yolo_processor.load_model()

    bad_videos = []
    all_yolo_logs = []

    try:
        if args.mode == "multiclass":
            metrics, yolo_logs = run_multiclass(args, train_df, val_df, test_df, device, yolo_processor, bad_videos,
                                                classes=active_classes)
        elif args.mode == "ovr":
            metrics, yolo_logs = run_ovr(args, train_df, val_df, test_df, device, yolo_processor, bad_videos)
        else:
            metrics, yolo_logs = run_ovo(args, train_df, val_df, test_df, device, yolo_processor, bad_videos)
    finally:
        # Always release GPU memory — even if the run failed mid-way
        free_gpu_memory()

    all_yolo_logs.extend(yolo_logs)
    save_yolo_log(all_yolo_logs, args.output_dir)

    bad_df = pd.DataFrame(bad_videos) if bad_videos else pd.DataFrame(columns=["video_path", "filename", "error"])
    bad_df.to_csv(os.path.join(args.output_dir, "bad_videos.csv"), index=False)

    # Save combined splits for traceability
    combined = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True)
    combined_cols = ["video_path", "filename", "class_name", "label"]
    if "source_id" in combined.columns:
        combined_cols.append("source_id")
    combined_cols.append("split")
    combined[combined_cols].to_csv(
        os.path.join(args.output_dir, "splits.csv"), index=False
    )

    print_and_save_summary(args, dist_df, metrics, args.output_dir)
    return metrics


def run_all(args, train_df, val_df, test_df, dist_df, device, active_classes=None):
    """Run multiclass, ovr, ovo and multiclass+yolo sequentially."""
    base_dir = args.output_dir
    runs = [
        {
            "label": "multiclass",
            "dir": os.path.join(base_dir, "run_multiclass"),
            "overrides": {"mode": "multiclass", "spatial_mode": "full_frame"},
            "needs_yolo": False,
        },
        {
            "label": "ovr",
            "dir": os.path.join(base_dir, "run_ovr"),
            "overrides": {"mode": "ovr", "spatial_mode": "full_frame"},
            "needs_yolo": False,
        },
        {
            "label": "ovo",
            "dir": os.path.join(base_dir, "run_ovo"),
            "overrides": {"mode": "ovo", "spatial_mode": "full_frame"},
            "needs_yolo": False,
        },
        {
            "label": "multiclass_yolo",
            "dir": os.path.join(base_dir, "run_multiclass_yolo"),
            "overrides": {"mode": "multiclass", "spatial_mode": "yolo_union_crop"},
            "needs_yolo": True,
        },
    ]

    all_metrics = {}

    for run in runs:
        label = run["label"]
        run_dir = run["dir"]
        needs_yolo = run["needs_yolo"]

        # Check YOLO availability before attempting
        if needs_yolo:
            if not ULTRALYTICS_AVAILABLE:
                logger.warning(f"Skipping {label}: ultralytics not installed")
                all_metrics[label] = {"skipped": True, "reason": "ultralytics_not_installed"}
                continue
            if not os.path.exists(args.yolo_model_path):
                msg = f"YOLO model not found: {args.yolo_model_path}"
                if args.skip_yolo_if_missing:
                    logger.warning(f"Skipping {label}: {msg}")
                    all_metrics[label] = {"skipped": True, "reason": "yolo_model_not_found"}
                    continue
                else:
                    logger.error(f"Skipping {label}: {msg}. Use --skip_yolo_if_missing to ignore.")
                    all_metrics[label] = {"skipped": True, "reason": "yolo_model_not_found"}
                    continue

        overrides = dict(run["overrides"])
        overrides["output_dir"] = run_dir
        run_args = _make_args_copy(args, overrides)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"STARTING RUN: {label.upper()}")
        logger.info(f"  mode={run_args.mode}  spatial={run_args.spatial_mode}")
        logger.info(f"  output_dir={run_dir}")
        logger.info("=" * 60)

        fix_seeds(args.seed)
        try:
            metrics = _run_single(run_args, train_df, val_df, test_df, dist_df, device,
                                  active_classes=active_classes)
            all_metrics[label] = metrics
        except Exception as e:
            logger.error(f"Run {label} failed: {e}", exc_info=True)
            all_metrics[label] = {"error": str(e)}
            free_gpu_memory()  # release memory so next run can start

    # Save comparison table
    _save_all_comparison(all_metrics, base_dir)
    return all_metrics


def run_compare_spatial(args, train_df, val_df, test_df, dist_df, device, active_classes):
    """Run full_frame and yolo_union_crop with the same splits and compare results."""
    base_dir = args.output_dir
    runs = [
        {
            "label": "full_frame",
            "dir": os.path.join(base_dir, "full_frame"),
            "overrides": {"spatial_mode": "full_frame"},
            "needs_yolo": False,
        },
        {
            "label": "yolo_union_crop",
            "dir": os.path.join(base_dir, "yolo_union_crop"),
            "overrides": {"spatial_mode": "yolo_union_crop"},
            "needs_yolo": True,
        },
    ]

    all_metrics = {}

    for run in runs:
        label = run["label"]
        run_dir = run["dir"]

        if run["needs_yolo"]:
            if not ULTRALYTICS_AVAILABLE:
                logger.warning(f"Skipping {label}: ultralytics not installed")
                all_metrics[label] = {"skipped": True, "reason": "ultralytics_not_installed"}
                continue
            if not os.path.exists(args.yolo_model_path):
                msg = f"YOLO model not found: {args.yolo_model_path}"
                if args.skip_yolo_if_missing:
                    logger.warning(f"Skipping {label}: {msg}")
                    all_metrics[label] = {"skipped": True, "reason": "yolo_model_not_found"}
                    continue
                else:
                    logger.error(f"{msg}. Use --skip_yolo_if_missing to skip.")
                    all_metrics[label] = {"skipped": True, "reason": "yolo_model_not_found"}
                    continue

        overrides = dict(run["overrides"])
        overrides["output_dir"] = run_dir
        run_args = _make_args_copy(args, overrides)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"COMPARE SPATIAL — {label.upper()}")
        logger.info(f"  spatial={run_args.spatial_mode}  output_dir={run_dir}")
        logger.info("=" * 60)

        fix_seeds(args.seed)
        try:
            metrics = _run_single(run_args, train_df, val_df, test_df, dist_df, device,
                                  active_classes=active_classes)
            all_metrics[label] = metrics
        except Exception as e:
            logger.error(f"Run {label} failed: {e}", exc_info=True)
            all_metrics[label] = {"error": str(e)}
            free_gpu_memory()

    _save_spatial_comparison(all_metrics, active_classes, base_dir)
    return all_metrics


def _save_spatial_comparison(all_metrics, active_classes, base_dir):
    """Save side-by-side comparison of full_frame vs yolo_union_crop."""
    rows = []
    for label, m in all_metrics.items():
        if "skipped" in m or "error" in m:
            row = {"run": label, "accuracy": "N/A", "macro_f1": "N/A", "note": m.get("reason", m.get("error", ""))}
        else:
            pcf = m.get("per_class_f1", {})
            row = {
                "run": label,
                "accuracy": round(m.get("accuracy", 0), 4),
                "macro_f1": round(m.get("macro_f1", 0), 4),
                "note": "",
            }
            for cls in active_classes:
                row[f"f1_{cls}"] = round(pcf.get(cls, 0), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    cmp_path = os.path.join(base_dir, "comparison_spatial.csv")
    df.to_csv(cmp_path, index=False)

    lines = ["=" * 70, "COMPARISON — full_frame vs yolo_union_crop", "=" * 70]
    lines.append(df.to_string(index=False))
    lines.append(f"\nSaved to: {cmp_path}")
    lines.append("=" * 70)
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(base_dir, "comparison_spatial.txt"), "w") as f:
        f.write(summary)

    try:
        valid = [(r["run"], float(r["macro_f1"])) for r in rows if r.get("macro_f1", "N/A") != "N/A"]
        if valid:
            labels_plot, f1_vals = zip(*valid)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels_plot, f1_vals, color=["#4C72B0", "#DD8452"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Macro F1 (test)")
            ax.set_title("Macro F1 — full_frame vs yolo_union_crop")
            for bar, val in zip(bars, f1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, "comparison_spatial.png"), dpi=100)
            plt.close()
    except Exception as e:
        logger.debug(f"Could not save spatial comparison chart: {e}")


def _save_all_comparison(all_metrics, base_dir):
    """Save a side-by-side comparison of all runs."""
    rows = []
    for label, m in all_metrics.items():
        if "skipped" in m or "error" in m:
            rows.append({
                "run": label,
                "accuracy": "N/A",
                "macro_f1": "N/A",
                "f1_sutemi_waza": "N/A",
                "f1_ashi_waza": "N/A",
                "f1_te_waza": "N/A",
                "note": m.get("reason", m.get("error", "")),
            })
        else:
            pcf = m.get("per_class_f1", {})
            rows.append({
                "run": label,
                "accuracy": round(m.get("accuracy", 0), 4),
                "macro_f1": round(m.get("macro_f1", 0), 4),
                "f1_sutemi_waza": round(pcf.get("sutemi_waza", 0), 4),
                "f1_ashi_waza": round(pcf.get("ashi_waza", 0), 4),
                "f1_te_waza": round(pcf.get("te_waza", 0), 4),
                "note": "",
            })

    df = pd.DataFrame(rows)
    cmp_path = os.path.join(base_dir, "comparison_all_runs.csv")
    df.to_csv(cmp_path, index=False)

    # Pretty-print to terminal and txt
    lines = ["=" * 70, "COMPARISON — ALL RUNS", "=" * 70]
    lines.append(df.to_string(index=False))
    lines.append("")
    lines.append(f"Saved to: {cmp_path}")
    lines.append("=" * 70)
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(base_dir, "comparison_all_runs.txt"), "w") as f:
        f.write(summary)

    # Bar chart of macro F1
    try:
        valid = [(r["run"], float(r["macro_f1"])) for r in rows if r["macro_f1"] != "N/A"]
        if valid:
            labels_plot, f1_vals = zip(*valid)
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(labels_plot, f1_vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(valid)])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Macro F1 (test)")
            ax.set_title("Macro F1 — All Runs Comparison")
            for bar, val in zip(bars, f1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(base_dir, "comparison_macro_f1.png"), dpi=100)
            plt.close()
    except Exception as e:
        logger.debug(f"Could not save comparison chart: {e}")


# ==============================================================
# MAIN
# ==============================================================

def main():
    args = get_args()
    fix_seeds(args.seed)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args, args.output_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            total_mb = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} ({total_mb:.0f} MiB)")
        if n_gpus > 1:
            logger.info(f"DataParallel will be used across {n_gpus} GPUs")
        suggest_memory_settings(device)

    # Scan dataset
    logger.info(f"Scanning dataset in: {args.data_dir}")
    records = scan_dataset(args.data_dir, recursive=getattr(args, "recursive_scan", False))
    if not records:
        logger.error(f"No valid video files found in {args.data_dir}")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.output_dir, "dataset_index.csv"), index=False)
    logger.info(f"Found {len(df)} videos:")
    for cls in CLASSES:
        n = (df["class_name"] == cls).sum()
        logger.info(f"  {cls}: {n}")

    # --exclude_videos: remocao manual (revisao de labels) antes de remap/balance/split
    if getattr(args, "exclude_videos", None):
        df = apply_exclude_list(df, args.exclude_videos, args.output_dir)

    # --binary_tachi: remap ashi_waza + te_waza -> tachi_waza, force multiclass
    active_classes = CLASSES
    if args.binary_tachi:
        if args.mode in ("ovr", "ovo"):
            logger.warning("--binary_tachi forces mode=multiclass; --mode=%s will be ignored", args.mode)
        if getattr(args, "all", False):
            logger.warning("--binary_tachi is not compatible with --all; --all will be ignored")
            args.all = False
        args.mode = "multiclass"
        active_classes = TACHI_CLASSES
        df["class_name"] = df["class_name"].apply(
            lambda c: "tachi_waza" if c in ("ashi_waza", "te_waza") else c
        )
        df["label"] = df["class_name"].map(TACHI_CLASS_TO_IDX)
        logger.info("binary_tachi mode: ashi_waza + te_waza -> tachi_waza (label 1), sutemi_waza -> label 0")
        for cls in TACHI_CLASSES:
            n = (df["class_name"] == cls).sum()
            logger.info(f"  {cls}: {n}")

    # --binary_ashi_te: drop sutemi_waza entirely, force multiclass over {ashi_waza, te_waza}
    if getattr(args, "binary_ashi_te", False):
        if args.mode in ("ovr", "ovo"):
            logger.warning("--binary_ashi_te forces mode=multiclass; --mode=%s will be ignored", args.mode)
        if getattr(args, "all", False):
            logger.warning("--binary_ashi_te is not compatible with --all; --all will be ignored")
            args.all = False
        args.mode = "multiclass"
        active_classes = ASHI_TE_CLASSES
        df = df[df["class_name"].isin(ASHI_TE_CLASSES)].reset_index(drop=True)
        df["label"] = df["class_name"].map(ASHI_TE_CLASS_TO_IDX)
        logger.info("binary_ashi_te mode: sutemi_waza dropped, ashi_waza (label 0) vs te_waza (label 1)")
        for cls in ASHI_TE_CLASSES:
            n = (df["class_name"] == cls).sum()
            logger.info(f"  {cls}: {n}")

    # Split
    good_df = df[df["status"] == "ok"].reset_index(drop=True)
    if len(good_df) == 0:
        logger.error("No readable videos found")
        sys.exit(1)

    # --balance_dataset: undersample to the minority class size BEFORE the split
    # (after the binary_tachi remap, so binary mode balances sutemi vs tachi and the
    # other modes balance the 3 original classes)
    if getattr(args, "balance_dataset", False):
        logger.info("balance_dataset: undersampling every class to the minority class size")
        good_df = balance_dataset_df(good_df, args.seed, args.output_dir)

    if getattr(args, "group_split", False):
        logger.info("group_split: split by SOURCE (StratifiedGroupKFold) — leakage-free")
        train_df, val_df, test_df, dist_df = make_group_splits(
            good_df, args.train_split, args.val_split, args.test_split, args.seed,
            args.output_dir, classes=active_classes, fold=args.group_fold,
        )
    else:
        train_df, val_df, test_df, dist_df = make_splits(
            good_df, args.train_split, args.val_split, args.test_split, args.seed, args.output_dir,
            classes=active_classes,
        )
    logger.info(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # --- Optical flow / two-stream (item 4) ---
    if getattr(args, "precompute_flow", False):
        all_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        precompute_flow_dataset(all_split_df, args)
        logger.info("precompute_flow finished; exiting without training.")
        return
    modality = getattr(args, "input_modality", "rgb")
    if modality != "rgb":
        if getattr(args, "all", False) or getattr(args, "compare_spatial", False) or args.mode != "multiclass":
            logger.error("--input_modality %s only supports mode=multiclass (includes --binary_tachi); "
                         "--all / --compare_spatial / ovr / ovo are not supported.", modality)
            sys.exit(1)
        if args.spatial_mode != "full_frame":
            logger.info("--input_modality %s + spatial_mode=%s: optical flow uses the SAME "
                        "YOLO box/crop as the RGB branch (cache with a dedicated tag).", modality, args.spatial_mode)

    # --all: run multiclass, ovr, ovo and multiclass+yolo sequentially
    if getattr(args, "all", False):
        run_all(args, train_df, val_df, test_df, dist_df, device, active_classes=active_classes)
        return

    # --compare_spatial: run full_frame and yolo_union_crop with the same splits
    if getattr(args, "compare_spatial", False):
        run_compare_spatial(args, train_df, val_df, test_df, dist_df, device, active_classes)
        return

    # Single run
    yolo_processor = None
    if args.spatial_mode != "full_frame":
        if not ULTRALYTICS_AVAILABLE:
            logger.error("ultralytics is not installed but spatial_mode is not full_frame. pip install ultralytics")
            sys.exit(1)
        yolo_processor = YOLOProcessor(
            model_path=args.yolo_model_path,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            margin=args.yolo_margin,
            every_n_frames=args.yolo_every_n_frames,
            smoothing=parse_bool(args.yolo_smoothing),
            smoothing_alpha=args.yolo_smoothing_alpha,
            mask_mode=args.mask_background_mode,
            spatial_mode=args.spatial_mode,
            box_mode=getattr(args, "yolo_box_mode", "per_frame"),
        )
        try:
            yolo_processor.load_model()
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    bad_videos = []
    all_yolo_logs = []

    if args.mode == "multiclass":
        metrics, yolo_logs = run_multiclass(
            args, train_df, val_df, test_df, device, yolo_processor, bad_videos,
            classes=active_classes,
        )
        all_yolo_logs.extend(yolo_logs)
    elif args.mode == "ovr":
        metrics, yolo_logs = run_ovr(args, train_df, val_df, test_df, device, yolo_processor, bad_videos)
        all_yolo_logs.extend(yolo_logs)
    elif args.mode == "ovo":
        metrics, yolo_logs = run_ovo(args, train_df, val_df, test_df, device, yolo_processor, bad_videos)
        all_yolo_logs.extend(yolo_logs)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    save_yolo_log(all_yolo_logs, args.output_dir)

    bad_df = pd.DataFrame(bad_videos) if bad_videos else pd.DataFrame(columns=["video_path", "filename", "error"])
    bad_df.to_csv(os.path.join(args.output_dir, "bad_videos.csv"), index=False)
    if bad_videos:
        logger.warning(f"Bad videos: {len(bad_videos)} (see bad_videos.csv)")

    print_and_save_summary(args, dist_df, metrics, args.output_dir)


# ==============================================================
# Entry point
# ==============================================================

if __name__ == "__main__":
    main()
