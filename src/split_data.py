#!/usr/bin/env python3
"""
Group‑Stratified Train/Val/Test Splitter
========================================
COMP4702 ‑ Machine Learning Assignment

This script creates **leakage‑proof** train / validation / test indices for the
Table‑Tennis Swing dataset.  It follows Week 5 lecture guidance:

* **Group isolation** – swings from the same player (`id`) never appear in
  multiple splits (prevents overly‑optimistic CV).
* **Class balance** – uses *StratifiedGroupKFold* so that each split has a
  similar `testmode` distribution (≈ stratified per player bucket).
* **Deterministic shuffling** – seeded RNG for reproducibility (Week 4 best‑
  practice).

Output: three JSON files (`train.json`, `val.json`, `test.json`) containing row
indices for direct use with a modelling pipeline.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------
# Config & helpers
# ----------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)

SEED = 123  # global default – override via CLI if desired


def first_two_folds_sgkf(df, group_col: str, target_col: str, seed: int = SEED):
    """Return indices for train / val / test using two SGKF rounds.

    1. **Round 1** – SGKF with *n_splits=5*  → pick one fold as *test*,
       concatenate the remaining four folds (*temp*).
    2. **Round 2** – SGKF on *temp* (n_splits=4) → pick one fold as *val*,
       remainder becomes *train*.

    Ratios ≈ 0.6 / 0.2 / 0.2, close to the 0.7 / 0.15 / 0.15 guideline but with
    perfect group stratification.
    """

    sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y = df[target_col]
    groups = df[group_col]

    # ── Round 1: choose the first yielded split as TEST ──────────────────────
    train_val_idx, test_idx = next(sgkf5.split(df, y, groups))
    df_temp = df.iloc[train_val_idx].copy()

    # ── Round 2 on the temp set for VAL ─────────────────────────────────────
    sgkf4 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    y_temp = df_temp[target_col]
    groups_temp = df_temp[group_col]
    train_idx_rel, val_idx_rel = next(sgkf4.split(df_temp, y_temp, groups_temp))

    # Map relative indices back to original dataframe
    train_idx = df_temp.iloc[train_idx_rel].index.to_numpy()
    val_idx = df_temp.iloc[val_idx_rel].index.to_numpy()

    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def save_splits(splits: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        path = output_dir / f"{name}.json"
        with open(path, "w") as fp:
            json.dump(idx, fp, indent=2)
        logger.info(f"Saved {len(idx):,} indices → {path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Create group‑stratified train/val/test splits for IMU data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Processed CSV (after ETL)")
    p.add_argument("--output_dir", required=True, help="Directory for JSON splits")
    p.add_argument("--group_col", default="id", help="Player/group column")
    p.add_argument("--target_col", default="testmode", help="Target label column")
    p.add_argument("--seed", type=int, default=SEED, help="RNG seed")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    # reproducible randomness
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    # ── Load data ───────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} rows from {args.input}")

    # Basic sanity
    for col in (args.group_col, args.target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # ── Create splits ───────────────────────────────────────────────────────
    splits = first_two_folds_sgkf(
        df, group_col=args.group_col, target_col=args.target_col, seed=args.seed
    )

    # quick diagnostics
    for split, idx in splits.items():
        sub = df.loc[idx, args.target_col]
        dist = sub.value_counts().sort_index()
        logger.info(
            f"{split.capitalize():5}  |  n={len(idx):5,}  |  class dist: "
            + ", ".join([f"{cls}:{cnt}" for cls, cnt in dist.items()])
        )

    # ── Persist ─────────────────────────────────────────────────────────────
    save_splits(splits, Path(args.output_dir))

    logger.info("✓ Splits created with group‑aware stratification")


if __name__ == "__main__":
    main()
