#!/usr/bin/env python3
"""
Enhanced ETL Module for Table‑Tennis Swing IMU Dataset
-----------------------------------------------------
Implements the lecture‑aligned cleaning pipeline discussed in chat:
* Unit conversion (LSB → physical units)
* Median despike + 5th‑order Butterworth low‑pass filter
* Linear interpolation for small numeric gaps before NaN drop
* One‑hot encoding of categorical buckets (age, playYears, height, weight)
* Strict outlier removal (‖a‖ > 16 g after scaling)
* Deduplication and player‑ID retention for leakage‑free CV
The resulting CSV is ready for GroupStratifiedKFold + scikit‑learn Pipelines.
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

###############################################################################
# Logging helpers
###############################################################################

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Create logs directory (if missing) and configure root logger."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/etl.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)

###############################################################################
# IMU helpers
###############################################################################

ACCEL_COLS = [
    "ax_mean",
    "ay_mean",
    "az_mean",
    "ax_var",
    "ay_var",
    "az_var",
    "ax_rms",
    "ay_rms",
    "az_rms",
    "a_max",
    "a_mean",
    "a_min",
]

GYRO_COLS = [
    "gx_mean",
    "gy_mean",
    "gz_mean",
    "gx_var",
    "gy_var",
    "gz_var",
    "gx_rms",
    "gy_rms",
    "gz_rms",
    "g_max",
    "g_mean",
    "g_min",
]

IMU_COLS = ACCEL_COLS + GYRO_COLS


def convert_imu_units(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Convert raw 16‑bit LSB values to g and °/s."""
    logger.info("Converting IMU units to physical scales …")

    # ±2 g range → scale 2/32768
    for col in ACCEL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float) * (2.0 / 32768.0)

    # ±250 °/s range → scale 250/32768
    for col in GYRO_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float) * (250.0 / 32768.0)

    return df

###############################################################################
# Filtering & smoothing
###############################################################################


def median_despike(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace single‑frame spikes with 3‑point rolling median."""
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .rolling(window=3, center=True, min_periods=1)
                .median()
                .fillna(method="bfill")
                .fillna(method="ffill")
            )
    return df


def butter_lowpass(data: np.ndarray, fs: float = 200.0, cutoff: float = 20.0, order: int = 5) -> np.ndarray:
    """Return low‑pass filtered copy of 1‑D vector using zero‑phase filtfilt."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)
    return filtfilt(b, a, data, method="gust")


def smooth_imu(df: pd.DataFrame, cols: list[str], logger: logging.Logger) -> pd.DataFrame:
    logger.info("Applying median despike + Butterworth LPF …")
    df = median_despike(df, cols)
    for col in cols:
        if col in df.columns:
            df[col] = butter_lowpass(df[col].values)
    return df

###############################################################################
# Data quality helpers
###############################################################################


def interpolate_numeric(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Linearly interpolate isolated numeric NaNs then return df."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_before = df[num_cols].isna().sum().sum()
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    n_after = df[num_cols].isna().sum().sum()
    logger.info(f"Interpolated {int(n_before - n_after)} numeric NaNs (remaining: {int(n_after)})")
    return df


def filter_outliers(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Drop rows where acceleration magnitude exceeds sensor range (16 g)."""
    logger.info("Filtering extreme outliers (‖a‖ > 16 g) …")

    if {"ax_mean", "ay_mean", "az_mean"}.issubset(df.columns):
        accel_mag = np.sqrt(df["ax_mean"] ** 2 + df["ay_mean"] ** 2 + df["az_mean"] ** 2)
    else:
        logger.warning("Acceleration components not all present; skipping outlier filter.")
        return df

    mask = accel_mag <= 16.0
    removed = (~mask).sum()
    logger.info(f"Removed {removed} rows exceeding ±16 g sensor range")
    return df[mask].copy()

###############################################################################
# Categorical handling
###############################################################################

CAT_COLS = ["age", "playYears", "height", "weight"]


def drop_unknown_categories(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove rows containing the placeholder '???' in any categorical bucket."""
    if not set(CAT_COLS).intersection(df.columns):
        return df

    mask_no_unknown = (df[CAT_COLS] != "???").all(axis=1)
    removed = (~mask_no_unknown).sum()
    logger.info(f"Dropped {removed} rows containing '???' categorical values")
    return df[mask_no_unknown].copy()


def one_hot_encode(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """One‑hot encode the four bucketised categorical columns."""
    present = [c for c in CAT_COLS if c in df.columns]
    if not present:
        return df
    logger.info(f"One‑hot encoding categorical columns: {present}")
    df = pd.get_dummies(df, columns=present, prefix=present, drop_first=True)
    return df

###############################################################################
# Column house‑keeping
###############################################################################

DROP_COLS = [
    "date",  # temporal metadata
    "fileindex",
    "teststage",
    "count",
]


def remove_unnecessary_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping non‑predictive columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df

###############################################################################
# Validation helpers
###############################################################################

def validate_target(df: pd.DataFrame, logger: logging.Logger) -> None:
    if "testmode" not in df.columns:
        raise ValueError("Target column 'testmode' missing after cleaning")
    expected = {0, 1, 2}
    found = set(df["testmode"].unique())
    if not found.issubset(expected):
        logger.warning(f"Unexpected class labels found: {found - expected}")
    logger.info("Target distribution: %s", df["testmode"].value_counts().to_dict())


def final_checks(df: pd.DataFrame, logger: logging.Logger) -> None:
    nan_total = df.isna().sum().sum()
    logger.info(f"Remaining NaNs in dataset: {nan_total}")
    if nan_total:
        logger.warning("Consider additional imputation before modelling.")

###############################################################################
# Main pipeline
###############################################################################


def process_data(input_csv: Path, output_csv: Path, logger: logging.Logger) -> None:
    logger.info("Loading raw CSV …")
    df = pd.read_csv(input_csv, low_memory=False)

    logger.info(f"Initial shape: {df.shape}")

    # Drop exact duplicates early
    before_dupes = len(df)
    df = df.drop_duplicates(ignore_index=True)
    logger.info(f"Dropped {before_dupes - len(df)} duplicate rows")

    # Convert + smooth IMU signals
    df = convert_imu_units(df, logger)
    df = smooth_imu(df, IMU_COLS, logger)

    # Interpolate numeric gaps then drop remaining NaNs later
    df = interpolate_numeric(df, logger)

    # Categorical cleaning & encoding
    df = drop_unknown_categories(df, logger)
    df = one_hot_encode(df, logger)

    # Remove leftover NaNs (now mostly categorical)
    before_nan = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before_nan - len(df)} rows still containing NaNs after interpolation")

    # Outlier filter
    df = filter_outliers(df, logger)

    # House‑keeping columns
    df = remove_unnecessary_columns(df, logger)

    # Validation
    validate_target(df, logger)
    final_checks(df, logger)

    # Ensure output dir exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved cleaned dataset → {output_csv} (rows: {len(df)}, cols: {df.shape[1]})")

###############################################################################
# CLI
###############################################################################


def main():
    parser = argparse.ArgumentParser(description="Clean table‑tennis IMU dataset for ML")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path for cleaned CSV")
    parser.add_argument(
        "--log‑level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logger = setup_logging(getattr(args, 'log‑level'))

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found → %s", input_path)
        sys.exit(1)

    process_data(input_path, output_path, logger)


if __name__ == "__main__":
    main()
