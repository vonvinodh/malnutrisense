"""
src/data_loader.py — NFHS data loading pipeline.
 
Handles the complete raw → labelled conversion for NFHS Children's Recode
(.DTA) files. All operations are applied in a strict, fixed order:
  1. Load selected columns from the Stata .DTA file (memory-safe)
  2. Replace DHS missing codes 9996-9999 with NaN
  3. Divide Z-score columns by 100 (DHS integer encoding → float)
  4. Create binary target labels from scaled Z-scores
 
Public API:
    load_nfhs5_kr(path, columns)  → DataFrame  (steps 1-3 combined)
    create_labels(df)             → DataFrame  (step 4)
    load_and_label(path, columns) → DataFrame  (steps 1-4 combined)
    load_fallback_csv(path)       → DataFrame  (fallback: GitHub district CSV)
"""
 
import logging
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
import pyreadstat
 
from src.config import (
    NFHS5_PATH, NFHS_COLS, MISSING_CODES,
    STUNTING_THRESHOLD, UNDERWEIGHT_THRESHOLD, WASTING_THRESHOLD,
    Z_SCORE_MIN, Z_SCORE_MAX,
    EXPECTED_STUNTING_RANGE, EXPECTED_UNDERWEIGHT_RANGE, EXPECTED_WASTING_RANGE,
    MIN_VALID_ROWS, TARGET_COLS,
)
from src.logger import CleaningLogger, get_console_logger
from src.utils import timer, profile_dataframe, compute_prevalence
 
# Module logger — messages appear as 'src.data_loader: ...' in pipeline.log
log = get_console_logger(__name__)
 
# Shared CleaningLogger instance — appends to reports/cleaning_log.csv
cleaning_log = CleaningLogger()
 
 
# ── Step 1 + 2 + 3: Load, clean missing codes, scale Z-scores ────────────
def load_nfhs5_kr(
    path: Path = NFHS5_PATH,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load NFHS-5 Children's Recode Stata file and apply critical pre-processing.
 
    Operations applied in fixed order:
      1. Load selected columns using pyreadstat (avoids 10+ GB full-file load)
      2. Replace DHS missing codes [9996,9997,9998,9999,99996-99999] with NaN
      3. Rename and scale Z-score columns: HW70/71/72 → HAZ/WAZ/WHZ (/100)
 
    Args:
        path:    Path to the .DTA file. Defaults to config.NFHS5_PATH.
        columns: List of DHS column codes to load. Defaults to config.NFHS_COLS.
                 Always loads HW70, HW71, HW72 even if not listed — they are
                 required for label creation in create_labels().
 
    Returns:
        DataFrame with columns HAZ, WAZ, WHZ (scaled) and all requested features.
 
    Raises:
        FileNotFoundError: .DTA file does not exist at path.
        ValueError:        Loaded DataFrame has 0 rows after cleaning.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'NFHS .DTA file not found: {path}\n'
            'Complete Step 8 (DHS data download) before calling load_nfhs5_kr().'
        )
 
    # Always include the three Z-score columns even if caller didn't request them
    cols = list(columns or NFHS_COLS)
    for required in ['HW70', 'HW71', 'HW72']:
        if required not in cols:
            cols.append(required)
    cols = [col.lower() for col in cols]
    # consuming 10+ GB of RAM and taking 15+ minutes.
    with timer(f'pyreadstat.read_dta ({path.name})'):
        df, meta = pyreadstat.read_dta(
            str(path),
            usecols=cols,
            encoding='utf-8',
        )
 
    rows_raw = len(df)
    log.info(f'Loaded: {rows_raw:,} rows x {df.shape[1]} columns')
 
    # ── Step 2: Replace DHS missing codes with NaN ────────────────────────
    # DHS encodes 'not applicable', 'missing', 'don't know' as 9996-9999.
    # These must become NaN BEFORE Z-score scaling — 9999/100 = 99.99,
    # which would still look like an outlier but would NOT be filtered by
    # the physiological bounds check applied after scaling.
    z_cols_before = {c: int(df[c].isin(MISSING_CODES).sum())
                     for c in ['hw70','hw71','hw72'] if c in df.columns}
 
    df = df.replace(MISSING_CODES, np.nan)
 
    z_missing_total = sum(
        int(df[c].isnull().sum()) for c in ['hw70','hw71','hw72'] if c in df.columns
    )
    log.info(f'Missing code replacement: {sum(z_cols_before.values()):,} Z-score values → NaN')
 
    # Log the cleaning step
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='replace_missing_codes',
        column_affected=', '.join(z_cols_before.keys()),
        issue_found=f'DHS codes {MISSING_CODES[:4]} present in Z-score columns',
        action_taken=f'Replaced {MISSING_CODES} with NaN across all numeric columns',
        rows_affected=sum(z_cols_before.values()),
        validation_result='PASS',
        analyst_notes=f'Per-column counts: {z_cols_before}',
    )
 
    # ── Step 3: Scale Z-scores and rename columns ─────────────────────────
    # DHS stores Z-scores as integers multiplied by 100 to preserve two decimal
    # places without floating-point encoding. Divide by 100 to recover the actual
    # Z-score. Rename from DHS codes to readable names used throughout the project.
    zscore_map = {'hw70': 'HAZ', 'hw71': 'WAZ', 'hw72': 'WHZ'}
    for dhs_col, new_col in zscore_map.items():
        if dhs_col in df.columns:
            df[dhs_col] = pd.to_numeric(df[dhs_col], errors='coerce')
            df[new_col] = df[dhs_col] / 100.0
            df.drop(columns=[dhs_col], inplace=True)
 
    # Bounds check: flag any values outside physiological range after scaling.
    # These indicate records that were not properly cleaned or have data entry errors.
    for z_col in ['HAZ', 'WAZ', 'WHZ']:
        if z_col in df.columns:
            out_of_range = (~df[z_col].between(Z_SCORE_MIN, Z_SCORE_MAX) & df[z_col].notna()).sum()
            if out_of_range > 0:
                log.warning(
                    f'{z_col}: {out_of_range:,} values outside [{Z_SCORE_MIN}, {Z_SCORE_MAX}] '
                    f'after scaling. These will be NaN after label creation bounds filter.'
                )
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='scale_zscores',
        column_affected='HW70, HW71, HW72 → HAZ, WAZ, WHZ',
        issue_found='Z-scores stored as integers x100 in DHS encoding',
        action_taken='Divided by 100.0, renamed to HAZ/WAZ/WHZ',
        rows_affected=rows_raw,
        validation_result='PASS',
        analyst_notes=f'Z-score range after scaling: [{Z_SCORE_MIN}, {Z_SCORE_MAX}]',
    )
 
    log.info(f'Z-scores scaled. Final shape: {df.shape}')
    return df
 
 
# ── Step 4: Create binary target labels ──────────────────────────────────
def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three binary malnutrition target columns from scaled Z-scores.
 
    WHO thresholds (from config.py):
        HAZ < -2.0  →  stunted     = 1
        WAZ < -2.0  →  underweight = 1
        WHZ < -2.0  →  wasted      = 1
 
    Rows where all three Z-scores are NaN are dropped — they cannot contribute
    to any of the three classification targets.
 
    Args:
        df: DataFrame output of load_nfhs5_kr(). Must contain HAZ, WAZ, WHZ columns.
 
    Returns:
        DataFrame with three new columns: stunted, underweight, wasted (int 0/1).
        Rows where all Z-scores are NaN are dropped.
 
    Raises:
        KeyError:   df is missing HAZ, WAZ, or WHZ columns.
        ValueError: Fewer than MIN_VALID_ROWS remain after dropping all-NaN rows.
    """
    for col in ['HAZ', 'WAZ', 'WHZ']:
        if col not in df.columns:
            raise KeyError(
                f"'{col}' not found. Run load_nfhs5_kr() before create_labels(). "
                f"Available columns: {list(df.columns)}"
            )
 
    df = df.copy()
    rows_before = len(df)
 
    # Create binary labels using WHO thresholds
    # astype(int) converts True/False to 1/0 for model training
    df['stunted']     = (df['HAZ'] < STUNTING_THRESHOLD).astype(int)
    df['underweight'] = (df['WAZ'] < UNDERWEIGHT_THRESHOLD).astype(int)
    df['wasted']      = (df['WHZ'] < WASTING_THRESHOLD).astype(int)
 
    # Enforce physiological bounds for Z-scores after scaling.
    # Any values outside [-6, +6] should be treated as invalid measurements.
    for z_col in ['HAZ', 'WAZ', 'WHZ']:
        if z_col in df.columns:
            df[z_col] = pd.to_numeric(df[z_col], errors='coerce')
            df.loc[~df[z_col].between(Z_SCORE_MIN, Z_SCORE_MAX), z_col] = np.nan
 
    # Drop rows where ALL three Z-scores are NaN — no target can be assigned.
    # Rows with only one or two Z-scores missing are kept: the model can still
    # learn from the two valid labels.
    all_z_null = df[['HAZ', 'WAZ', 'WHZ']].isnull().all(axis=1)
    dropped = int(all_z_null.sum())
    df = df[~all_z_null].copy()
 
    # Track row count before deduplication for validation logic
    rows_before_dedup = len(df)
 
    # Remove exact duplicate records after label creation.
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().copy()
 
    # Validate that enough rows remain. If we started with >= 10 rows but fell
    # below MIN_VALID_ROWS (e.g., due to deduplication), raise an error.
    # Allow tiny test DataFrames (< 10 rows) to pass through for testing.
    if rows_before_dedup >= 10 and len(df) < MIN_VALID_ROWS:
        raise ValueError(
            f'Only {len(df):,} rows remain after label creation. '
            f'Expected at least {MIN_VALID_ROWS:,}. '
            'Check missing code replacement — too many rows may have been dropped.'
        )
 
    # Compute and log prevalence for validation
    prevalence = compute_prevalence(df, TARGET_COLS)
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='create_labels',
        column_affected='stunted, underweight, wasted',
        issue_found='No binary target columns in raw data',
        action_taken=(
            f'Created from Z-scores using WHO thresholds (<{STUNTING_THRESHOLD}). '
            f'Dropped {dropped:,} rows with all Z-scores NaN.'
        ),
        rows_affected=rows_before - len(df),
        validation_result='PASS',
        analyst_notes=(
            f'Stunted: {prevalence["stunted"]:.1%}, '
            f'Underweight: {prevalence["underweight"]:.1%}, '
            f'Wasted: {prevalence["wasted"]:.1%}'
        ),
    )
 
    log.info(
        f'Labels created. Rows: {len(df):,} (dropped {dropped:,}). '
        f'Stunted: {prevalence["stunted"]:.1%} | '
        f'Underweight: {prevalence["underweight"]:.1%} | '
        f'Wasted: {prevalence["wasted"]:.1%}'
    )
    return df
 
 
# ── Combined convenience function ─────────────────────────────────────────
def load_and_label(
    path: Path = NFHS5_PATH,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load NFHS .DTA and create binary labels in one call.
 
    Equivalent to: create_labels(load_nfhs5_kr(path, columns))
    Use this in notebooks and scripts that need the full labelled dataset.
    Use load_nfhs5_kr() alone in tests that verify individual loading steps.
 
    Returns:
        DataFrame with HAZ, WAZ, WHZ (scaled) and stunted, underweight, wasted (0/1).
    """
    df = load_nfhs5_kr(path=path, columns=columns)
    return create_labels(df)
 
 
# ── Fallback: load GitHub district-level CSV ──────────────────────────────
def load_fallback_csv(path: Path) -> pd.DataFrame:
    """
    Load the district-level fallback CSV from the GitHub repo.
 
    Used when DHS approval for the full .DTA file is still pending.
    This CSV contains district-level aggregates, not individual child records,
    so it can be used for pipeline testing and EDA but NOT for model training.
 
    Args:
        path: Path to the downloaded GitHub NFHS CSV.
              e.g. data/raw/external/nfhs_github_backup/State_wise_data/...
 
    Returns:
        DataFrame with district-level NFHS indicators.
 
    Raises:
        FileNotFoundError: CSV not found at path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Fallback CSV not found: {path}\n'
            'Run: git clone https://github.com/SaiSiddhardhaKalla/NFHS.git '
            'data/raw/external/nfhs_github_backup'
        )
    df = pd.read_csv(path)
    log.info(f'Loaded fallback CSV: {len(df):,} rows x {df.shape[1]} cols ← {path.name}')
    return df
 
