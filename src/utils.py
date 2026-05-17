"""
src/utils.py — MalnutriSense base utility functions.
 
Reusable helpers used across data loading, cleaning, modelling, and reporting.
This module has NO imports from other src/ modules — only stdlib and third-party.
Any src/ module can safely import from here without circular dependency risk.
 
Functions:
    timer()             — context manager that logs execution time
    save_dataframe()    — save a DataFrame to CSV with row-count validation
    load_dataframe()    — load a CSV with existence and shape checks
    profile_dataframe() — return a per-column profile dict for the cleaning log
    compute_class_weights() — compute XGBoost scale_pos_weight per label
    compute_prevalence()    — compute malnutrition prevalence from label columns
    format_number()     — format large integers for report output (e.g. 232,920)
    assert_columns_exist()  — raise with clear message if columns are missing
"""
 
import time
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
 
import pandas as pd
import numpy as np
 
# Module-level logger — uses the module name as the logger name.
# Messages appear as 'src.utils: ...' in pipeline.log.
logger = logging.getLogger(__name__)
 
 
# ── Context manager: execution timer ─────────────────────────────────────
@contextmanager
def timer(operation: str = 'Operation') -> Generator[None, None, None]:
    """
    Context manager that logs the wall-clock time of a code block.
 
    Usage:
        from src.utils import timer
        with timer('Loading NFHS-5 file'):
            df, meta = pyreadstat.read_dta(NFHS5_PATH, usecols=NFHS_COLS)
        # Prints: Loading NFHS-5 file completed in 47.3s
    """
    start = time.perf_counter()
    logger.info(f'{operation}...')
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f'{operation} completed in {elapsed:.1f}s')
 
 
# ── Save DataFrame to CSV ────────────────────────────────────────────────
def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    description: str = 'DataFrame',
) -> None:
    """
    Save a DataFrame to CSV with validation and logging.
 
    Creates any missing parent directories automatically.
    Logs the file path and row/column count on success.
    Raises ValueError if df is empty — empty files are almost always a bug.
 
    Args:
        df:          DataFrame to save.
        path:        Destination path (pathlib.Path or str).
        description: Human-readable name for log messages.
 
    Raises:
        ValueError: DataFrame is empty (0 rows).
    """
    path = Path(path)
    if df.empty:
        raise ValueError(
            f'Attempted to save empty DataFrame as {description}. '
            f'Check your cleaning pipeline — a filter may have removed all rows.'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        f'Saved {description}: {format_number(len(df))} rows x {df.shape[1]} cols → {path}'
    )
 
 
# ── Load DataFrame from CSV ──────────────────────────────────────────────
def load_dataframe(
    path: Path,
    description: str = 'DataFrame',
    min_rows: int = 1,
) -> pd.DataFrame:
    """
    Load a CSV file with existence and minimum-row validation.
 
    Args:
        path:        Path to the CSV file.
        description: Human-readable name for log messages and errors.
        min_rows:    Raise if the loaded DataFrame has fewer than this many rows.
                     Use min_rows=190_000 when loading the full NFHS-5 cleaned file.
 
    Returns:
        Loaded DataFrame.
 
    Raises:
        FileNotFoundError: File does not exist at path.
        ValueError:        Loaded DataFrame has fewer than min_rows rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'{description} not found at {path}. '
            'Run the cleaning pipeline (preprocessing.py) to generate it.'
        )
    df = pd.read_csv(path)
    if len(df) < min_rows:
        raise ValueError(
            f'{description} has only {format_number(len(df))} rows. '
            f'Expected at least {format_number(min_rows)}. '
            'The file may be truncated or the cleaning pipeline may have over-filtered.'
        )
    logger.info(
        f'Loaded {description}: {format_number(len(df))} rows x {df.shape[1]} cols ← {path}'
    )
    return df
 
 
# ── Column profile for the cleaning log ──────────────────────────────────
def profile_dataframe(df: pd.DataFrame) -> dict[str, dict]:
    """
    Return a per-column profile dictionary for use in the cleaning log.
 
    Computes, for each column:
      - dtype: the pandas data type
      - null_count: integer count of NaN/None values
      - null_pct: proportion of nulls (0.0 to 1.0)
      - min, max: for numeric columns (None for non-numeric)
      - nunique: number of distinct values
      - sample: first 3 non-null values as a list
 
    Returns:
        dict mapping column name → profile dict.
 
    Example:
        profile = profile_dataframe(df)
        hw70 = profile['HW70']
        print(hw70['null_pct'])   # e.g. 0.124 (12.4% missing)
    """
    profile = {}
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        is_numeric = pd.api.types.is_numeric_dtype(series)
        profile[col] = {
            'dtype':      str(series.dtype),
            'null_count': null_count,
            'null_pct':   round(null_count / max(len(series), 1), 4),
            'min':        float(series.min())    if is_numeric and not series.isnull().all() else None,
            'max':        float(series.max())    if is_numeric and not series.isnull().all() else None,
            'nunique':    int(series.nunique()),
            'sample':     series.dropna().head(3).tolist(),
        }
    return profile
 
 
# ── Class imbalance statistics ────────────────────────────────────────────
def compute_class_weights(
    df: pd.DataFrame,
    label_cols: list[str],
) -> dict[str, float]:
    """
    Compute XGBoost scale_pos_weight for each binary label column.
 
    scale_pos_weight = (number of negative examples) / (number of positive examples)
    This corrects for class imbalance — malnutrition labels are typically 20–35%
    positive, so the model would otherwise predict 'healthy' for everything.
 
    Args:
        df:         DataFrame containing the binary label columns.
        label_cols: List of column names containing 0/1 labels.
 
    Returns:
        dict mapping label column name → scale_pos_weight float.
 
    Example:
        weights = compute_class_weights(df, ['stunted','underweight','wasted'])
        # {'stunted': 1.82, 'underweight': 2.12, 'wasted': 4.24}
        xgb = XGBClassifier(scale_pos_weight=weights['stunted'])
    """
    weights = {}
    for col in label_cols:
        if col not in df.columns:
            raise KeyError(f"Label column '{col}' not found in DataFrame.")
        pos = int(df[col].sum())
        neg = int((df[col] == 0).sum())
        if pos == 0:
            raise ValueError(
                f"Column '{col}' has zero positive examples. "
                "Check your label creation logic."
            )
        weights[col] = round(neg / pos, 4)
        logger.info(f'{col}: {pos:,} positive | {neg:,} negative | scale_pos_weight={weights[col]}')
    return weights
 
 
# ── Prevalence calculation ────────────────────────────────────────────────
def compute_prevalence(
    df: pd.DataFrame,
    label_cols: list[str],
) -> dict[str, float]:
    """
    Compute the proportion of positive (malnourished) cases per label column.
 
    Used to validate that cleaning produced the expected prevalence rates
    (stunting ~35.5%, underweight ~32.1%, wasting ~19.3% per NFHS-5 report).
 
    Args:
        df:         DataFrame with binary label columns.
        label_cols: Column names to compute prevalence for.
 
    Returns:
        dict mapping label name → prevalence as a float (0.0 to 1.0).
 
    Example:
        prev = compute_prevalence(df, ['stunted','underweight','wasted'])
        print(f'Stunting: {prev["stunted"]:.1%}')  # e.g. 35.2%
    """
    prevalence = {}
    for col in label_cols:
        if col not in df.columns:
            raise KeyError(f"Label column '{col}' not found in DataFrame.")
        prev = round(float(df[col].mean()), 4)
        prevalence[col] = prev
        logger.info(f'{col} prevalence: {prev:.1%}')
    return prevalence
 
 
# ── Number formatting for reports ─────────────────────────────────────────
def format_number(n: int | float, decimals: int = 0) -> str:
    """
    Format a number with thousands separators for report output.
 
    Args:
        n:        Number to format.
        decimals: Decimal places for floats (default 0 = integer format).
 
    Examples:
        format_number(232920)       → '232,920'
        format_number(0.355, 1)     → '35.5%'  (note: no % — add manually)
        format_number(1234567.89, 2) → '1,234,567.89'
    """
    if decimals == 0:
        return f'{int(n):,}'
    return f'{n:,.{decimals}f}'
 
 
# ── Column existence guard ─────────────────────────────────────────────────
def assert_columns_exist(
    df: pd.DataFrame,
    required: list[str],
    context: str = 'DataFrame',
) -> None:
    """
    Raise a descriptive KeyError if any required columns are missing.
 
    Use at the start of any function that accesses specific columns to get
    a clear error message instead of an obscure KeyError later.
 
    Args:
        df:       DataFrame to check.
        required: List of column names that must be present.
        context:  Human-readable name for the error message.
 
    Raises:
        KeyError: With a list of all missing columns.
 
    Example:
        assert_columns_exist(df, ['HAZ','WAZ','WHZ'], 'cleaned NFHS-5')
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f'{context} is missing required columns: {missing}\n'
            f'Available columns: {list(df.columns)}'
        )
 
