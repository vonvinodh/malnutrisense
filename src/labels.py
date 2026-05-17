"""
src/labels.py — Binary malnutrition label pipeline.
 
Standalone module for all label creation, validation, and audit logging.
Separated from data_loader.py so label thresholds and audit logic can
change independently of the file loading code.
 
WHO thresholds (from src/config.py):
    HAZ < -2.0  →  stunted     = 1
    WAZ < -2.0  →  underweight = 1
    WHZ < -2.0  →  wasted      = 1
 
Public API:
    apply_who_thresholds(df)   → DataFrame  (creates binary label columns)
    audit_labels(df)           → dict        (prevalence + class weights)
    validate_label_integrity(df) → bool      (runs all label quality checks)
"""
 
import pandas as pd
import numpy as np
 
from src.config import (
    STUNTING_THRESHOLD, UNDERWEIGHT_THRESHOLD, WASTING_THRESHOLD,
    EXPECTED_STUNTING_RANGE, EXPECTED_UNDERWEIGHT_RANGE, EXPECTED_WASTING_RANGE,
    TARGET_COLS, MIN_VALID_ROWS,
)
from src.logger import CleaningLogger, ValidationLogger, get_console_logger
from src.utils import compute_class_weights, compute_prevalence, format_number
 
log          = get_console_logger(__name__)
cleaning_log = CleaningLogger()
vlog         = ValidationLogger()
 
 
# ── Core label creation: Z < -2.0 ────────────────────────────────────────
def apply_who_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create three binary malnutrition label columns using WHO thresholds.
 
    Threshold logic (strictly less than — not less than or equal to):
        HAZ < STUNTING_THRESHOLD    (-2.0) → stunted     = 1
        WAZ < UNDERWEIGHT_THRESHOLD (-2.0) → underweight = 1
        WHZ < WASTING_THRESHOLD     (-2.0) → wasted      = 1
 
    When the corresponding Z-score is NaN (measurement not available),
    the label column retains NaN — it is NOT imputed to 0.
    Rows with NaN labels are excluded from training for that phenotype
    using XGBoost's native missing value handling.
 
    Args:
        df: DataFrame containing HAZ, WAZ, WHZ columns (scaled float values).
 
    Returns:
        DataFrame with stunted, underweight, wasted columns added.
        Returns a copy — original DataFrame is not modified.
 
    Raises:
        KeyError: One or more of HAZ, WAZ, WHZ not present in df.
    """
    for col in ['HAZ', 'WAZ', 'WHZ']:
        if col not in df.columns:
            raise KeyError(
                f"'{col}' not found. Call data_loader.load_nfhs5_kr() first. "
                f"Available columns: {list(df.columns)[:10]}..."
            )
 
    df = df.copy()
 
    # Apply WHO thresholds — pd.Series comparison propagates NaN naturally.
    # (NaN < -2.0) evaluates to False in pandas, so astype(int) would convert
    # NaN to 0 incorrectly. We use np.where to preserve NaN explicitly.
    df['stunted'] = np.where(
        df['HAZ'].isnull(), np.nan,
        (df['HAZ'] < STUNTING_THRESHOLD).astype(float)
    )
    df['underweight'] = np.where(
        df['WAZ'].isnull(), np.nan,
        (df['WAZ'] < UNDERWEIGHT_THRESHOLD).astype(float)
    )
    df['wasted'] = np.where(
        df['WHZ'].isnull(), np.nan,
        (df['WHZ'] < WASTING_THRESHOLD).astype(float)
    )
 
    # Convert to nullable integer type so 0/1/NaN coexist without float dtype
    for col in TARGET_COLS:
        df[col] = df[col].astype(pd.Int8Dtype())
 
    log.info(
        f'WHO thresholds applied: stunted={STUNTING_THRESHOLD}, '
        f'underweight={UNDERWEIGHT_THRESHOLD}, wasted={WASTING_THRESHOLD}'
    )
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='apply_who_thresholds',
        column_affected='stunted, underweight, wasted',
        issue_found='No binary target columns in dataset',
        action_taken=(
            f'HAZ<{STUNTING_THRESHOLD}→stunted, '
            f'WAZ<{UNDERWEIGHT_THRESHOLD}→underweight, '
            f'WHZ<{WASTING_THRESHOLD}→wasted. '
            'NaN Z-scores preserved as NaN labels (not imputed).'
        ),
        rows_affected=len(df),
        validation_result='PASS',
        analyst_notes='WHO Child Growth Standards 2006. Strict < not <=.',
    )
    return df
 
 
# ── Label audit ──────────────────────────────────────────────────────────
def audit_labels(df: pd.DataFrame) -> dict:
    """
    Compute and log prevalence, class weights, and label statistics.
 
    Returns a dictionary with all audit metrics — useful for the paper
    Methods section and for setting XGBoost scale_pos_weight.
 
    Args:
        df: DataFrame with stunted, underweight, wasted columns.
 
    Returns:
        dict with keys: prevalence, class_weights, null_counts, total_rows.
    """
    audit: dict = {}
 
    # Prevalence (computed over non-null rows for each label)
    prevalence: dict[str, float] = {}
    for col in TARGET_COLS:
        if col in df.columns:
            valid = df[col].dropna()
            prevalence[col] = round(float(valid.mean()), 4)
 
    audit['prevalence']    = prevalence
    audit['total_rows']    = len(df)
    audit['null_counts']   = {c: int(df[c].isnull().sum()) for c in TARGET_COLS if c in df.columns}
 
    # Class weights for XGBoost (only computable for non-null, binary columns)
    df_nonnull = df[TARGET_COLS].dropna()
    if len(df_nonnull) > 0:
        audit['class_weights'] = compute_class_weights(
            df_nonnull.astype(int), TARGET_COLS
        )
    else:
        audit['class_weights'] = {}
 
    # Log the audit summary
    log.info('─── Label Audit ───')
    for col, prev in prevalence.items():
        log.info(f'  {col}: {prev:.1%} prevalence')
    for col, w in audit.get('class_weights', {}).items():
        log.info(f'  {col}: scale_pos_weight = {w}')
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='audit_labels',
        column_affected='stunted, underweight, wasted',
        issue_found='N/A — audit step',
        action_taken='Computed prevalence and class weights',
        rows_affected=len(df),
        validation_result='PASS',
        analyst_notes=str({
            'prevalence': prevalence,
            'class_weights': audit.get('class_weights', {}),
        }),
    )
    return audit
 
 
# ── Label integrity validation ────────────────────────────────────────────
def validate_label_integrity(df: pd.DataFrame) -> bool:
    """
    Run all label quality checks and write results to ValidationLogger.
 
    Checks performed:
      V1: All three target columns exist
      V2: Labels contain only 0, 1, or NaN (no other values)
      V3: Stunting prevalence within EXPECTED_STUNTING_RANGE
      V4: Underweight prevalence within EXPECTED_UNDERWEIGHT_RANGE
      V5: Wasting prevalence within EXPECTED_WASTING_RANGE
      V6: Minimum row count met
      V7: No row has valid Z-score but NaN label (logic consistency)
 
    Args:
        df: DataFrame after apply_who_thresholds().
 
    Returns:
        True if all checks pass, False if any fail.
    """
    vlog.start_section('Label integrity validation post-apply_who_thresholds')
    all_passed = True
 
    # V1 — target columns present
    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if not missing_targets:
        vlog.pass_('V1: All target columns present', str(TARGET_COLS))
    else:
        vlog.fail_('V1: Missing target columns', str(missing_targets))
        all_passed = False
 
    # V2 — binary values only (0, 1, or NaN — using nullable int)
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        # dropna() removes NaN before checking unique values
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, pd.NA}):
            vlog.pass_(f'V2: {col} is binary', f'Values: {unique_vals}')
        else:
            vlog.fail_(f'V2: {col} has non-binary values', str(unique_vals))
            all_passed = False
 
    # V3-V5 — prevalence within NFHS-5 expected ranges
    checks = [
        ('stunted',     EXPECTED_STUNTING_RANGE,    'V3'),
        ('underweight', EXPECTED_UNDERWEIGHT_RANGE, 'V4'),
        ('wasted',      EXPECTED_WASTING_RANGE,     'V5'),
    ]
    for col, (lo, hi), code in checks:
        if col not in df.columns:
            vlog.fail_(f'{code}: {col} column missing', 'Cannot compute prevalence')
            all_passed = False
            continue
        prev = float(df[col].dropna().mean())
        detail = f'{prev:.1%} (expected {lo:.0%}–{hi:.0%})'
        if lo <= prev <= hi:
            vlog.pass_(f'{code}: {col} prevalence in expected range', detail)
        else:
            vlog.fail_(f'{code}: {col} prevalence out of range', detail)
            all_passed = False
 
    # V6 — row count
    if len(df) >= MIN_VALID_ROWS:
        vlog.pass_('V6: Row count meets minimum', format_number(len(df)))
    else:
        vlog.fail_('V6: Row count below minimum',
                   f'{format_number(len(df))} < {format_number(MIN_VALID_ROWS)}')
        all_passed = False
 
    # V7 — no row has valid Z-score but NaN label
    pairs = [('HAZ','stunted'), ('WAZ','underweight'), ('WHZ','wasted')]
    logic_errors = 0
    for z_col, label_col in pairs:
        if z_col not in df.columns or label_col not in df.columns:
            continue
        # A row with non-NaN Z-score must have a non-NaN label
        bad = (df[z_col].notna() & df[label_col].isnull()).sum()
        logic_errors += int(bad)
    if logic_errors == 0:
        vlog.pass_('V7: Label-Z-score consistency', 'No orphaned Z-scores')
    else:
        vlog.fail_('V7: Z-score/label mismatch', f'{logic_errors:,} rows affected')
        all_passed = False
 
    result = vlog.finish_section()
    log.info(f'Label validation: {"PASSED" if result else "FAILED"}')
    return result
 
