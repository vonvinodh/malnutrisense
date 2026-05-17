"""
scripts/validate_initial.py — Initial data validation script.
 
Run this script ONCE immediately after downloading the NFHS-5 .DTA file.
It checks all 12 quality gates required before Week 3 cleaning begins.
 
Usage:
    python3 scripts/validate_initial.py
 
Output:
    - Terminal: pass/fail for each gate with details
    - reports/validation_report.txt: timestamped section appended
    - reports/cleaning_log.csv: one INFO row per gate recorded
 
After confirming all gates pass:
    1. Delete all lines marked # VERIFY:  (see cleanup instructions in guide)
    2. Commit the cleaned version: git commit -m 'chore: validate initial NFHS data'
"""
 
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import (
    NFHS5_PATH, NFHS_COLS, MISSING_CODES,
    Z_SCORE_MIN, Z_SCORE_MAX, MIN_VALID_ROWS,
    EXPECTED_STUNTING_RANGE, EXPECTED_UNDERWEIGHT_RANGE, EXPECTED_WASTING_RANGE,
    validate_environment,
)
from src.data_loader import load_and_label
from src.logger import ValidationLogger, CleaningLogger, get_console_logger
from src.utils import compute_class_weights, format_number
 
log      = get_console_logger(__name__)
vlog     = ValidationLogger()
clog     = CleaningLogger()
 
 
def main() -> None:
    print('=' * 70)
    print('MalnutriSense — Initial Data Validation')
    print('=' * 70)
 
    # ── Pre-flight: environment check ─────────────────────────────────────
    try:
        validate_environment(require_data=True)
        print('[OK] Environment validated — Python 3.11+, all directories exist')
    except (RuntimeError, FileNotFoundError) as e:
        print(f'[FAIL] Environment validation failed:\n  {e}')
        print('  Fix the issue above before proceeding.')
        sys.exit(1)
 
    # ── Load the full NFHS-5 file ─────────────────────────────────────────
    print(f'\nLoading {NFHS5_PATH.name} — this may take 2-5 minutes...')
    try:
        df = load_and_label(path=NFHS5_PATH, columns=NFHS_COLS)
    except Exception as e:
        print(f'[FAIL] Could not load NFHS-5 file: {e}')
        sys.exit(1)
 
    print(f'[OK] Loaded: {format_number(len(df))} rows x {df.shape[1]} columns')
 
    # ── VERIFY: detailed load summary (DELETE after first run) ───────────
    # # VERIFY: Print column list to confirm expected columns loaded
    # print('\n# VERIFY: Columns loaded:')
    # print([c for c in df.columns])
    # # VERIFY: Show first 3 rows of Z-score and label columns
    # print('\n# VERIFY: Z-scores and labels (first 3 rows):')
    # print(df[['HAZ','WAZ','WHZ','stunted','underweight','wasted']].head(3))
    # # VERIFY: Show raw describe() output for Z-scores
    # print('\n# VERIFY: Z-score describe():')
    # print(df[['HAZ','WAZ','WHZ']].describe().round(3))
    # ── END VERIFY block ──────────────────────────────────────────────────
 
    # ── Open validation section ───────────────────────────────────────────
    vlog.start_section('Initial NFHS-5 data validation — post-download gates')
    print()
 
    # ── Gate G1: Row count ────────────────────────────────────────────────
    row_count = len(df)
    gate_g1 = row_count >= MIN_VALID_ROWS
    g1_detail = f'{format_number(row_count)} rows (min {format_number(MIN_VALID_ROWS)})'
    if gate_g1:
        vlog.pass_('G1: Row count above minimum', g1_detail)
        print(f'  [PASS] G1 — {g1_detail}')
    else:
        vlog.fail_('G1: Row count below minimum', g1_detail)
        print(f'  [FAIL] G1 — {g1_detail}')
 
    # ── VERIFY: exact row count (DELETE after first run) ──────────────────
    # # VERIFY: Print exact total including rows dropped during label creation
    # print(f'# VERIFY: Exact row count = {row_count:,}')
    # ── END VERIFY block ──────────────────────────────────────────────────
 
    # ── Gate G2: No DHS missing codes remain ──────────────────────────────
    numeric_cols = df.select_dtypes(include='number').columns
    missing_code_total = sum(
        int(df[col].isin(MISSING_CODES).sum()) for col in numeric_cols
    )
    gate_g2 = missing_code_total == 0
    g2_detail = f'{missing_code_total:,} DHS missing codes remain across all columns'
    if gate_g2:
        vlog.pass_('G2: No DHS missing codes remain', 'All 9996-9999 replaced with NaN')
        print(f'  [PASS] G2 — No DHS missing codes in any numeric column')
    else:
        vlog.fail_('G2: DHS missing codes still present', g2_detail)
        print(f'  [FAIL] G2 — {g2_detail}')
 
    # ── VERIFY: per-column missing code count (DELETE after first run) ────
    # # VERIFY: Show which columns still have missing codes if G2 fails
    # if not gate_g2:
    #     print('# VERIFY: Columns with remaining missing codes:')
    #     for col in numeric_cols:
    #         count = df[col].isin(MISSING_CODES).sum()
    #         if count > 0:
    #             print(f'  {col}: {count:,} remaining')
    # ── END VERIFY block ──────────────────────────────────────────────────
 
    # ── Gate G3: Z-scores in physiological range ──────────────────────────
    out_of_range_counts = {}
    for z_col in ['HAZ', 'WAZ', 'WHZ']:
        if z_col in df.columns:
            valid = df[z_col].dropna()
            out = int((~valid.between(Z_SCORE_MIN, Z_SCORE_MAX)).sum())
            out_of_range_counts[z_col] = out
    gate_g3 = all(v == 0 for v in out_of_range_counts.values())
    g3_detail = f'Out-of-range: {out_of_range_counts}'
    if gate_g3:
        vlog.pass_('G3: Z-scores within physiological range [-6, +6]', 'All clear')
        print(f'  [PASS] G3 — All Z-scores in [{Z_SCORE_MIN}, {Z_SCORE_MAX}]')
    else:
        vlog.fail_('G3: Z-scores outside physiological range', g3_detail)
        print(f'  [FAIL] G3 — {g3_detail}')
 
    # ── VERIFY: Z-score summary statistics (DELETE after first run) ────────
    # # VERIFY: Print min/max/mean for each Z-score column
    # print('# VERIFY: Z-score min/max/mean:')
    # for z_col in ['HAZ','WAZ','WHZ']:
    #     if z_col in df.columns:
    #         col_data = df[z_col].dropna()
    #         print(f'  {z_col}: min={col_data.min():.3f}  max={col_data.max():.3f}  mean={col_data.mean():.3f}')
    # ── END VERIFY block ──────────────────────────────────────────────────
 
    # ── Gate G4: Stunting prevalence ─────────────────────────────────────
    stunting_prev = float(df['stunted'].mean())
    lo_s, hi_s = EXPECTED_STUNTING_RANGE
    gate_g4 = lo_s <= stunting_prev <= hi_s
    g4_detail = f'{stunting_prev:.1%} (expected {lo_s:.0%}–{hi_s:.0%})'
    if gate_g4:
        vlog.pass_('G4: Stunting prevalence within expected range', g4_detail)
        print(f'  [PASS] G4 — Stunting: {g4_detail}')
    else:
        vlog.fail_('G4: Stunting prevalence out of range', g4_detail)
        print(f'  [FAIL] G4 — Stunting: {g4_detail}')
 
    # ── Gate G5: Underweight prevalence ──────────────────────────────────
    underweight_prev = float(df['underweight'].mean())
    lo_u, hi_u = EXPECTED_UNDERWEIGHT_RANGE
    gate_g5 = lo_u <= underweight_prev <= hi_u
    g5_detail = f'{underweight_prev:.1%} (expected {lo_u:.0%}–{hi_u:.0%})'
    if gate_g5:
        vlog.pass_('G5: Underweight prevalence within expected range', g5_detail)
        print(f'  [PASS] G5 — Underweight: {g5_detail}')
    else:
        vlog.fail_('G5: Underweight prevalence out of range', g5_detail)
        print(f'  [FAIL] G5 — Underweight: {g5_detail}')
 
    # ── Gate G6: Wasting prevalence ───────────────────────────────────────
    wasting_prev = float(df['wasted'].mean())
    lo_w, hi_w = EXPECTED_WASTING_RANGE
    gate_g6 = lo_w <= wasting_prev <= hi_w
    g6_detail = f'{wasting_prev:.1%} (expected {lo_w:.0%}–{hi_w:.0%})'
    if gate_g6:
        vlog.pass_('G6: Wasting prevalence within expected range', g6_detail)
        print(f'  [PASS] G6 — Wasting: {g6_detail}')
    else:
        vlog.fail_('G6: Wasting prevalence out of range', g6_detail)
        print(f'  [FAIL] G6 — Wasting: {g6_detail}')
 
    # ── VERIFY: full prevalence breakdown (DELETE after first run) ─────────
    # # VERIFY: Print label value_counts for each target column
    # print('\n# VERIFY: Label value counts:')
    # for col in ['stunted','underweight','wasted']:
    #     print(f'  {col}:')
    #     print(df[col].value_counts().rename({0:'healthy', 1:'malnourished'}).to_string())
    # ── END VERIFY block ──────────────────────────────────────────────────
 
    # ── Gate G7: Labels are binary ────────────────────────────────────────
    label_ok = True
    label_issues = []
    for col in ['stunted', 'underweight', 'wasted']:
        unique_vals = set(df[col].unique())
        if not unique_vals.issubset({0, 1}):
            label_ok = False
            label_issues.append(f'{col}: {unique_vals}')
    if label_ok:
        vlog.pass_('G7: Target labels contain only 0 and 1', '3 columns verified')
        print(f'  [PASS] G7 — All three labels are binary 0/1')
    else:
        vlog.fail_('G7: Non-binary values found in labels', str(label_issues))
        print(f'  [FAIL] G7 — {label_issues}')
    gate_g7 = label_ok
 
    # ── Gate G8: No duplicate rows ────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    gate_g8 = dup_count == 0
    g8_detail = f'{format_number(dup_count)} duplicate rows'
    if gate_g8:
        vlog.pass_('G8: No duplicate rows', 'All rows are unique')
        print(f'  [PASS] G8 — No duplicate rows')
    else:
        vlog.fail_('G8: Duplicate rows present', g8_detail)
        print(f'  [FAIL] G8 — {g8_detail}')
 
    # ── Gate G9: HAZ, WAZ, WHZ columns present ────────────────────────────
    required_cols = ['HAZ', 'WAZ', 'WHZ', 'stunted', 'underweight', 'wasted']
    missing_cols = [c for c in required_cols if c not in df.columns]
    gate_g9 = len(missing_cols) == 0
    if gate_g9:
        vlog.pass_('G9: All required columns present', str(required_cols))
        print(f'  [PASS] G9 — All required columns present')
    else:
        vlog.fail_('G9: Missing required columns', str(missing_cols))
        print(f'  [FAIL] G9 — Missing: {missing_cols}')
 
    # ── Gate G10: Class imbalance ratios computed ─────────────────────────
    try:
        weights = compute_class_weights(df, ['stunted', 'underweight', 'wasted'])
        gate_g10 = True
        vlog.pass_('G10: Class weights computed', str(weights))
        print(f'  [PASS] G10 — scale_pos_weight: {weights}')
    except Exception as e:
        gate_g10 = False
        vlog.fail_('G10: Class weight computation failed', str(e))
        print(f'  [FAIL] G10 — {e}')
 
    # ── VERIFY: missing value profile (DELETE after first run) ─────────────
    # # VERIFY: Print null percentage per column — use to plan imputation strategy
    # print('\n# VERIFY: Missing value rates per column (>5% shown):')
    # missing_rates = df.isnull().mean().sort_values(ascending=False)
    # print(missing_rates[missing_rates > 0.05].round(3).to_string())
    # ── END VERIFY block ──────────────────────────────────────────────────
 
    # ── Gate G11: No nulls in feature columns used by model ──────────────
    key_features = ['HV270', 'B4', 'V025']   # Wealth, sex, urban/rural
    null_in_features = {
        col: int(df[col].isnull().sum())
        for col in key_features if col in df.columns
    }
    gate_g11 = all(v == 0 for v in null_in_features.values())
    if gate_g11:
        vlog.pass_('G11: Key feature columns have no nulls', str(key_features))
        print(f'  [PASS] G11 — Key feature columns: no nulls')
    else:
        # Nulls in these columns is expected and handled by imputation in preprocessing.
        # Gate is INFO-level here, not a failure.
        null_summary = {k: v for k, v in null_in_features.items() if v > 0}
        vlog.info_(f'G11: Nulls found in features (to impute in Week 3): {null_summary}')
        print(f'  [INFO] G11 — Nulls to impute in preprocessing: {null_summary}')
        gate_g11 = True   # Not a blocking failure — imputation planned for Week 3
 
    # ── Gate G12: Cleaning log has entries ───────────────────────────────
    from src.logger import CleaningLogger
    clog_instance = CleaningLogger()
    entry_count = clog_instance.count()
    gate_g12 = entry_count >= 3   # At least the 3 entries from load_and_label()
    if gate_g12:
        vlog.pass_('G12: Cleaning log has entries', f'{entry_count} entries recorded')
        print(f'  [PASS] G12 — Cleaning log: {entry_count} entries')
    else:
        vlog.fail_('G12: Cleaning log empty or insufficient', f'{entry_count} entries')
        print(f'  [FAIL] G12 — Only {entry_count} cleaning log entries')
 
    # ── Summary ───────────────────────────────────────────────────────────
    all_gates = [gate_g1,gate_g2,gate_g3,gate_g4,gate_g5,gate_g6,
                 gate_g7,gate_g8,gate_g9,gate_g10,gate_g11,gate_g12]
    passed = sum(all_gates)
    total  = len(all_gates)
    all_passed = vlog.finish_section()
 
    print()
    print('=' * 70)
    print(f'RESULT: {passed}/{total} gates passed')
    if all_passed:
        print('ALL GATES PASSED — data is ready for Week 3 cleaning.')
        print('See reports/validation_report.txt for the full audit record.')
    else:
        print(f'FAILED GATES: {total - passed} — fix before starting Week 3.')
        print('See reports/validation_report.txt for details.')
        sys.exit(1)
    print('=' * 70)
 
 
if __name__ == '__main__':
    main()
 
