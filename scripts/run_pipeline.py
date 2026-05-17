"""
scripts/run_pipeline.py — End-to-end data pipeline runner.
 
Runs the complete pipeline from raw NFHS .DTA file to model-ready CSVs.
All outputs are saved to data/processed/ and data/processed/train_test_splits/.
 
Usage:
    python3 scripts/run_pipeline.py
 
Outputs (all in data/processed/):
    nfhs5_cleaned.csv       — full cleaned feature + label matrix
    y_labels.csv            — target columns only (stunted, underweight, wasted)
    train_test_splits/
        X_train.csv, X_test.csv, y_train.csv, y_test.csv
 
Reports updated:
    reports/cleaning_log.csv         — one row per transformation
    reports/validation_report.txt    — label integrity section appended
    reports/tables/pipeline_summary.csv  — per-split row/col counts
"""
 
import sys
from pathlib import Path
 
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import (
    NFHS5_PATH, NFHS_COLS,
    NFHS5_CLEANED_PATH, PROCESSED_DIR, TRAIN_TEST_DIR,
    TABLES_DIR,
    validate_environment,
)
from src.data_loader import load_nfhs5_kr
from src.labels import apply_who_thresholds, audit_labels, validate_label_integrity
from src.preprocessing import run_full_pipeline, make_train_test_split
from src.logger import get_console_logger
from src.utils import save_dataframe, timer, format_number
 
log = get_console_logger(__name__)
 
 
def main() -> None:
    print('=' * 70)
    print('MalnutriSense — Data Pipeline')
    print('=' * 70)
 
    # ── Step 0: Environment validation ───────────────────────────────────
    try:
        validate_environment(require_data=True)
        log.info('Environment validated')
    except (RuntimeError, FileNotFoundError) as e:
        print(f'[FAIL] Environment check failed: {e}')
        sys.exit(1)
 
    # ── Step 1: Load raw NFHS-5 file ─────────────────────────────────────
    print(f'\n[1/6] Loading {NFHS5_PATH.name}...')
    with timer('Load NFHS-5 .DTA'):
        df_raw = load_nfhs5_kr(path=NFHS5_PATH, columns=NFHS_COLS)
    print(f'      {format_number(len(df_raw))} rows loaded')
 
    # ── Step 2: Apply WHO malnutrition thresholds ─────────────────────────
    print('\n[2/6] Applying WHO Z < -2.0 thresholds...')
    with timer('Apply WHO thresholds'):
        df_labelled = apply_who_thresholds(df_raw)
    print(f'      Labels created: stunted, underweight, wasted')
 
    # ── Step 3: Run preprocessing pipeline ────────────────────────────────
    print('\n[3/6] Running preprocessing pipeline...')
    with timer('Preprocessing pipeline'):
        df_clean = run_full_pipeline(df_labelled)
    print(f'      {format_number(len(df_clean))} rows, {df_clean.shape[1]} columns')
 
    # ── Step 4: Validate labels ───────────────────────────────────────────
    print('\n[4/6] Validating label integrity...')
    label_ok = validate_label_integrity(df_clean)
    if not label_ok:
        print('[FAIL] Label validation failed — check reports/validation_report.txt')
        sys.exit(1)
    print('      All label integrity checks PASSED')
 
    # ── Step 5: Audit labels ─────────────────────────────────────────────
    print('\n[5/6] Auditing labels and class balance...')
    audit = audit_labels(df_clean)
    for col, prev in audit['prevalence'].items():
        print(f'      {col}: {prev:.1%}')
    for col, w in audit.get('class_weights', {}).items():
        print(f'      scale_pos_weight[{col}] = {w}')
 
    # ── Step 6: Save all outputs ──────────────────────────────────────────
    print('\n[6/6] Saving outputs...')
 
    # Full cleaned matrix
    save_dataframe(df_clean, NFHS5_CLEANED_PATH, 'nfhs5_cleaned')
    print(f'      Saved: {NFHS5_CLEANED_PATH}')
 
    # Label columns only
    y_path = PROCESSED_DIR / 'y_labels.csv'
    from src.config import TARGET_COLS
    y_cols_present = [c for c in TARGET_COLS if c in df_clean.columns]
    y_df = df_clean[y_cols_present].dropna(subset=y_cols_present)
    save_dataframe(y_df, y_path, 'y_labels')
    print(f'      Saved: {y_path}')
    X_train, X_test, y_train, y_test = make_train_test_split(df_clean)
 
    for name, split_df in [
        ('X_train', X_train), ('X_test', X_test),
        ('y_train', y_train), ('y_test', y_test),
    ]:
        save_dataframe(split_df, TRAIN_TEST_DIR / f'{name}.csv', name)
        print(f'      Saved: {TRAIN_TEST_DIR / name}.csv  ({format_number(len(split_df))} rows)')
 
    # Pipeline summary report
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {'split': 'full_cleaned',  'rows': len(df_clean),  'cols': df_clean.shape[1]},
        {'split': 'X_train',       'rows': len(X_train),   'cols': X_train.shape[1]},
        {'split': 'X_test',        'rows': len(X_test),    'cols': X_test.shape[1]},
        {'split': 'y_train',       'rows': len(y_train),   'cols': y_train.shape[1]},
        {'split': 'y_test',        'rows': len(y_test),    'cols': y_test.shape[1]},
    ])
    summary_path = TABLES_DIR / 'pipeline_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'      Saved: {summary_path}')
 
    # ── Done ─────────────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print('PIPELINE COMPLETE')
    print(f'  Cleaned dataset: {format_number(len(df_clean))} rows x {df_clean.shape[1]} cols')
    print(f'  Train split:     {format_number(len(X_train))} rows')
    print(f'  Test split:      {format_number(len(X_test))} rows')
    print(f'  All outputs in:  {PROCESSED_DIR}')
    print('=' * 70)
 
 
if __name__ == '__main__':
    main()
 
