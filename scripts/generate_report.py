"""
scripts/generate_report.py — Validation report generator.
 
Reads the processed dataset, runs all validation checks in soft mode,
and writes a complete human-readable report to reports/validation_report.txt.
 
Usage:  python3 scripts/generate_report.py
 
Outputs:
    reports/validation_report.txt      — full human-readable report (appended)
    reports/tables/validation_summary.csv  — structured summary for paper
"""
 
import sys, platform
from datetime import datetime, timezone
from pathlib import Path
 
import pandas as pd
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import (
    NFHS5_CLEANED_PATH, TRAIN_TEST_DIR, TABLES_DIR, VALIDATION_REPORT_PATH,
    PROCESSED_DIR, TARGET_COLS, validate_environment,
)
from src.validation import (
    validate_all_soft, PREVALENCE_THRESHOLDS, ValidationError,
)
from src.utils import compute_class_weights, format_number
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
 
def _write(path: Path, text: str) -> None:
    """Append text to the report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
 
 
def _divider(path: Path, char: str = '─', width: int = 70) -> None:
    _write(path, char * width + '\n')
 
 
def generate_report() -> bool:
    """Generate the validation report. Returns True if all checks pass."""
    rpt = VALIDATION_REPORT_PATH
    ts  = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
 
    # Header
    _write(rpt, '\n')
    _divider(rpt, '=')
    _write(rpt, 'MALNUTRISENSE — DATA VALIDATION REPORT\n')
    _write(rpt, f'Generated: {ts}\n')
    _write(rpt, 'Script:    scripts/generate_report.py\n')
    _divider(rpt, '=')
    _write(rpt, '\n')
 
    # Section 1: Environment
    _write(rpt, 'SECTION 1: ENVIRONMENT\n')
    _divider(rpt)
    try:
        validate_environment(require_data=True)
        env_status = 'PASS'
    except (RuntimeError, FileNotFoundError) as e:
        env_status = f'FAIL — {e}'
    _write(rpt, f'  Python version:  {sys.version.split()[0]}\n')
    _write(rpt, f'  Platform:        {platform.system()} {platform.release()}\n')
    _write(rpt, f'  Environment:     {env_status}\n')
    _write(rpt, f'  Cleaned data:    {NFHS5_CLEANED_PATH}\n\n')
 
    # Section 2: Dataset statistics
    _write(rpt, 'SECTION 2: DATASET STATISTICS\n')
    _divider(rpt)
    if not NFHS5_CLEANED_PATH.exists():
        _write(rpt, '  ERROR: nfhs5_cleaned.csv not found.\n')
        _write(rpt, '  Run: python3 scripts/run_pipeline.py\n')
        _write(rpt, '\nOVERALL RESULT: FAIL — cleaned dataset missing\n')
        return False
 
    log.info(f'Loading {NFHS5_CLEANED_PATH.name}...')
    df = pd.read_csv(NFHS5_CLEANED_PATH)
    _write(rpt, f'  Rows:     {format_number(len(df))}\n')
    _write(rpt, f'  Columns:  {df.shape[1]}\n')
    _write(rpt, f'  Cols:     {list(df.columns)}\n\n')
 
    _write(rpt, '  Missing values per column (non-zero only):\n')
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        _write(rpt, '    None — all feature columns complete\n')
    else:
        for col, cnt in missing.items():
            _write(rpt, f'    {col:<25} {format_number(cnt):>10} ({cnt/len(df):.1%})\n')
    _write(rpt, '\n')
 
    # Section 3: Validation checks
    _write(rpt, 'SECTION 3: VALIDATION CHECKS\n')
    _divider(rpt)
    _write(rpt, '  Running validate_all_soft() — collects all failures\n\n')
    all_passed, failures = validate_all_soft(
        df, section_title=f'generate_report — {ts}'
    )
    if all_passed:
        _write(rpt, '  RESULT: ALL 6 CHECKS PASSED\n\n')
    else:
        _write(rpt, f'  RESULT: {len(failures)} CHECK(S) FAILED\n')
        for i, msg in enumerate(failures, 1):
            _write(rpt, f'\n  Failure {i}:\n')
            for line in str(msg).strip().split('\n'):
                _write(rpt, f'    {line}\n')
        _write(rpt, '\n')
 
    # Section 4: Prevalence table
    _write(rpt, 'SECTION 4: PREVALENCE AND CLASS BALANCE\n')
    _divider(rpt)
    _write(rpt, f'  {"Phenotype":<20} {"Measured":>10}  {"Threshold":>18}  {"Status":>8}\n')
    _write(rpt, f'  {"-"*20} {"-"*10}  {"-"*18}  {"-"*8}\n')
    summary_rows = []
    for phenotype, (lo, hi) in PREVALENCE_THRESHOLDS.items():
        if phenotype not in df.columns:
            _write(rpt, f'  {phenotype:<20} {"MISSING":>10}  {"N/A":>18}  {"FAIL":>8}\n')
            summary_rows.append({'phenotype':phenotype,'measured':None,'lo':lo,'hi':hi,'status':'FAIL'})
            continue
        valid = df[phenotype].dropna()
        prev  = float(valid.mean())
        ok    = lo <= prev <= hi
        status = 'PASS' if ok else 'FAIL'
        _write(rpt, f'  {phenotype:<20} {prev:>9.1%}  {lo:.0%} – {hi:.0%}          {status:>8}\n')
        summary_rows.append({'phenotype':phenotype,'measured':round(prev,4),'lo':lo,'hi':hi,'status':status})
    _write(rpt, '\n')
    _write(rpt, '  XGBoost scale_pos_weight:\n')
    try:
        weights = compute_class_weights(df[TARGET_COLS].dropna().astype(int), TARGET_COLS)
        for col, w in weights.items():
            _write(rpt, f'    {col:<20} {w}\n')
    except Exception as e:
        _write(rpt, f'    Could not compute: {e}\n')
    _write(rpt, '\n')
 
    # Section 5: Train/test split
    _write(rpt, 'SECTION 5: TRAIN / TEST SPLIT\n')
    _divider(rpt)
    for name, path in [('X_train',TRAIN_TEST_DIR/'X_train.csv'),('X_test',TRAIN_TEST_DIR/'X_test.csv'),
                        ('y_train',TRAIN_TEST_DIR/'y_train.csv'),('y_test',TRAIN_TEST_DIR/'y_test.csv')]:
        if path.exists():
            s = pd.read_csv(path)
            _write(rpt, f'  {name:<10}  {format_number(len(s)):>10} rows  x  {s.shape[1]} cols\n')
        else:
            _write(rpt, f'  {name:<10}  NOT FOUND — run scripts/run_pipeline.py\n')
    _write(rpt, '\n')
 
    # Section 6: Overall verdict
    _write(rpt, 'SECTION 6: OVERALL VERDICT\n')
    _divider(rpt, '=')
    if all_passed:
        _write(rpt, '  VERDICT: PASS\n')
        _write(rpt, '  All validation checks passed. Dataset ready for model training.\n')
    else:
        _write(rpt, f'  VERDICT: FAIL ({len(failures)} failure(s))\n')
        _write(rpt, '  Fix all failures above before starting model training.\n')
        _write(rpt, '  Re-run: python3 scripts/run_pipeline.py\n')
        _write(rpt, '          python3 scripts/generate_report.py\n')
    _divider(rpt, '=')
    _write(rpt, '\n')
 
    # Machine-readable CSV
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / 'validation_summary.csv', index=False)
    log.info('Saved: reports/tables/validation_summary.csv')
    return all_passed
 
 
def main() -> None:
    print('=' * 70)
    print('MalnutriSense — Validation Report Generator')
    print('=' * 70)
    all_passed = generate_report()
    print(f'\nReport: {VALIDATION_REPORT_PATH}')
    print(f'CSV:    {TABLES_DIR / "validation_summary.csv"}')
    print()
    if all_passed:
        print('VERDICT: PASS — dataset ready for model training.')
    else:
        print('VERDICT: FAIL — see report for details')
        sys.exit(1)
    print('=' * 70)
 
if __name__ == '__main__':
    main()
 
