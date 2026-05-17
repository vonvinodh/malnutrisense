"""
src/validation.py — Data quality validation with hard-fail thresholds.
 
Enforces data quality as a code contract:
    - ValidationLogger (src/logger.py) RECORDS pass/fail results
    - This module RAISES exceptions when thresholds are violated
 
Prevalence thresholds (project-specific, tighter than config.py defaults):
    Stunting:     30% – 42%   (NFHS-5 national: 35.5%)
    Underweight:  30% – 35%   (NFHS-5 national: 32.1%)
    Wasting:      17% – 22%   (NFHS-5 national: 19.3%)
 
Public API:
    check_prevalence(df)         -> dict   (raises ValidationError on failure)
    check_row_count(df)          -> int    (raises ValidationError on failure)
    check_no_missing_codes(df)   -> bool   (raises ValidationError on failure)
    check_z_score_scale(df)      -> bool   (raises ValidationError on failure)
    check_label_binary(df)       -> bool   (raises ValidationError on failure)
    check_no_duplicates(df)      -> int    (raises ValidationError on failure)
    validate_all(df)             -> dict   (raises on first failure)
    validate_all_soft(df)        -> tuple  (collects all failures)
"""
 
import pandas as pd
import numpy as np
from typing import Optional
 
from src.config import (
    MISSING_CODES, Z_SCORE_MIN, Z_SCORE_MAX,
    MIN_VALID_ROWS, TARGET_COLS,
)
from src.logger import CleaningLogger, ValidationLogger, get_console_logger
from src.utils import format_number
 
log          = get_console_logger(__name__)
cleaning_log = CleaningLogger()
vlog         = ValidationLogger()
 
 
# ── Project-specific prevalence thresholds ────────────────────────────────
# Tighter than config.py broad guards — reflects fully preprocessed dataset.
PREVALENCE_THRESHOLDS: dict[str, tuple[float, float]] = {
    'stunted':     (0.30, 0.42),  # NFHS-5 national: 35.5%
    'underweight': (0.30, 0.35),  # NFHS-5 national: 32.1%
    'wasted':      (0.17, 0.22),  # NFHS-5 national: 19.3%
}
THRESHOLD_SOURCE = 'NFHS-5 National Fact Sheet (2019-21), India'
 
 
class ValidationError(Exception):
    """Raised when a data quality check fails."""
    def __init__(self, check_name: str, actual: str, expected: str,
                 action: str = 'Fix the preprocessing pipeline before training.') -> None:
        self.check_name = check_name
        self.actual     = actual
        self.expected   = expected
        message = (
            f'\n[VALIDATION FAILED] {check_name}'
            f'\n  Found:    {actual}'
            f'\n  Expected: {expected}'
            f'\n  Action:   {action}'
        )
        super().__init__(message)
 
 
def check_prevalence(df: pd.DataFrame, raise_on_fail: bool = True) -> dict[str, float] | tuple[dict[str, float], list[ValidationError]]:
    """
    Validate stunting, underweight, and wasting prevalence against thresholds.
    If raise_on_fail=True, raises ValidationError with direction ('too low' / 'too high') and action on first failure.
    If raise_on_fail=False, returns (results, errors) where errors is list of ValidationError for each failure.
    """
    results: dict[str, float] = {}
    errors: list[ValidationError] = []
    for phenotype, (lo, hi) in PREVALENCE_THRESHOLDS.items():
        if phenotype not in df.columns:
            err = ValidationError(
                check_name=f'{phenotype}_column_present',
                actual='column missing',
                expected=f'Column "{phenotype}" in DataFrame',
                action='Run labels.apply_who_thresholds() before validation.',
            )
            errors.append(err)
            if raise_on_fail:
                raise err
            continue
        valid = df[phenotype].dropna()
        prev  = float(valid.mean())
        results[phenotype] = round(prev, 4)
        log.info(f'{phenotype}: {prev:.1%}  (threshold {lo:.0%}–{hi:.0%})')
        if not (lo <= prev <= hi):
            direction = 'low' if prev < lo else 'high'
            err = ValidationError(
                check_name=f'{phenotype}_prevalence',
                actual=f'{prev:.2%}',
                expected=f'{lo:.0%} – {hi:.0%}  (source: {THRESHOLD_SOURCE})',
                action=(
                    f'Prevalence is too {direction}. '
                    'Check missing code replacement, Z-score scaling (÷100), '
                    'and the WHO threshold in labels.py.'
                ),
            )
            errors.append(err)
            if raise_on_fail:
                raise err
    if errors and not raise_on_fail:
        log.warning(f'Prevalence check FAILED for {len(errors)} phenotypes')
    elif not errors:
        log.info('Prevalence check PASSED for all three phenotypes')
        cleaning_log.log(
            dataset='nfhs5_kr', step='check_prevalence',
            column_affected='stunted, underweight, wasted',
            issue_found='N/A — validation step',
            action_taken=f'Prevalence verified: {results}',
            rows_affected=len(df), validation_result='PASS',
            analyst_notes=str(PREVALENCE_THRESHOLDS),
        )
    if raise_on_fail:
        return results
    else:
        return results, errors
 
 
def check_row_count(df: pd.DataFrame) -> int:
    """Raise ValidationError if DataFrame has fewer than MIN_VALID_ROWS rows."""
    count = len(df)
    if count < MIN_VALID_ROWS:
        raise ValidationError(
            check_name='row_count',
            actual=format_number(count),
            expected=f'>= {format_number(MIN_VALID_ROWS)} rows',
            action='Too many rows removed during cleaning. Check imputation and dedup.',
        )
    log.info(f'Row count check PASSED: {format_number(count)}')
    return count
 
 
def check_no_missing_codes(df: pd.DataFrame) -> bool:
    """Raise ValidationError if any DHS missing codes (9996-9999) remain."""
    offending: dict[str, int] = {}
    for col in df.select_dtypes(include='number').columns:
        count = int(df[col].isin(MISSING_CODES).sum())
        if count > 0:
            offending[col] = count
    if offending:
        raise ValidationError(
            check_name='no_dhs_missing_codes',
            actual=f'{offending}',
            expected='0 DHS missing codes in all numeric columns',
            action='Check data_loader.load_nfhs5_kr() step 2 (missing code replacement).',
        )
    log.info('Missing code check PASSED')
    return True
 
 
def check_z_score_scale(df: pd.DataFrame) -> bool:
    """Raise ValidationError if HAZ/WAZ/WHZ values are outside [-6, +6]."""
    violations: dict[str, int] = {}
    for z_col in ['HAZ', 'WAZ', 'WHZ']:
        if z_col not in df.columns:
            continue
        valid = df[z_col].dropna()
        out   = int((~valid.between(Z_SCORE_MIN, Z_SCORE_MAX)).sum())
        if out > 0:
            violations[z_col] = out
    if violations:
        raise ValidationError(
            check_name='z_score_scale',
            actual=f'Out-of-range: {violations}',
            expected=f'All values in [{Z_SCORE_MIN}, {Z_SCORE_MAX}]',
            action='Z-scores not divided by 100. Recheck data_loader.load_nfhs5_kr().',
        )
    log.info(f'Z-score scale check PASSED')
    return True
 
 
def check_label_binary(df: pd.DataFrame) -> bool:
    """Raise ValidationError if label columns contain values other than 0, 1, NaN."""
    problems: dict[str, set] = {}
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            problems[col] = unique_vals
    if problems:
        raise ValidationError(
            check_name='label_binary',
            actual=f'Non-binary values: {problems}',
            expected='All label columns contain only 0, 1, or NaN',
            action='Recheck labels.apply_who_thresholds().',
        )
    log.info('Label binary check PASSED')
    return True
 
 
def check_no_duplicates(df: pd.DataFrame) -> int:
    """Raise ValidationError if more than 100 duplicate rows found."""
    dup_count = int(df.duplicated().sum())
    if dup_count > 100:
        raise ValidationError(
            check_name='no_duplicate_rows',
            actual=f'{format_number(dup_count)} duplicate rows',
            expected='<= 100 duplicate rows (NFHS tolerance)',
            action='Check preprocessing.remove_duplicates() was called.',
        )
    if dup_count > 0:
        log.warning(f'{dup_count} duplicate rows (within tolerance of 100)')
    log.info(f'Duplicate check PASSED: {dup_count} duplicates')
    return dup_count
 
 
def validate_all(df: pd.DataFrame,
                 section_title: str = 'Full dataset validation') -> dict:
    """Run all six checks. Raises ValidationError on the first failure."""
    log.info(f'Running validate_all: {section_title}')
    vlog.start_section(section_title)
    results: dict = {}
    checks = [
        ('row_count',        lambda: check_row_count(df)),
        ('no_missing_codes', lambda: check_no_missing_codes(df)),
        ('z_score_scale',    lambda: check_z_score_scale(df)),
        ('label_binary',     lambda: check_label_binary(df)),
        ('no_duplicates',    lambda: check_no_duplicates(df)),
        ('prevalence',       lambda: check_prevalence(df)),
    ]
    for check_name, check_fn in checks:
        try:
            result = check_fn()
            results[check_name] = result
            vlog.pass_(check_name, str(result)[:120])
        except ValidationError as e:
            vlog.fail_(check_name, str(e)[:200])
            vlog.finish_section()
            raise
    results['all_passed'] = True
    vlog.finish_section()
    log.info('validate_all PASSED')
    return results
 
 
def validate_all_soft(df: pd.DataFrame,
                      section_title: str = 'Soft validation run') -> tuple[bool, list[str]]:
    """Run all checks and collect all failures without stopping on first."""
    vlog.start_section(section_title)
    failures: list[str] = []
    checks = [
        ('row_count',        lambda: check_row_count(df)),
        ('no_missing_codes', lambda: check_no_missing_codes(df)),
        ('z_score_scale',    lambda: check_z_score_scale(df)),
        ('label_binary',     lambda: check_label_binary(df)),
        ('no_duplicates',    lambda: check_no_duplicates(df)),
        ('prevalence',       lambda: check_prevalence(df, raise_on_fail=False)),
    ]
    for check_name, check_fn in checks:
        try:
            if check_name == 'prevalence':
                results, errors = check_fn()
                for err in errors:
                    failures.append(str(err))
                    vlog.fail_(check_name, str(err)[:200])
            else:
                result = check_fn()
                vlog.pass_(check_name, str(result)[:120])
        except ValidationError as e:
            failures.append(str(e))
            vlog.fail_(check_name, str(e)[:200])
    all_passed = vlog.finish_section()
    if all_passed:
        log.info('validate_all_soft PASSED')
    else:
        log.warning(f'validate_all_soft: {len(failures)} failures')
    return all_passed, failures
 
