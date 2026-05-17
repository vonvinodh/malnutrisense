"""
src/logger.py — MalnutriSense reusable logging system.
 
Provides three logging channels:
  1. CleaningLogger  — writes structured rows to reports/cleaning_log.csv
                        One row per data transformation step.
  2. ValidationLogger — writes timestamped sections to reports/validation_report.txt
                        One section per validation run (pass or fail).
  3. get_console_logger() — returns a standard Python logger that writes to both
                             the terminal and reports/pipeline.log.
 
Usage in any module:
    from src.logger import CleaningLogger, ValidationLogger, get_console_logger
 
    log = get_console_logger(__name__)   # for console/file output
    cleaning = CleaningLogger()           # for cleaning_log.csv
    validation = ValidationLogger()       # for validation_report.txt
"""
 
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
 
# Import paths from config — do not hardcode paths here
from src.config import CLEANING_LOG_PATH, VALIDATION_REPORT_PATH, LOGS_DIR
 
 
# ── Column headers for cleaning_log.csv ──────────────────────────────────
# Every call to CleaningLogger.log() writes exactly these columns.
# Never add values without a matching header — the CSV will corrupt.
_CLEANING_LOG_HEADERS = [
    'timestamp',        # ISO-8601 UTC timestamp of when the step ran
    'dataset',          # Which dataset was modified (e.g. 'nfhs5_kr')
    'step',             # Short step name (e.g. 'replace_missing_codes')
    'column_affected',  # Column(s) modified, or 'all' / 'N/A'
    'issue_found',      # Description of the problem before the fix
    'action_taken',     # What was done to fix it
    'rows_affected',    # Integer count of rows changed, or -1 if N/A
    'validation_result',# 'PASS', 'FAIL', or 'INFO'
    'analyst_notes',    # Free-text notes for the paper Methods section
]
 
 
class CleaningLogger:
    """
    Writes one row per data cleaning operation to reports/cleaning_log.csv.
 
    The CSV is append-only. If the file does not exist, it is created with
    a header row on first use. All subsequent calls append without rewriting
    the header — safe to call across multiple scripts and Codespace sessions.
 
    Example:
        cleaning = CleaningLogger()
        cleaning.log(
            dataset='nfhs5_kr',
            step='replace_missing_codes',
            column_affected='HW70, HW71, HW72',
            issue_found='DHS codes 9996-9999 present in Z-score columns',
            action_taken='Replaced [9996,9997,9998,9999] with NaN',
            rows_affected=28450,
            validation_result='PASS',
            analyst_notes='Z-score max dropped from 9999 to 597 after replacement'
        )
    """
 
    def __init__(self, path: Path = CLEANING_LOG_PATH) -> None:
        self.path = path
        # Create the CSV with header if it does not yet exist
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=_CLEANING_LOG_HEADERS)
                writer.writeheader()
 
    def log(
        self,
        dataset: str,
        step: str,
        column_affected: str,
        issue_found: str,
        action_taken: str,
        rows_affected: int = -1,
        validation_result: str = 'INFO',
        analyst_notes: str = '',
    ) -> None:
        """
        Append one cleaning step record to the CSV.
 
        Args:
            dataset:           Short name identifying the dataset.
            step:              Snake_case name of the cleaning operation.
            column_affected:   Column(s) touched, comma-separated string.
            issue_found:       Plain-English description of the problem.
            action_taken:      What was done. Precise enough to reproduce.
            rows_affected:     Number of rows changed. Use -1 if not applicable.
            validation_result: 'PASS', 'FAIL', or 'INFO'.
            analyst_notes:     Extra notes for the paper Methods section.
        """
        # Validate result value before writing
        if validation_result not in ('PASS', 'FAIL', 'INFO'):
            raise ValueError(
                f"validation_result must be 'PASS', 'FAIL', or 'INFO'. "
                f"Got: '{validation_result}'"
            )
 
        row = {
            'timestamp':          datetime.now(timezone.utc).isoformat(),
            'dataset':            dataset,
            'step':               step,
            'column_affected':    column_affected,
            'issue_found':        issue_found,
            'action_taken':       action_taken,
            'rows_affected':      rows_affected,
            'validation_result':  validation_result,
            'analyst_notes':      analyst_notes,
        }
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_CLEANING_LOG_HEADERS)
            writer.writerow(row)
 
    def read_all(self) -> list[dict]:
        """Return all rows from the cleaning log as a list of dicts."""
        if not self.path.exists():
            return []
        with open(self.path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
 
    def count(self) -> int:
        """Return the number of log entries (excluding header row)."""
        return len(self.read_all())
 
 
class ValidationLogger:
    """
    Writes timestamped validation summaries to reports/validation_report.txt.
 
    Each call to start_section() begins a new dated block in the file.
    Calls to pass_() and fail_() add individual rule results within the block.
    finish_section() closes the block with a summary (X passed, Y failed).
 
    The file is append-only — multiple runs accumulate as dated sections,
    creating a full audit trail of data quality checks over time.
 
    Example:
        vlog = ValidationLogger()
        vlog.start_section('Week 2 — NFHS-5 cleaning validation')
        vlog.pass_('Row count within expected range', '232,920 rows')
        vlog.fail_('No missing codes remain', 'HW70 still has 43 values = 9999')
        vlog.finish_section()
    """
 
    def __init__(self, path: Path = VALIDATION_REPORT_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pass_count: int = 0
        self._fail_count: int = 0
        self._in_section: bool = False
 
    def start_section(self, title: str) -> None:
        """Open a new dated validation section in the report."""
        self._pass_count = 0
        self._fail_count = 0
        self._in_section = True
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        divider = '=' * 70
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'\n{divider}\n')
            f.write(f'VALIDATION RUN: {title}\n')
            f.write(f'Timestamp: {ts}\n')
            f.write(f'{divider}\n\n')
 
    def pass_(self, rule: str, detail: str = '') -> None:
        """Record a passing validation rule."""
        self._require_section()
        self._pass_count += 1
        detail_str = f'  — {detail}' if detail else ''
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'  [PASS] {rule}{detail_str}\n')
 
    def fail_(self, rule: str, detail: str = '') -> None:
        """Record a failing validation rule."""
        self._require_section()
        self._fail_count += 1
        detail_str = f'  — {detail}' if detail else ''
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'  [FAIL] {rule}{detail_str}\n')
 
    def info_(self, message: str) -> None:
        """Record an informational note (not pass/fail)."""
        self._require_section()
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'  [INFO] {message}\n')
 
    def finish_section(self) -> bool:
        """
        Close the current validation section and write the summary.
 
        Returns:
            True if all rules passed, False if any failed.
        """
        self._require_section()
        self._in_section = False
        all_passed = self._fail_count == 0
        status = 'ALL PASSED' if all_passed else f'FAILED ({self._fail_count} failures)'
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(f'\nSummary: {self._pass_count} passed, '
                    f'{self._fail_count} failed — {status}\n')
        return all_passed
 
    def _require_section(self) -> None:
        if not self._in_section:
            raise RuntimeError(
                'Call start_section() before pass_(), fail_(), or info_().'
            )
 
 
# ── Console + file logger ──────────────────────────────────────────────────
def get_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to both the terminal and reports/pipeline.log.
 
    Uses the standard library logging module. The logger is idempotent —
    calling get_console_logger() with the same name twice returns the same
    logger with handlers attached only once.
 
    Args:
        name:  Logger name. Convention: pass __name__ from the calling module.
               This makes log messages show the source module automatically.
        level: Minimum severity to log. Default: logging.INFO.
               Use logging.DEBUG during active development.
 
    Returns:
        Configured logging.Logger instance.
 
    Example:
        log = get_console_logger(__name__)
        log.info('Loading NFHS-5 file...')
        log.warning('Missing values found in HW70: 28,450 rows')
        log.error('Z-score column still contains value 9999 after cleaning')
    """
    logger = logging.getLogger(name)
 
    # Guard: if handlers already attached, return the existing logger
    if logger.handlers:
        return logger
 
    logger.setLevel(level)
 
    # Format: [timestamp]  [LEVEL]  module_name: message
    fmt = logging.Formatter(
        fmt='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
 
    # Handler 1: stdout (visible in Codespace terminal)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
 
    # Handler 2: file (persistent in reports/pipeline.log)
    log_file = LOGS_DIR / 'pipeline.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
 
    return logger
 
