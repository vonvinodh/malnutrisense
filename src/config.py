"""
src/config.py — MalnutriSense project configuration.
 
Single source of truth for:
  - All file and directory paths
  - Environment variable loading with type-safety
  - NFHS column selection and DHS missing-value codes
  - WHO malnutrition thresholds and model constants
  - Environment validation (call validate_environment() at startup)
 
Import in any module:
    from src.config import NFHS5_PATH, NFHS_COLS, MISSING_CODES
    from src.config import validate_environment, get_env
"""
 
import os
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
 
# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
# load_dotenv() reads .env in the project root and injects variables into
# os.environ so os.getenv() picks them up. Safe to call multiple times —
# subsequent calls are no-ops if variables are already set.
load_dotenv()
 
 
# ---------------------------------------------------------------------------
# Helper: typed environment variable access
# ---------------------------------------------------------------------------
def get_env(key: str, default: Any = None, cast: type = str) -> Any:
    """
    Fetch an environment variable with optional type casting.
 
    Args:
        key:     The environment variable name (e.g. 'RANDOM_STATE')
        default: Value to return when the variable is not set.
        cast:    Python type to cast the string value to (int, float, bool, str).
 
    Returns:
        The value cast to the requested type, or default if not set.
 
    Example:
        RANDOM_STATE = get_env('RANDOM_STATE', default=42, cast=int)
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    if cast is bool:
        # Accept 'true'/'1'/'yes' as True, anything else as False
        return raw.strip().lower() in ('true', '1', 'yes')
    try:
        return cast(raw.strip())
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Environment variable '{key}' has value '{raw}' which cannot "
            f"be cast to {cast.__name__}. Check your .env file."
        ) from e
 
 
# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# Path(__file__) is src/config.py
# .parent      is src/
# .parent      is the project root /workspaces/malnutrisense
# This resolves correctly regardless of the current working directory.
ROOT: Path = Path(__file__).parent.parent.resolve()
 
 
# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR:       Path = ROOT / 'data'
RAW_DIR:        Path = DATA_DIR / 'raw'
PROCESSED_DIR:  Path = DATA_DIR / 'processed'
INTERIM_DIR:    Path = DATA_DIR / 'interim'
TRAIN_TEST_DIR: Path = PROCESSED_DIR / 'train_test_splits'
 
MODELS_DIR:  Path = ROOT / 'models'
REPORTS_DIR: Path = ROOT / 'reports'
FIGURES_DIR: Path = REPORTS_DIR / 'figures'
TABLES_DIR:  Path = REPORTS_DIR / 'tables'
LOGS_DIR:    Path = ROOT / 'reports'   # logs live in reports/ so they are committed
 
 
# ---------------------------------------------------------------------------
# Raw data file paths
# ---------------------------------------------------------------------------
# These files are in .gitignore (too large for git). They must be downloaded
# separately (Steps 8–10). Use paths_exist() to check before loading.
NFHS5_PATH:        Path = RAW_DIR / 'nfhs5' / 'IAKR7EFL.DTA'
NFHS4_PATH:        Path = RAW_DIR / 'nfhs4' / 'IAKR74FL.DTA'
SHAPEFILE_PATH:    Path = RAW_DIR / 'shapefiles' / 'gadm41_IND_2.shp'
ASPIRATIONAL_PATH: Path = RAW_DIR / 'external' / 'aspirational_districts.csv'
 
 
# ---------------------------------------------------------------------------
# Processed data paths (created by cleaning scripts in Weeks 3–4)
# ---------------------------------------------------------------------------
NFHS5_CLEANED_PATH:  Path = PROCESSED_DIR / 'nfhs5_cleaned.csv'
NFHS5_FEATURES_PATH: Path = PROCESSED_DIR / 'nfhs5_features.csv'
DISTRICTS_GEOJSON:   Path = PROCESSED_DIR / 'india_districts.geojson'
ASPIRATIONAL_CLEAN:  Path = PROCESSED_DIR / 'aspirational_districts_clean.csv'
 
# Train/test split paths (created at end of Week 3)
X_TRAIN_PATH: Path = TRAIN_TEST_DIR / 'X_train.csv'
X_TEST_PATH:  Path = TRAIN_TEST_DIR / 'X_test.csv'
Y_TRAIN_PATH: Path = TRAIN_TEST_DIR / 'y_train.csv'
Y_TEST_PATH:  Path = TRAIN_TEST_DIR / 'y_test.csv'
 
 
# ---------------------------------------------------------------------------
# Log file paths
# ---------------------------------------------------------------------------
CLEANING_LOG_PATH:    Path = LOGS_DIR / 'cleaning_log.csv'
VALIDATION_REPORT_PATH: Path = LOGS_DIR / 'validation_report.txt'
 
 
# ---------------------------------------------------------------------------
# NFHS column selection
# ---------------------------------------------------------------------------
# Load ONLY these columns from the 1,300+ column DTA file.
# Using usecols= saves ~70% memory and ~80% load time compared to loading all.
# Each column is documented with its DHS code, description, and usage.
NFHS_COLS: list[str] = [
    # Anthropometric measurements
    'HW1',   # Child age in months (0–59)
    'HW2',   # Child weight in kg × 10  (divide by 10 for kg)
    'HW3',   # Child height in cm × 10  (divide by 10 for cm)
    'HW70',  # Height-for-age Z-score × 100  → HAZ after /100  (stunting target)
    'HW71',  # Weight-for-age Z-score × 100  → WAZ after /100  (underweight target)
    'HW72',  # Weight-for-height Z-score × 100 → WHZ after /100 (wasting target)
 
    # Socioeconomic predictors
    'V106',  # Mother's highest education: 0=None 1=Primary 2=Secondary 3=Higher
    'HV270', # Wealth quintile: 1=Poorest 2=Poorer 3=Middle 4=Richer 5=Richest
    'V130',  # Religion (coded by DHS country file)
 
    # Geographic identifiers
    'V024',  # State code (numeric, matches NFHS state list)
    'V025',  # Type of residence: 1=Urban 2=Rural
    'HV001', # Cluster number (used for district mapping via GPS file)
 
    # Health and nutrition practices
    'H11',   # Had diarrhoea in last 2 weeks: 0=No 1=Yes
    'M4',    # Duration of breastfeeding in months
    'M19',   # Birth weight in grams  (9996=missing)
 
    # WASH (Water, Sanitation, Hygiene)
    'HV201', # Source of drinking water (coded: 11=Piped 21=Tube well 32=Unprotected)
    'HV205', # Type of toilet facility (coded: 11=Flush 21=Pit 31=None)
 
    # Fairness audit variables
    'B4',    # Sex of child: 1=Male 2=Female
]
 
 
# ---------------------------------------------------------------------------
# DHS missing value codes
# ---------------------------------------------------------------------------
# DHS encodes 'not applicable', 'missing', and 'don't know' as these integers.
# Must be replaced with float('nan') before any analysis — see data_loader.py.
# Includes both 4-digit and 5-digit variants used across different DHS versions.
MISSING_CODES: list[int] = [9996, 9997, 9998, 9999, 99996, 99997, 99998, 99999]
 
 
# ---------------------------------------------------------------------------
# WHO malnutrition thresholds
# ---------------------------------------------------------------------------
# Z-score strictly below this threshold = malnourished for that phenotype.
# Source: WHO Child Growth Standards (2006, reaffirmed 2022).
STUNTING_THRESHOLD:    float = -2.0   # HAZ < -2.0 → stunted
UNDERWEIGHT_THRESHOLD: float = -2.0   # WAZ < -2.0 → underweight
WASTING_THRESHOLD:     float = -2.0   # WHZ < -2.0 → wasted
 
# Physiological Z-score bounds (values outside these are measurement errors)
Z_SCORE_MIN: float = -6.0
Z_SCORE_MAX: float = +6.0
 
# Physiological bounds for birth weight (grams)
BIRTH_WEIGHT_MIN: int = 500
BIRTH_WEIGHT_MAX: int = 5000
 
 
# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int   = get_env('RANDOM_STATE', default=42,   cast=int)
TEST_SIZE:    float = get_env('TEST_SIZE',    default=0.20,  cast=float)
CV_FOLDS:     int   = get_env('CV_FOLDS',     default=5,     cast=int)
DEBUG:        bool  = get_env('DEBUG',         default=False, cast=bool)
 
# Target column names (output of create_labels() in data_loader.py)
TARGET_COLS:  list[str] = ['stunted', 'underweight', 'wasted']
 
# Expected prevalence ranges from NFHS-5 report (used in validation)
# Source: NFHS-5 National Fact Sheet (2019-21)
EXPECTED_STUNTING_RANGE:    tuple[float,float] = (0.30, 0.42)  # 35.5% nationally
EXPECTED_UNDERWEIGHT_RANGE: tuple[float,float] = (0.27, 0.38)  # 32.1% nationally
EXPECTED_WASTING_RANGE:     tuple[float,float] = (0.15, 0.25)  # 19.3% nationally
MIN_VALID_ROWS: int = 190_000  # Minimum acceptable rows after cleaning
 
 
# ---------------------------------------------------------------------------
# Helper: check if a set of paths exist
# ---------------------------------------------------------------------------
def paths_exist(*paths: Path) -> tuple[bool, list[str]]:
    """
    Check whether all supplied paths exist on disk.
 
    Args:
        *paths: One or more pathlib.Path objects to check.
 
    Returns:
        (all_exist: bool, missing: list of str paths that do not exist)
 
    Example:
        ok, missing = paths_exist(NFHS5_PATH, SHAPEFILE_PATH)
        if not ok:
            raise FileNotFoundError(f'Missing: {missing}')
    """
    missing = [str(p) for p in paths if not p.exists()]
    return len(missing) == 0, missing
 
 
# ---------------------------------------------------------------------------
# Environment validation (call once at the top of any main script)
# ---------------------------------------------------------------------------
def validate_environment(require_data: bool = False) -> None:
    """
    Validate that the project environment is correctly configured.
 
    Checks performed:
      1. Python version is 3.11 or higher
      2. All output directories exist (creates them if missing)
      3. If require_data=True, raw data files are present on disk
 
    Args:
        require_data: Set True in data-loading scripts to enforce that
                      NFHS5_PATH, NFHS4_PATH, SHAPEFILE_PATH, and
                      ASPIRATIONAL_PATH exist before execution begins.
 
    Raises:
        RuntimeError:      Python version is below 3.11.
        FileNotFoundError: require_data=True and one or more raw data
                           files are absent from data/raw/.
 
    Example (at the top of any script that processes data):
        from src.config import validate_environment
        validate_environment(require_data=True)
    """
    # 1. Python version check
    if sys.version_info < (3, 11):
        raise RuntimeError(
            f'Python 3.11+ required. Current version: {sys.version}. '
            'In Codespaces: verify devcontainer.json uses python:3.11 image.'
        )
 
    # 2. Ensure all output directories exist
    # These are safe to create if absent — they only contain generated files.
    for directory in [
        PROCESSED_DIR, INTERIM_DIR, TRAIN_TEST_DIR,
        MODELS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
 
    # 3. Raw data file check (optional — only for data-processing scripts)
    if require_data:
        required_paths = [
            NFHS5_PATH,
            SHAPEFILE_PATH,
            ASPIRATIONAL_PATH,
        ]
        ok, missing = paths_exist(*required_paths)
        if not ok:
            missing_str = '\n  '.join(missing)
            raise FileNotFoundError(
                f'Required data files are missing:\n  {missing_str}\n'
                'Complete Steps 8–10 (DHS download + shapefile) before running '
                'data-processing scripts.'
            )
 
