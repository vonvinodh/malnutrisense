"""
src/preprocessing.py — Feature cleaning and preparation pipeline.
 
Transforms the output of data_loader.load_and_label() into a
model-ready feature matrix. All transformations are DHS-aware:
  - Categorical encoding uses DHS codebooks, not generic label encoding
  - SC/ST column is detected across NFHS version variants and preserved
  - Imputation excludes sensitive identity variables
  - Every transformation step writes to the CleaningLogger
 
Public API:
    detect_sc_st_column(df)        → str | None
    encode_categorical(df)         → DataFrame
    impute_missing(df)             → DataFrame
    build_socioeconomic_index(df)  → DataFrame
    run_full_pipeline(df)          → DataFrame
    make_train_test_split(df)      → tuple[4 DataFrames]
"""
 
import logging
from typing import Optional
 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
 
from src.config import (
    RANDOM_STATE, TEST_SIZE, TARGET_COLS,
    BIRTH_WEIGHT_MIN, BIRTH_WEIGHT_MAX,
)
from src.logger import CleaningLogger, get_console_logger
from src.utils import assert_columns_exist, profile_dataframe
 
log         = get_console_logger(__name__)
cleaning_log = CleaningLogger()
 
 
# ── Known SC/ST column variants across DHS / NFHS versions ───────────────
# DHS uses different variable names for Scheduled Caste / Scheduled Tribe
# status across country files, state sub-files, and survey rounds.
# This list covers all known variants in NFHS-4 and NFHS-5.
_SC_ST_CANDIDATES: list[str] = [
    'V131',   # Most common in NFHS-5 national KR file
    'S116',   # State-level NFHS-5 supplement files
    'SH46',   # Older NFHS-4 state modules
    'V130',   # Religion — proxied for caste in some analyses (not ideal)
]
 
# ── DHS categorical encodings ─────────────────────────────────────────────
# Maps DHS integer codes to readable string labels.
# Source: DHS Model Questionnaire codebooks (India-specific recode).
_ENCODINGS: dict[str, dict] = {
    'B4': {1: 'male', 2: 'female'},
    'V025': {1: 'urban', 2: 'rural'},
    'V106': {
        0: 'no_education',
        1: 'primary',
        2: 'secondary',
        3: 'higher',
    },
    'HV201': {
        10: 'piped_on_premises', 11: 'piped_on_premises',
        12: 'piped_to_neighbour', 13: 'public_tap',
        20: 'tube_well', 21: 'tube_well',
        30: 'dug_well', 31: 'protected_well', 32: 'unprotected_well',
        40: 'surface_water', 41: 'surface_water',
        51: 'rainwater', 61: 'tanker', 71: 'bottled_water',
        96: 'other',
    },
    'HV205': {
        10: 'flush_piped', 11: 'flush_septic', 12: 'flush_pit',
        13: 'flush_open', 14: 'flush_unknown',
        20: 'pit_ventilated', 21: 'pit_with_slab', 22: 'pit_without_slab',
        30: 'composting', 31: 'composting',
        41: 'hanging_toilet', 42: 'bucket_toilet',
        96: 'other', 99: 'unknown',
    },
}
 
 
# ── SC/ST column detection ────────────────────────────────────────────────
def detect_sc_st_column(df: pd.DataFrame) -> Optional[str]:
    """
    Search for the SC/ST variable across all known NFHS column name variants.
 
    The Scheduled Caste / Scheduled Tribe column has different names across
    NFHS versions (V131, S116, SH46) and may be absent from some state files.
    This function finds whichever variant is present and returns its name.
 
    Args:
        df: DataFrame from load_and_label().
 
    Returns:
        Column name (str) if found, or None if no variant is present.
 
    Example:
        sc_st_col = detect_sc_st_column(df)
        if sc_st_col:
            print(f'SC/ST column: {sc_st_col}')
    """
    for candidate in _SC_ST_CANDIDATES:
        if candidate in df.columns:
            log.info(f'SC/ST column detected: {candidate}')
            cleaning_log.log(
                dataset='nfhs5_kr',
                step='detect_sc_st_column',
                column_affected=candidate,
                issue_found='SC/ST column name varies by NFHS version',
                action_taken=f'Using {candidate} as SC/ST proxy for fairness audit',
                rows_affected=-1,
                validation_result='INFO',
                analyst_notes=f'Candidates searched: {_SC_ST_CANDIDATES}',
            )
            return candidate
    log.warning(
        f'No SC/ST column found. Searched: {_SC_ST_CANDIDATES}. '
        'Fairness audit for Scheduled Tribe subgroup will be unavailable. '
        'Consider requesting the Household Members Recode (PR) file from DHS.'
    )
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='detect_sc_st_column',
        column_affected='N/A',
        issue_found=f'No SC/ST column found. Searched: {_SC_ST_CANDIDATES}',
        action_taken='Fairness audit ST subgroup will be skipped',
        rows_affected=-1,
        validation_result='INFO',
        analyst_notes='Request PR (Household Members Recode) file for V131',
    )
    return None
 
 
# ── Categorical encoding ──────────────────────────────────────────────────
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace DHS integer codes with readable string labels.
 
    Applies DHS codebook mappings from _ENCODINGS. Columns not in _ENCODINGS
    are left unchanged. After mapping, unmapped codes (integers not in the
    dictionary) become NaN — these are handled by the imputation step.
 
    Preserves the SC/ST column in its original integer encoding. The fairness
    auditor (Objective 4) expects integer codes for grouping operations.
 
    Args:
        df: DataFrame with raw DHS integer codes.
 
    Returns:
        DataFrame with categorical columns replaced by string labels.
    """
    df = df.copy()
    encoded_cols = []
 
    for col, mapping in _ENCODINGS.items():
        if col not in df.columns:
            continue
 
        before_nulls = int(df[col].isnull().sum())
        df[col] = df[col].map(mapping)
        after_nulls  = int(df[col].isnull().sum())
        new_nulls    = after_nulls - before_nulls
        encoded_cols.append(col)
 
        if new_nulls > 0:
            log.warning(
                f'{col}: {new_nulls:,} values became NaN after encoding '
                '(unmapped DHS codes). Will be imputed as mode.'
            )
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='encode_categorical',
        column_affected=', '.join(encoded_cols),
        issue_found='DHS integer codes not interpretable by model',
        action_taken='Mapped integers to string labels per DHS codebook',
        rows_affected=len(df),
        validation_result='PASS',
        analyst_notes=f'Columns encoded: {encoded_cols}',
    )
    log.info(f'Categorical encoding applied to: {encoded_cols}')
    return df
 
 
# ── Missing value imputation ──────────────────────────────────────────────
def impute_missing(
    df: pd.DataFrame,
    sc_st_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Impute missing values using DHS-appropriate strategies.
 
    Strategy per column type:
      - Numeric predictors (age, birth weight, breastfeeding): median imputation
      - Categorical predictors (water source, toilet type, education): mode imputation
      - Z-score columns (HAZ, WAZ, WHZ): NOT imputed — NaN = no measurement taken
      - Target labels (stunted, underweight, wasted): NOT imputed — binary labels
        created from Z-scores; if Z-score is NaN, label stays NaN (excluded later)
      - SC/ST column (if detected): NOT imputed — NaN means not collected.
        Imputing a caste or tribe identity is ethically inappropriate.
      - Wealth quintile (HV270): NOT imputed — always present in NFHS data.
 
    Args:
        df:        DataFrame after encode_categorical().
        sc_st_col: Column name of SC/ST variable (from detect_sc_st_column()).
                   Excluded from imputation even if it has missing values.
 
    Returns:
        DataFrame with imputed feature columns. Protected columns unchanged.
    """
    df = df.copy()
 
    # Columns to never impute
    # Z-scores and labels: NaN has semantic meaning (no measurement / no target)
    # SC/ST: imputing identity is ethically inappropriate
    never_impute: set[str] = {'HAZ', 'WAZ', 'WHZ',
                               'stunted', 'underweight', 'wasted'}
    if sc_st_col:
        never_impute.add(sc_st_col)
 
    # Separate numeric and categorical feature columns for different strategies
    numeric_feats = [
        c for c in df.select_dtypes(include='number').columns
        if c not in never_impute
    ]
    categoric_feats = [
        c for c in df.select_dtypes(include='object').columns
        if c not in never_impute
    ]
 
    # ── Numeric: median imputation ─────────────────────────────────────
    # Median is preferred over mean for health data — skewed distributions
    # (birth weight, breastfeeding duration) inflate the mean unfairly.
    if numeric_feats:
        null_counts_before = {c: int(df[c].isnull().sum()) for c in numeric_feats
                              if df[c].isnull().sum() > 0}
        imp_median = SimpleImputer(strategy='median')
        df[numeric_feats] = imp_median.fit_transform(df[numeric_feats])
        log.info(f'Median imputation applied to {len(numeric_feats)} numeric columns')
 
        if null_counts_before:
            cleaning_log.log(
                dataset='nfhs5_kr',
                step='impute_numeric_median',
                column_affected=', '.join(null_counts_before.keys()),
                issue_found=f'Missing values in numeric features: {null_counts_before}',
                action_taken='SimpleImputer(strategy=median) applied',
                rows_affected=sum(null_counts_before.values()),
                validation_result='PASS',
                analyst_notes=
                    'Median chosen over mean: birth weight and breastfeeding are right-skewed',
            )
 
    # ── Categorical: mode (most frequent) imputation ───────────────────
    # Mode preserves the most common value — appropriate for nominal categories
    # like water source and toilet type where 'middle' has no meaning.
    if categoric_feats:
        null_counts_cat = {c: int(df[c].isnull().sum()) for c in categoric_feats
                           if df[c].isnull().sum() > 0}
        imp_mode = SimpleImputer(strategy='most_frequent')
        df[categoric_feats] = imp_mode.fit_transform(df[categoric_feats])
        log.info(f'Mode imputation applied to {len(categoric_feats)} categorical columns')
 
        if null_counts_cat:
            cleaning_log.log(
                dataset='nfhs5_kr',
                step='impute_categorical_mode',
                column_affected=', '.join(null_counts_cat.keys()),
                issue_found=f'Missing values in categorical features: {null_counts_cat}',
                action_taken='SimpleImputer(strategy=most_frequent) applied',
                rows_affected=sum(null_counts_cat.values()),
                validation_result='PASS',
                analyst_notes='Mode preserves most common nominal category',
            )
 
    # Log the protected columns
    log.info(f'Not imputed (protected): {sorted(never_impute)}')
    return df
 
 
# ── Birth weight outlier capping ──────────────────────────────────────────
def cap_birth_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap M19 (birth weight in grams) at physiological limits.
 
    Values outside [BIRTH_WEIGHT_MIN, BIRTH_WEIGHT_MAX] = [500g, 5000g]
    are data entry errors or DHS coding artefacts and must be clipped.
    Clipping is applied after imputation so it acts on the filled values.
 
    Args:
        df: DataFrame after impute_missing().
 
    Returns:
        DataFrame with M19 clipped to physiological range.
    """
    if 'M19' not in df.columns:
        log.warning('M19 (birth weight) column not found — skipping outlier cap')
        return df
 
    df = df.copy()
    outliers_low  = int((df['M19'] < BIRTH_WEIGHT_MIN).sum())
    outliers_high = int((df['M19'] > BIRTH_WEIGHT_MAX).sum())
    total_outliers = outliers_low + outliers_high
 
    df['M19'] = df['M19'].clip(lower=BIRTH_WEIGHT_MIN, upper=BIRTH_WEIGHT_MAX)
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='cap_birth_weight',
        column_affected='M19',
        issue_found=f'{total_outliers:,} values outside [{BIRTH_WEIGHT_MIN},{BIRTH_WEIGHT_MAX}]g',
        action_taken=f'clip(lower={BIRTH_WEIGHT_MIN}, upper={BIRTH_WEIGHT_MAX})',
        rows_affected=total_outliers,
        validation_result='PASS',
        analyst_notes=(
            f'{outliers_low:,} below {BIRTH_WEIGHT_MIN}g, '
            f'{outliers_high:,} above {BIRTH_WEIGHT_MAX}g capped'
        ),
    )
    log.info(f'Birth weight capped: {total_outliers:,} values clipped to [{BIRTH_WEIGHT_MIN},{BIRTH_WEIGHT_MAX}]')
    return df
 
 
# ── Socioeconomic deprivation index ──────────────────────────────────────
def build_socioeconomic_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a composite Socioeconomic Deprivation Index (SDI) feature.
 
    The SDI is a normalised composite of wealth quintile, maternal education,
    sanitation access, and water source — the four structural determinants most
    strongly associated with child malnutrition in the NFHS literature.
 
    Construction method:
      1. Recode each component to a 0 (deprived) → 1 (not deprived) scale
      2. Average the four components equally
      3. Invert so that SDI=1 = maximum deprivation (aligns with outcome direction)
 
    The SDI is an additional feature — it does not replace the individual
    component columns, which remain in the DataFrame.
 
    Args:
        df: DataFrame after impute_missing() and encode_categorical().
 
    Returns:
        DataFrame with new column 'sdi' (float, 0.0 = least deprived, 1.0 = most).
    """
    df = df.copy()
    components: list[pd.Series] = []
 
    # Component 1: Wealth quintile (HV270)
    # 1=Poorest → 5=Richest. Normalise to 0-1, invert so 1=poorest.
    if 'HV270' in df.columns:
        wq = pd.to_numeric(df['HV270'], errors='coerce')
        wq_norm = (wq - 1) / 4.0       # maps 1→0, 5→1
        components.append(1.0 - wq_norm)  # invert: 1=poorest
 
    # Component 2: Maternal education (V106)
    # 0=None → 3=Higher. Normalise to 0-1, invert so 1=no education.
    if 'V106' in df.columns:
        # After encode_categorical(), V106 is a string. Map back to ordinal.
        edu_map = {'no_education': 0, 'primary': 1, 'secondary': 2, 'higher': 3}
        edu_num = df['V106'].map(edu_map)
        edu_norm = edu_num / 3.0
        components.append(1.0 - edu_norm)  # invert: 1=no education
 
    # Component 3: Sanitation (HV205)
    # Group into 'improved' vs 'unimproved' based on WHO/JMP classification.
    if 'HV205' in df.columns:
        improved_sanitation = {
            'flush_piped', 'flush_septic', 'flush_pit',
            'pit_ventilated', 'pit_with_slab', 'composting',
        }
        sanitation_deprived = (~df['HV205'].isin(improved_sanitation)).astype(float)
        components.append(sanitation_deprived)  # 1=deprived (unimproved/none)
 
    # Component 4: Water source (HV201)
    # Group into 'improved' vs 'unimproved' per WHO/JMP classification.
    if 'HV201' in df.columns:
        improved_water = {
            'piped_on_premises', 'piped_to_neighbour', 'public_tap',
            'tube_well', 'protected_well', 'bottled_water', 'rainwater',
        }
        water_deprived = (~df['HV201'].isin(improved_water)).astype(float)
        components.append(water_deprived)  # 1=deprived
 
    if not components:
        log.warning('No SDI components found — skipping index construction')
        return df
 
    # Average the available components into a single index
    sdi = pd.concat(components, axis=1).mean(axis=1)
    df['sdi'] = sdi.round(4)
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='build_socioeconomic_index',
        column_affected='sdi (new)',
        issue_found='No composite deprivation feature for model',
        action_taken=f'SDI constructed from {len(components)} components',
        rows_affected=len(df),
        validation_result='PASS',
        analyst_notes='0=least deprived, 1=most deprived. Inversion applied.',
    )
    log.info(f'SDI built from {len(components)} components. Mean={df["sdi"].mean():.3f}')
    return df
 
 
# ── Remove duplicates ────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows and log the count removed.
 
    In NFHS data, true duplicates are rare but can occur if the same household
    was sampled in multiple cluster rounds or if the file was accidentally
    concatenated. Deduplication runs before the train/test split to prevent
    the same child appearing in both train and test sets.
    """
    df = df.copy()
    rows_before = len(df)
    df = df.drop_duplicates()
    removed = rows_before - len(df)
 
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='remove_duplicates',
        column_affected='all',
        issue_found=f'{removed:,} exact duplicate rows detected',
        action_taken='drop_duplicates() applied',
        rows_affected=removed,
        validation_result='PASS',
        analyst_notes='Exact row deduplication only — no fuzzy matching',
    )
    log.info(f'Duplicates removed: {removed:,}. Rows remaining: {len(df):,}')
    return df
 
 
# ── Full pipeline orchestrator ────────────────────────────────────────────
def run_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all preprocessing steps in the correct fixed order.
 
    Step order (must not be changed):
      1. remove_duplicates  — before any feature changes
      2. detect_sc_st_column — identify before encode changes column types
      3. encode_categorical  — convert DHS codes to strings
      4. impute_missing       — fill NaN in features (excludes SC/ST and targets)
      5. cap_birth_weight     — clip M19 outliers after imputation
      6. build_socioeconomic_index — composite feature from cleaned components
 
    Args:
        df: Output of data_loader.load_and_label().
 
    Returns:
        Fully cleaned, feature-engineered DataFrame ready for label extraction
        and train/test split.
    """
    log.info('Starting full preprocessing pipeline...')
    log.info(f'Input: {len(df):,} rows x {df.shape[1]} columns')
 
    df = remove_duplicates(df)
    sc_st_col = detect_sc_st_column(df)
    df = encode_categorical(df)
    df = impute_missing(df, sc_st_col=sc_st_col)
    df = cap_birth_weight(df)
    df = build_socioeconomic_index(df)
    df = remove_duplicates(df)  # Remove any duplicates created by imputation
 
    log.info(f'Pipeline complete. Output: {len(df):,} rows x {df.shape[1]} columns')
    return df
 
 
# ── Train/test split ─────────────────────────────────────────────────────
def make_train_test_split(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split preserving label distribution.
 
    Stratification is applied on the 'stunted' column because it is the
    most common phenotype (~35%) and its class ratio is most important to
    preserve across both splits. XGBoost multi-output handles the other
    two labels without requiring stratification on each independently.
 
    Args:
        df:           Fully preprocessed DataFrame.
        feature_cols: Columns to include in X. If None, uses all columns
                      except TARGET_COLS and Z-score columns.
 
    Returns:
        (X_train, X_test, y_train, y_test) as separate DataFrames.
    """
    assert_columns_exist(df, TARGET_COLS, 'preprocessed DataFrame')
 
    # Auto-select feature columns if not provided
    exclude = set(TARGET_COLS) | {'HAZ', 'WAZ', 'WHZ'}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude]

    rows_before = len(df)
    df = df.dropna(subset=TARGET_COLS).copy()
    dropped = rows_before - len(df)
    if dropped > 0:
        log.warning(
            f'Dropped {dropped:,} rows with missing target labels before train/test split.'
        )
        cleaning_log.log(
            dataset='nfhs5_kr',
            step='drop_missing_labels',
            column_affected=', '.join(TARGET_COLS),
            issue_found=f'{dropped:,} missing target labels',
            action_taken='Dropped rows with NaN in target labels before stratified split',
            rows_affected=dropped,
            validation_result='PASS',
            analyst_notes='Required for sklearn train_test_split stratify on stunted',
        )

    X = df[feature_cols]
    y = df[TARGET_COLS]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['stunted'],  # Preserve class ratio in both splits
    )
 
    log.info(
        f'Train/test split: {len(X_train):,} train | {len(X_test):,} test '
        f'({TEST_SIZE:.0%} test, stratified on stunted)'
    )
    cleaning_log.log(
        dataset='nfhs5_kr',
        step='train_test_split',
        column_affected='all',
        issue_found='N/A — split step',
        action_taken=(
            f'Stratified split: {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test, '
            f'RANDOM_STATE={RANDOM_STATE}'
        ),
        rows_affected=len(df),
        validation_result='PASS',
        analyst_notes=(
            f'Train: {len(X_train):,} rows, Test: {len(X_test):,} rows'
        ),
    )
    return X_train, X_test, y_train, y_test
 
