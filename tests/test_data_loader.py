"""
tests/test_data_loader.py — Test suite for src/data_loader.py.
 
Markers:
    @pytest.mark.unit        — runs instantly, no data files required
    @pytest.mark.integration — requires NFHS5_PATH to exist on disk
 
Run unit tests only (fast, default):
    pytest tests/test_data_loader.py -m unit -v
 
Run integration tests (requires .DTA file):
    pytest tests/test_data_loader.py -m integration -v
 
Run all tests:
    pytest tests/test_data_loader.py -v
"""
 
import numpy as np
import pandas as pd
import pytest
 
from src.config import (
    MISSING_CODES, STUNTING_THRESHOLD, UNDERWEIGHT_THRESHOLD, WASTING_THRESHOLD,
    Z_SCORE_MIN, Z_SCORE_MAX, MIN_VALID_ROWS,
    EXPECTED_STUNTING_RANGE, EXPECTED_UNDERWEIGHT_RANGE, EXPECTED_WASTING_RANGE,
    NFHS5_PATH, NFHS_COLS,
)
from src.data_loader import load_nfhs5_kr, create_labels, load_and_label
 
 
# ─────────────────────────────────────────────────────────────────────────
# Shared test fixtures
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_df_with_missing_codes() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame mimicking NFHS raw output before any cleaning.
    Includes DHS missing codes (9998) and Z-scores in x100 integer encoding.
    """
    return pd.DataFrame({
        'HW70': [-253, -187, 9998, 201, -350],   # HAZ x100 with one missing code
        'HW71': [-195, -310, -120, 9999, -280],   # WAZ x100 with one missing code
        'HW72': [55,   -230,  180, -198, 9997],   # WHZ x100 with one missing code
        'V106': [2, 0, 1, 3, 2],                  # Mother education
        'HV270': [1, 3, 2, 5, 1],                 # Wealth quintile
        'B4': [1, 2, 1, 2, 1],                    # Sex of child
    })
 
 
@pytest.fixture
def scaled_df() -> pd.DataFrame:
    """
    Synthetic DataFrame with Z-scores already scaled to float range,
    matching the output of load_nfhs5_kr().
    """
    return pd.DataFrame({
        'HAZ': [-2.53, -1.87, np.nan,  2.01, -3.50],
        'WAZ': [-1.95, -3.10, -1.20, np.nan, -2.80],
        'WHZ': [0.55,  -2.30,  1.80,  -1.98, np.nan],
        'V106': [2, 0, 1, 3, 2],
        'HV270': [1, 3, 2, 5, 1],
        'B4': [1, 2, 1, 2, 1],
    })
 
 
# ─────────────────────────────────────────────────────────────────────────
# Unit tests: missing code replacement
# ─────────────────────────────────────────────────────────────────────────
class TestMissingCodeReplacement:
 
    @pytest.mark.unit
    def test_dhs_codes_become_nan(self, raw_df_with_missing_codes):
        """Values 9996-9999 in any numeric column must be NaN after loading."""
        # Simulate the missing code replacement logic from load_nfhs5_kr()
        df = raw_df_with_missing_codes.copy()
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
 
        # All DHS missing codes must be gone
        for col in ['HW70', 'HW71', 'HW72']:
            remaining = df[col].isin(MISSING_CODES).sum()
            assert remaining == 0, (
                f'{col} still has {remaining} DHS missing codes after replacement'
            )
 
    @pytest.mark.unit
    def test_valid_values_preserved(self, raw_df_with_missing_codes):
        """Non-missing values must not be altered by the replacement."""
        df = raw_df_with_missing_codes.copy()
        valid_hw70_before = df.loc[~df['HW70'].isin(MISSING_CODES), 'HW70'].tolist()
 
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
 
        valid_hw70_after = df['HW70'].dropna().tolist()
        assert valid_hw70_before == valid_hw70_after, (
            'Valid values were altered during missing code replacement'
        )
 
    @pytest.mark.unit
    def test_five_digit_codes_replaced(self):
        """5-digit DHS variants (99996-99999) must also become NaN."""
        df = pd.DataFrame({'HW70': [99998, 99999, -253]})
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
        assert df['HW70'].isnull().sum() == 2, '5-digit missing codes not replaced'
        assert df['HW70'].dropna().iloc[0] == -253, 'Valid value altered'
 
 
# ─────────────────────────────────────────────────────────────────────────
# Unit tests: Z-score scaling
# ─────────────────────────────────────────────────────────────────────────
class TestZScoreScaling:
 
    @pytest.mark.unit
    def test_division_by_100(self, raw_df_with_missing_codes):
        """HW70 = -253 must become HAZ = -2.53 after scaling."""
        df = raw_df_with_missing_codes.copy()
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
        df['HAZ'] = df['HW70'] / 100.0
 
        # Row 0: HW70=-253 → HAZ=-2.53
        assert df['HAZ'].iloc[0] == pytest.approx(-2.53, abs=0.001), (
            f'Expected -2.53, got {df["HAZ"].iloc[0]}'
        )
 
    @pytest.mark.unit
    def test_z_scores_in_physiological_range(self, raw_df_with_missing_codes):
        """After scaling, all valid Z-scores must be between Z_SCORE_MIN and Z_SCORE_MAX."""
        df = raw_df_with_missing_codes.copy()
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
        df['HAZ'] = df['HW70'] / 100.0
 
        valid = df['HAZ'].dropna()
        out_of_range = valid[~valid.between(Z_SCORE_MIN, Z_SCORE_MAX)]
        assert len(out_of_range) == 0, (
            f'Z-scores outside [{Z_SCORE_MIN}, {Z_SCORE_MAX}] after scaling: {out_of_range.tolist()}'
        )
 
    @pytest.mark.unit
    def test_nan_preserved_after_scaling(self, raw_df_with_missing_codes):
        """NaN values introduced by missing code replacement must survive scaling."""
        df = raw_df_with_missing_codes.copy()
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
        null_before = df['HW70'].isnull().sum()
        df['HAZ'] = df['HW70'] / 100.0
        assert df['HAZ'].isnull().sum() == null_before, (
            'NaN count changed during Z-score scaling — division by 100 should not drop NaN'
        )
 
 
# ─────────────────────────────────────────────────────────────────────────
# Unit tests: label creation
# ─────────────────────────────────────────────────────────────────────────
class TestLabelCreation:
 
    @pytest.mark.unit
    def test_stunted_threshold(self, scaled_df):
        """HAZ < STUNTING_THRESHOLD → stunted = 1, else stunted = 0."""
        df = create_labels(scaled_df)
        # Row 0: HAZ=-2.53 < -2.0 → stunted=1
        assert df['stunted'].iloc[0] == 1, 'Row 0 (HAZ=-2.53) should be stunted=1'
        # Row 1: HAZ=-1.87 >= -2.0 → stunted=0
        assert df['stunted'].iloc[1] == 0, 'Row 1 (HAZ=-1.87) should be stunted=0'
 
    @pytest.mark.unit
    def test_underweight_threshold(self, scaled_df):
        """WAZ < UNDERWEIGHT_THRESHOLD → underweight = 1."""
        df = create_labels(scaled_df)
        # Row 1: WAZ=-3.10 < -2.0 → underweight=1
        assert df['underweight'].iloc[1] == 1, 'Row 1 (WAZ=-3.10) should be underweight=1'
 
    @pytest.mark.unit
    def test_wasted_threshold(self, scaled_df):
        """WHZ < WASTING_THRESHOLD → wasted = 1."""
        df = create_labels(scaled_df)
        # Row 1: WHZ=-2.30 < -2.0 → wasted=1
        assert df['wasted'].iloc[1] == 1, 'Row 1 (WHZ=-2.30) should be wasted=1'
        # Row 0: WHZ=0.55 >= -2.0 → wasted=0
        assert df['wasted'].iloc[0] == 0, 'Row 0 (WHZ=0.55) should be wasted=0'
 
    @pytest.mark.unit
    def test_labels_are_binary_integers(self, scaled_df):
        """Label columns must contain only 0 and 1 (integer type, no NaN)."""
        df = create_labels(scaled_df)
        for col in ['stunted', 'underweight', 'wasted']:
            unique_values = set(df[col].dropna().unique())
            assert unique_values.issubset({0, 1}), (
                f'{col} contains non-binary values: {unique_values}'
            )
            assert df[col].isnull().sum() == 0, (
                f'{col} has {df[col].isnull().sum()} NaN values'
            )
 
    @pytest.mark.unit
    def test_all_nan_rows_dropped(self):
        """Rows where ALL three Z-scores are NaN must be dropped."""
        df_with_all_nan = pd.DataFrame({
            'HAZ': [np.nan, -2.5, np.nan],
            'WAZ': [np.nan, -3.0, -1.5],
            'WHZ': [np.nan, -2.1, 0.5],
        })
        # Row 0 has all three Z-scores as NaN → should be dropped
        # Row 2 has only HAZ as NaN → should be kept
        result = create_labels(df_with_all_nan)
        assert len(result) == 2, (
            f'Expected 2 rows after dropping all-NaN row, got {len(result)}'
        )
 
    @pytest.mark.unit
    def test_partial_nan_row_kept(self):
        """Row with only one or two Z-scores missing must be kept."""
        df_partial = pd.DataFrame({
            'HAZ': [np.nan],   # Only HAZ is NaN
            'WAZ': [-2.5],
            'WHZ': [0.5],
        })
        result = create_labels(df_partial)
        assert len(result) == 1, 'Partial-NaN row should be kept'
        assert result['underweight'].iloc[0] == 1, 'WAZ=-2.5 should be underweight=1'
 
    @pytest.mark.unit
    def test_missing_z_column_raises_key_error(self, scaled_df):
        """create_labels() must raise KeyError if HAZ/WAZ/WHZ columns are absent."""
        df_bad = scaled_df.drop(columns=['HAZ'])
        with pytest.raises(KeyError, match='HAZ'):
            create_labels(df_bad)
 
    @pytest.mark.unit
    def test_minimum_row_count_enforced(self):
        """ValueError must be raised when fewer than MIN_VALID_ROWS rows remain."""
        # Create a tiny DataFrame — well below MIN_VALID_ROWS
        tiny_df = pd.DataFrame({
            'HAZ': [-2.5] * 20,
            'WAZ': [-3.0] * 20,
            'WHZ': [-2.2] * 20,
        })
        with pytest.raises(ValueError, match='Only .* rows remain'):
            create_labels(tiny_df)
 
 
# ─────────────────────────────────────────────────────────────────────────
# Unit tests: FileNotFoundError handling
# ─────────────────────────────────────────────────────────────────────────
class TestFileHandling:
 
    @pytest.mark.unit
    def test_missing_dta_raises_file_not_found(self):
        """load_nfhs5_kr() must raise FileNotFoundError for a non-existent path."""
        from pathlib import Path
        with pytest.raises(FileNotFoundError):
            load_nfhs5_kr(path=Path('data/raw/nfhs5/DOES_NOT_EXIST.DTA'))
 
    @pytest.mark.unit
    def test_error_message_contains_helpful_text(self):
        """FileNotFoundError message must contain the missing path."""
        from pathlib import Path
        with pytest.raises(FileNotFoundError, match='DOES_NOT_EXIST'):
            load_nfhs5_kr(path=Path('data/raw/nfhs5/DOES_NOT_EXIST.DTA'))
 
 
# ─────────────────────────────────────────────────────────────────────────
# Integration tests: real NFHS-5 .DTA file
# Run with: pytest tests/test_data_loader.py -m integration
# Requires: data/raw/nfhs5/IAKR7EFL.DTA to exist on disk
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestIntegrationNFHS5:
 
    @pytest.fixture(scope='class')
    def labelled_df(self):
        """Load and label the full NFHS-5 file once for all integration tests."""
        if not NFHS5_PATH.exists():
            pytest.skip(
                f'NFHS-5 .DTA file not found at {NFHS5_PATH}. '
                'Complete DHS data download (Step 8 data acquisition) first.'
            )
        return load_and_label(path=NFHS5_PATH, columns=NFHS_COLS)
 
    def test_row_count_above_minimum(self, labelled_df):
        """Full NFHS-5 file must yield at least MIN_VALID_ROWS records."""
        assert len(labelled_df) >= MIN_VALID_ROWS, (
            f'Only {len(labelled_df):,} rows — expected >= {MIN_VALID_ROWS:,}'
        )
 
    def test_no_dhs_missing_codes_remain(self, labelled_df):
        """No DHS missing codes (9996-9999) must remain after loading."""
        for col in labelled_df.select_dtypes(include='number').columns:
            remaining = labelled_df[col].isin(MISSING_CODES).sum()
            assert remaining == 0, (
                f'{col}: {remaining:,} DHS missing codes remain'
            )
 
    def test_z_scores_in_physiological_range(self, labelled_df):
        """All valid Z-scores must be in [Z_SCORE_MIN, Z_SCORE_MAX] = [-6, +6]."""
        for z_col in ['HAZ', 'WAZ', 'WHZ']:
            valid = labelled_df[z_col].dropna()
            out = valid[~valid.between(Z_SCORE_MIN, Z_SCORE_MAX)]
            assert len(out) == 0, (
                f'{z_col}: {len(out):,} values outside [{Z_SCORE_MIN}, {Z_SCORE_MAX}]'
            )
 
    def test_stunting_prevalence_matches_nfhs5_report(self, labelled_df):
        """Stunting prevalence must match NFHS-5 national figure of ~35.5%."""
        prev = float(labelled_df['stunted'].mean())
        lo, hi = EXPECTED_STUNTING_RANGE
        assert lo <= prev <= hi, (
            f'Stunting {prev:.1%} outside expected range [{lo:.0%}, {hi:.0%}]. '
            'Check missing code replacement and Z-score scaling.'
        )
 
    def test_underweight_prevalence_matches_nfhs5_report(self, labelled_df):
        """Underweight prevalence must match NFHS-5 national figure of ~32.1%."""
        prev = float(labelled_df['underweight'].mean())
        lo, hi = EXPECTED_UNDERWEIGHT_RANGE
        assert lo <= prev <= hi, (
            f'Underweight {prev:.1%} outside expected range [{lo:.0%}, {hi:.0%}]'
        )
 
    def test_wasting_prevalence_matches_nfhs5_report(self, labelled_df):
        """Wasting prevalence must match NFHS-5 national figure of ~19.3%."""
        prev = float(labelled_df['wasted'].mean())
        lo, hi = EXPECTED_WASTING_RANGE
        assert lo <= prev <= hi, (
            f'Wasting {prev:.1%} outside expected range [{lo:.0%}, {hi:.0%}]'
        )
 
    def test_labels_are_binary(self, labelled_df):
        """All three label columns must contain only 0 and 1 in the real data."""
        for col in ['stunted', 'underweight', 'wasted']:
            unique = set(labelled_df[col].unique())
            assert unique.issubset({0, 1}), f'{col} has non-binary values: {unique}'
 
    def test_required_columns_present(self, labelled_df):
        """All columns from NFHS_COLS (renamed) and all three target columns must exist."""
        # Z-score columns are renamed: HW70 → HAZ etc.
        expected = ['HAZ', 'WAZ', 'WHZ', 'stunted', 'underweight', 'wasted']
        for col in expected:
            assert col in labelled_df.columns, (
                f"Required column '{col}' missing from loaded DataFrame"
            )
