"""
src/fairness.py — Equity auditing for Objective 4.
 
Stratifies false-negative rates by demographic subgroups using Fairlearn.
Applies threshold correction to reduce FNR disparity across groups.
 
Public API:
  FairnessAuditor(model, X_test, y_test, df_test) -- class
  .compute_fnr_by_group(label, sensitive_col)     -> MetricFrame
  .audit_all_groups(label)                         -> dict
  .apply_threshold_correction(label, target_recall)-> dict
  .build_equity_report()                           -> DataFrame
  .save_equity_report(df)                          -> Path
"""
 
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
 
from fairlearn.metrics import MetricFrame, false_negative_rate, equalized_odds_difference
from sklearn.metrics import recall_score
 
from src.config import TARGET_COLS, TABLES_DIR
from src.logger import get_console_logger, CleaningLogger
from src.utils import format_number
 
log  = get_console_logger(__name__)
clog = CleaningLogger()
 
# Demographic columns used as sensitive features in the fairness audit.
# Each must be present in df_test (merged from the cleaned dataset).
SENSITIVE_FEATURES = {
    'wealth_quintile':    'HV270',    # 1=Poorest to 5=Richest
    'child_sex':          'B4',       # 1=Male / 2=Female
    'aspirational_dist':  'is_aspirational',  # 0/1 flag
    'sc_st_group':        None,       # detected dynamically by detect_sc_st_column()
}
 
# FNR tolerance threshold: subgroups above this trigger threshold correction
FNR_TOLERANCE = 0.15   # 15% false-negative rate maximum
 
 
class FairnessAuditor:
    """
    Equity auditor for the MLTP model.
 
    Usage:
        auditor = FairnessAuditor(mltp_model, X_test, y_test, df_with_demographics)
        report = auditor.build_equity_report()
        auditor.save_equity_report(report)
    """
 
    def __init__(
        self,
        model,
        X_test:  pd.DataFrame,
        y_test:  pd.DataFrame,
        df_meta: pd.DataFrame,
    ) -> None:
        """
        Args:
            model:   Fitted MLTP Pipeline.
            X_test:  Feature matrix (test split).
            y_test:  Label DataFrame (test split, stunted/underweight/wasted).
            df_meta: DataFrame aligned with X_test/y_test containing
                     demographic columns (HV270, B4, is_aspirational, SC/ST).
        """
        self.model   = model
        self.X_test  = X_test.reset_index(drop=True)
        self.y_test  = y_test.reset_index(drop=True)
        self.df_meta = df_meta.reset_index(drop=True)
 
        # Compute predict_proba once for all labels
        self._proba_list = model.predict_proba(X_test)  # list of n_labels arrays
        log.info(f'FairnessAuditor initialised: {len(X_test):,} test rows')
 
    def _get_proba(self, label: str) -> np.ndarray:
        """Positive-class probability for a given label."""
        idx = TARGET_COLS.index(label)
        return self._proba_list[idx][:, 1]
 
    def compute_fnr_by_group(
        self,
        label: str,
        sensitive_col: str,
        threshold: float = 0.5,
    ) -> MetricFrame:
        """
        Compute false-negative rate per subgroup using Fairlearn MetricFrame.
 
        Args:
            label:         Phenotype to evaluate (e.g. 'stunted').
            sensitive_col: Column in df_meta defining the demographic groups.
            threshold:     Classification threshold (default 0.5).
 
        Returns:
            Fairlearn MetricFrame with FNR per subgroup.
        """
        if sensitive_col not in self.df_meta.columns:
            log.warning(f'{sensitive_col} not in df_meta — skipping')
            return None
 
        mask    = self.y_test[label].notna()
        y_true  = self.y_test.loc[mask, label].astype(int)
        y_pred  = (self._get_proba(label)[mask] >= threshold).astype(int)
        groups  = self.df_meta.loc[mask, sensitive_col]
 
        mf = MetricFrame(
            metrics={'false_negative_rate': false_negative_rate,
                     'recall': recall_score},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=groups,
        )
        return mf
 
    def audit_all_groups(
        self,
        label: str,
        threshold: float = 0.5,
    ) -> dict:
        """
        Run FNR audit across all four sensitive feature categories.
        Returns dict mapping sensitive_feature_name -> MetricFrame.
        """
        results = {}
        for feat_name, col in SENSITIVE_FEATURES.items():
            if col is None:
                continue
            mf = self.compute_fnr_by_group(label, col, threshold)
            if mf is not None:
                results[feat_name] = mf
                log.info(f'{label}/{feat_name}: FNR overall={mf.overall["false_negative_rate"]:.3f}')
        return results
 
    def apply_threshold_correction(
        self,
        label: str,
        sensitive_col: str,
        target_fnr: float = FNR_TOLERANCE,
    ) -> dict:
        """
        Lower the classification threshold for subgroups where FNR > target_fnr.
 
        For each violating subgroup, performs a binary search to find the
        lowest threshold that achieves FNR <= target_fnr.
 
        Returns:
            dict mapping group_value -> corrected_threshold.
        """
        if sensitive_col not in self.df_meta.columns:
            log.warning(f'{sensitive_col} not found — cannot apply correction')
            return {}
 
        mask    = self.y_test[label].notna()
        y_true  = self.y_test.loc[mask, label].astype(int).values
        y_proba = self._get_proba(label)[mask]
        groups  = self.df_meta.loc[mask, sensitive_col].values
 
        corrected_thresholds = {}
 
        for group_val in np.unique(groups):
            g_mask   = groups == group_val
            g_true   = y_true[g_mask]
            g_proba  = y_proba[g_mask]
 
            # Current FNR at default threshold
            y_pred_default = (g_proba >= 0.5).astype(int)
            tp = ((y_pred_default == 1) & (g_true == 1)).sum()
            fn = ((y_pred_default == 0) & (g_true == 1)).sum()
            current_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
 
            if current_fnr <= target_fnr:
                corrected_thresholds[group_val] = 0.5  # no correction needed
                continue
 
            # Binary search for threshold that achieves FNR <= target_fnr
            lo, hi = 0.01, 0.49
            best_threshold = 0.5
            for _ in range(20):   # 20 iterations → precision ~0.0003
                mid = (lo + hi) / 2
                y_pred_mid = (g_proba >= mid).astype(int)
                tp_m = ((y_pred_mid==1)&(g_true==1)).sum()
                fn_m = ((y_pred_mid==0)&(g_true==1)).sum()
                fnr_m = fn_m / (fn_m + tp_m) if (fn_m + tp_m) > 0 else 0.0
                if fnr_m <= target_fnr:
                    best_threshold = mid
                    lo = mid   # can raise threshold
                else:
                    hi = mid   # must lower threshold
 
            corrected_thresholds[group_val] = round(best_threshold, 4)
            log.info(
                f'{label}/{sensitive_col}/{group_val}: '
                f'FNR {current_fnr:.3f} → threshold {best_threshold:.4f}'
            )
 
        clog.log(
            dataset='fairness_audit', step='threshold_correction',
            column_affected=f'{label}/{sensitive_col}',
            issue_found=f'FNR > {target_fnr} in some subgroups',
            action_taken=f'Binary search threshold correction: {corrected_thresholds}',
            rows_affected=int(mask.sum()), validation_result='PASS',
            analyst_notes=f'target_fnr={target_fnr}',
        )
        return corrected_thresholds
 
    def build_equity_report(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Build a DataFrame summarising FNR and recall per label and subgroup.
        One row per (label, sensitive_feature, group_value) combination.
        """
        rows = []
        for label in TARGET_COLS:
            audit_results = self.audit_all_groups(label, threshold)
            for feat_name, mf in audit_results.items():
                by_group = mf.by_group
                for group_val, group_metrics in by_group.iterrows():
                    rows.append({
                        'label':             label,
                        'sensitive_feature': feat_name,
                        'group_value':       str(group_val),
                        'fnr':               round(group_metrics['false_negative_rate'], 4),
                        'recall':            round(group_metrics['recall'], 4),
                        'fnr_exceeds_tolerance': group_metrics['false_negative_rate'] > FNR_TOLERANCE,
                    })
        df = pd.DataFrame(rows)
        log.info(f'Equity report: {len(df)} rows')
        return df
 
    def save_equity_report(self, df: pd.DataFrame) -> Path:
        """Save equity report to reports/tables/equity_audit.csv."""
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        path = TABLES_DIR / 'equity_audit.csv'
        df.to_csv(path, index=False)
        log.info(f'Equity report saved: {path}')
        return path
 
