

"""
src/explainability.py — SHAP-based model explainability for Objective 3.
 
Computes exact Shapley values using TreeExplainer for the MLTP model.
Generates three plot types:
  - Summary plot: global feature importance ranked by mean |SHAP|
  - Waterfall plot: individual prediction explanation for a single child
  - Dependence plot: how one feature affects the prediction
 
Public API:
  SHAPExplainer(model, X_background)   -- class
  .compute_shap_values(X)              -> list[np.ndarray]
  .plot_summary(shap_values, X, label) -> Path
  .plot_waterfall(shap_values, X, i, label) -> Path
  .plot_dependence(shap_values, X, feature, label) -> Path
  .get_top_features(shap_values, X, label, n)  -> list[str]
"""
 
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # headless backend for Codespace / server use
 
import shap
 
from src.config import FIGURES_DIR, TARGET_COLS
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
SHAP_DIR = FIGURES_DIR / 'shap'
SHAP_DIR.mkdir(parents=True, exist_ok=True)
 
 
class SHAPExplainer:
    """
    Wrapper around shap.TreeExplainer for the MLTP XGBoost model.
 
    Usage:
        explainer = SHAPExplainer(mltp_model, X_train)
        shap_values = explainer.compute_shap_values(X_test)
        explainer.plot_summary(shap_values, X_test, 'stunted')
    """
 
    def __init__(
        self,
        model,
        X_background: pd.DataFrame,
        label_index_map: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Args:
            model:           Fitted Pipeline with MultiOutputClassifier at clf step.
            X_background:    DataFrame used to build the SHAP background distribution.
                             Use a sample of X_train (100–500 rows is enough).
            label_index_map: Maps label name -> index in TARGET_COLS.
                             Default: {'stunted':0,'underweight':1,'wasted':2}
        """
        self.model    = model
        self.lim      = label_index_map or {l: i for i, l in enumerate(TARGET_COLS)}
 
        # Extract preprocessor output to pass background to TreeExplainer
        # MultiOutputClassifier stores estimators_[i] per label
        pre = model.named_steps['pre']
        X_bg_pre = pd.DataFrame(
            pre.transform(X_background),
            columns=self._get_feature_names(pre, X_background),
        )
 
        # Build one explainer per label estimator
        clf = model.named_steps['clf']
        self.explainers = [
            shap.TreeExplainer(estimator, X_bg_pre)
            for estimator in clf.estimators_
        ]
        self.feature_names: list[str] = list(X_bg_pre.columns)
        log.info(f'SHAPExplainer initialised for {len(self.explainers)} labels')
 
    def _get_feature_names(self, preprocessor, X: pd.DataFrame) -> list[str]:
        """Extract feature names after ColumnTransformer."""
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            return [f'f{i}' for i in range(preprocessor.transform(X.iloc[:1]).shape[1])]
 
    def _transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessor and return a named DataFrame."""
        pre = self.model.named_steps['pre']
        return pd.DataFrame(
            pre.transform(X),
            columns=self.feature_names,
            index=X.index,
        )
 
    def compute_shap_values(
        self,
        X: pd.DataFrame,
    ) -> list[np.ndarray]:
        """
        Compute SHAP values for all rows in X.
 
        Returns:
            List of arrays, one per label.
            Each array shape: (n_samples, n_features).
        """
        X_pre = self._transform_X(X)
        values = [
            exp.shap_values(X_pre, check_additivity=False)
            for exp in self.explainers
        ]
        log.info(f'SHAP values computed: {X_pre.shape[0]} rows, {X_pre.shape[1]} features')
        return values
 
    def plot_summary(
        self,
        shap_values: list[np.ndarray],
        X: pd.DataFrame,
        label: str,
        max_features: int = 15,
    ) -> Path:
        """
        Global feature importance bar plot ranked by mean |SHAP| value.
        Saved to reports/figures/shap/summary_{label}.png.
        """
        idx = self.lim[label]
        X_pre = self._transform_X(X)
 
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values[idx],
            X_pre,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_features,
            show=False,
            ax=ax,
        )
        ax.set_title(f'SHAP Feature Importance — {label.capitalize()}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = SHAP_DIR / f'summary_{label}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f'Summary plot saved: {path}')
        return path
 
    def plot_waterfall(
        self,
        shap_values: list[np.ndarray],
        X: pd.DataFrame,
        row_index: int,
        label: str,
        max_features: int = 10,
    ) -> Path:
        """
        Individual child explanation — shows which features pushed
        the prediction above or below the base rate.
 
        Args:
            row_index: Row position in X (0-indexed).
        """
        idx   = self.lim[label]
        X_pre = self._transform_X(X)
        base  = self.explainers[idx].expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(base[1])   # positive class base value
 
        sv_row = shap_values[idx][row_index]
        exp_obj = shap.Explanation(
            values=sv_row,
            base_values=base,
            data=X_pre.iloc[row_index].values,
            feature_names=self.feature_names,
        )
 
        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(exp_obj, max_display=max_features, show=False)
        plt.title(f'SHAP Waterfall — {label.capitalize()} (row {row_index})',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = SHAP_DIR / f'waterfall_{label}_row{row_index}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f'Waterfall plot saved: {path}')
        return path
 
    def plot_dependence(
        self,
        shap_values: list[np.ndarray],
        X: pd.DataFrame,
        feature: str,
        label: str,
    ) -> Path:
        """Dependence plot: how feature value affects SHAP contribution."""
        idx   = self.lim[label]
        X_pre = self._transform_X(X)
 
        if feature not in self.feature_names:
            log.warning(f'{feature} not in feature_names after preprocessing')
            return None
 
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feature, shap_values[idx], X_pre,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title(f'SHAP Dependence: {feature} → {label.capitalize()}',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = SHAP_DIR / f'dependence_{label}_{feature}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f'Dependence plot saved: {path}')
        return path
 
    def get_top_features(
        self,
        shap_values: list[np.ndarray],
        label: str,
        n: int = 10,
    ) -> list[str]:
        """Return top-N features by mean absolute SHAP value for a label."""
        idx       = self.lim[label]
        mean_abs  = np.abs(shap_values[idx]).mean(axis=0)
        ranked    = sorted(zip(self.feature_names, mean_abs),
                           key=lambda x: x[1], reverse=True)
        top = [name for name, _ in ranked[:n]]
        log.info(f'Top {n} features for {label}: {top}')
        return top
 
