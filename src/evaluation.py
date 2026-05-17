

"""
src/evaluation.py — Model evaluation, benchmark comparison, and MLflow logging.
 
Computes per-label metrics for multi-label classifiers and single-label baselines.
Primary metric: Recall (sensitivity) — minimises missed at-risk children.
Secondary:      F1-score, ROC-AUC, Precision.
 
Public API:
  evaluate_single_label(model, X_test, y_test, label)  -> dict
  evaluate_multilabel(model, X_test, y_test)           -> dict
  build_benchmark_table(results)                        -> DataFrame
  save_benchmark(df, name)                              -> Path
  log_to_mlflow(model_name, metrics, params)            -> None
"""
 
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score,
    precision_score, classification_report,
    confusion_matrix,
)
 
from src.config import TARGET_COLS, TABLES_DIR, MODELS_DIR
from src.logger import get_console_logger, CleaningLogger
from src.utils import format_number
 
log  = get_console_logger(__name__)
clog = CleaningLogger()
 
# Try to import MLflow — optional dependency (not installed in minimal env)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning('MLflow not available — experiment tracking disabled')
 
 
def evaluate_single_label(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a single-label binary classifier.
 
    Args:
        model:     Fitted sklearn Pipeline.
        X_test:    Test feature matrix.
        y_test:    True binary labels for this phenotype.
        label:     Name of the phenotype (e.g. 'stunted').
        threshold: Decision threshold for predict_proba (default 0.5).
 
    Returns:
        dict with keys: label, recall, f1, roc_auc, precision, n_test.
    """
    # Filter out rows where the label is NaN
    mask = y_test.notna()
    X_eval = X_test[mask]
    y_eval = y_test[mask].astype(int)
 
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
 
    metrics = {
        'label':     label,
        'recall':    round(recall_score(y_eval, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y_eval, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_eval, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_eval, y_proba), 4),
        'threshold': threshold,
        'n_test':    int(mask.sum()),
    }
    log.info(
        f'{label}: recall={metrics["recall"]:.3f} '
        f'f1={metrics["f1"]:.3f} roc_auc={metrics["roc_auc"]:.3f}'
    )
    return metrics
 
 
def evaluate_multilabel(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a MultiOutputClassifier on all three labels simultaneously.
 
    Args:
        model:     Fitted Pipeline with MultiOutputClassifier at clf step.
        X_test:    Test feature matrix.
        y_test:    DataFrame with stunted, underweight, wasted columns.
        threshold: Decision threshold applied to all three labels.
 
    Returns:
        dict with per-label metrics and macro-averaged summary.
    """
    y_test = y_test.copy()
 
    # Collect per-label probabilities from MultiOutputClassifier
    # .predict_proba() returns a list of arrays, one per label
    proba_list = model.predict_proba(X_test)
 
    results = {}
    recalls, f1s, aucs = [], [], []
 
    for i, label in enumerate(TARGET_COLS):
        if label not in y_test.columns:
            log.warning(f'{label} not in y_test — skipping')
            continue
 
        y_true  = y_test[label].dropna().astype(int)
        mask    = y_test[label].notna()
        y_proba = proba_list[i][mask, 1]
        y_pred  = (y_proba >= threshold).astype(int)
 
        rec = recall_score(y_true, y_pred, zero_division=0)
        pre = precision_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba)
 
        results[label] = {
            'recall':    round(rec, 4),
            'precision': round(pre, 4),
            'f1':        round(f1, 4),
            'roc_auc':   round(auc, 4),
            'n_test':    int(mask.sum()),
        }
        recalls.append(rec); f1s.append(f1); aucs.append(auc)
        log.info(f'{label}: recall={rec:.3f} f1={f1:.3f} roc_auc={auc:.3f}')
 
    # Macro average across all three labels
    results['macro_avg'] = {
        'recall':  round(float(np.mean(recalls)), 4),
        'f1':      round(float(np.mean(f1s)), 4),
        'roc_auc': round(float(np.mean(aucs)), 4),
    }
    log.info(f'Macro avg: recall={results["macro_avg"]["recall"]:.3f}')
    return results
 
 
def build_benchmark_table(results: dict) -> pd.DataFrame:
    """
    Build a tidy benchmark comparison DataFrame from a dict of model results.
 
    Args:
        results: Dict mapping model_name -> per-label metrics dict.
                 e.g. {'logistic': {'stunted': {...}, 'underweight': {...}},
                        'mltp_xgb': {'stunted': {...}, ...}}
 
    Returns:
        DataFrame with columns: model, label, recall, f1, roc_auc, precision.
    """
    rows = []
    for model_name, label_metrics in results.items():
        for label, metrics in label_metrics.items():
            if label == 'macro_avg':
                continue
            rows.append({
                'model':     model_name,
                'label':     label,
                'recall':    metrics.get('recall', None),
                'precision': metrics.get('precision', None),
                'f1':        metrics.get('f1', None),
                'roc_auc':   metrics.get('roc_auc', None),
                'n_test':    metrics.get('n_test', None),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(['label', 'recall'], ascending=[True, False])
    return df.reset_index(drop=True)
 
 
def save_benchmark(df: pd.DataFrame, name: str = 'benchmark') -> Path:
    """Save benchmark DataFrame to reports/tables/ as CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f'{name}.csv'
    df.to_csv(path, index=False)
    log.info(f'Benchmark saved: {path}')
    clog.log(
        dataset='evaluation', step='save_benchmark',
        column_affected='all',
        issue_found='N/A — benchmark save step',
        action_taken=f'Saved {len(df)} rows to {path}',
        rows_affected=len(df), validation_result='INFO',
        analyst_notes=name,
    )
    return path
 
 
def log_to_mlflow(
    model_name: str,
    metrics: dict,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
) -> None:
    """
    Log metrics and parameters to an MLflow run.
    No-op if MLflow is not installed.
    """
    if not MLFLOW_AVAILABLE:
        return
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        # Flatten nested per-label metrics
        for label, label_metrics in metrics.items():
            if isinstance(label_metrics, dict):
                for metric_name, value in label_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f'{label}_{metric_name}', value)
    log.info(f'MLflow: logged run for {model_name}')
 
 
def print_benchmark(df: pd.DataFrame) -> None:
    """Print a formatted benchmark table to stdout."""
    print('\nBenchmark Comparison (sorted by Recall per label):')
    print('='*75)
    print(f'{"Model":<22} {"Label":<14} {"Recall":>8} {"F1":>8} {"ROC-AUC":>9} {"Precision":>10}')
    print('-'*75)
    for _, row in df.iterrows():
        print(f'{row["model"]:<22} {row["label"]:<14} '
              f'{row["recall"]:>8.3f} {row["f1"]:>8.3f} '
              f'{row["roc_auc"]:>9.3f} {row["precision"]:>10.3f}')
    print('='*75)
 
