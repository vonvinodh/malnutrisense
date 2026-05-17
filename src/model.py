"""
src/model.py — Classifier definitions for Objective 1 (Multi-Label MLTP).
 
Defines four classifiers:
  1. Logistic Regression baseline (single-label per phenotype)
  2. Random Forest baseline
  3. Single-label XGBoost baseline (trained 3×, once per label)
  4. MLTP — XGBoost in MultiOutputClassifier (predicts all 3 simultaneously)
 
Public API:
  build_logistic_baseline()   -> Pipeline
  build_rf_baseline()         -> Pipeline
  build_xgb_single(label)     -> Pipeline
  build_mltp(class_weights)   -> Pipeline
  tune_hyperparameters(pipe, X, y, param_grid) -> GridSearchCV
  save_model(model, path)     -> None
  load_model(path)            -> model
"""
 
import json
from pathlib import Path
from typing import Optional
 
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
 
from src.config import (
    RANDOM_STATE, CV_FOLDS, MODELS_DIR, TARGET_COLS, TABLES_DIR,
)
from src.logger import get_console_logger, CleaningLogger
 
log = get_console_logger(__name__)
clog = CleaningLogger()
 
 
# ── Shared preprocessing sub-pipeline ───────────────────────────────────
# Used by every classifier to ensure identical feature transformation.
# Numeric: median imputation (handles any residual NaN) + StandardScaler.
# Categorical (object dtype): most-frequent imputation + OrdinalEncoder.
def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer matching the dtypes in X."""
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    object_cols  = X.select_dtypes(include='object').columns.tolist()
 
    transformers = []
    if numeric_cols:
        transformers.append(('num',
            Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('scale',  StandardScaler())]),
            numeric_cols))
    if object_cols:
        transformers.append(('cat',
            Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                      ('encode', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                unknown_value=-1))]),
            object_cols))
 
    return ColumnTransformer(transformers=transformers, remainder='drop')
 
 
# ── Baseline 1: Logistic Regression ─────────────────────────────────────
def build_logistic_baseline(X: pd.DataFrame) -> Pipeline:
    """
    Single-label Logistic Regression baseline.
    Wrap in MultiOutputClassifier externally when predicting all 3 labels.
    """
    return Pipeline([
        ('pre', _make_preprocessor(X)),
        ('clf', LogisticRegression(
            max_iter=500,
            class_weight='balanced',   # handles class imbalance without manual weight
            random_state=RANDOM_STATE,
            solver='lbfgs',
            C=1.0,
        ))
    ])
 
 
# ── Baseline 2: Random Forest ────────────────────────────────────────────
def build_rf_baseline(X: pd.DataFrame) -> Pipeline:
    """Single-label Random Forest baseline."""
    return Pipeline([
        ('pre', _make_preprocessor(X)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced_subsample',
            max_depth=12,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
 
 
# ── Baseline 3: Single-label XGBoost ─────────────────────────────────────
def build_xgb_single(
    X: pd.DataFrame,
    scale_pos_weight: float = 2.0,
) -> Pipeline:
    """
    Single-label XGBoost baseline.
    Train separately for each phenotype — pass scale_pos_weight per label.
    """
    return Pipeline([
        ('pre', _make_preprocessor(X)),
        ('clf', XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            tree_method='hist',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ))
    ])
 
 
# ── Primary model: Multi-Label Trajectory Predictor (MLTP) ───────────────
def build_mltp(
    X: pd.DataFrame,
    class_weights: Optional[dict[str, float]] = None,
) -> Pipeline:
    """
    Multi-Label Trajectory Predictor.
    XGBoost wrapped in MultiOutputClassifier — predicts stunted, underweight,
    and wasted simultaneously. Each estimator gets its own scale_pos_weight.
 
    Args:
        X:             Feature matrix (used to infer dtypes for preprocessor).
        class_weights: Dict from reports/tables/class_weights.json.
                       Keys: 'stunted', 'underweight', 'wasted'.
                       If None, defaults to 2.0 for all labels.
    """
    cw = class_weights or {}
    # Use the mean class weight across all labels as a single value for
    # MultiOutputClassifier (it trains one XGBClassifier per label internally).
    # The actual per-label weight is applied by train_mltp() which trains each
    # label separately using the single-label path with the correct weight.
    avg_weight = float(np.mean(list(cw.values()))) if cw else 2.0
 
    base_xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=avg_weight,
        eval_metric='aucpr',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
 
    return Pipeline([
        ('pre', _make_preprocessor(X)),
        ('clf', MultiOutputClassifier(base_xgb, n_jobs=-1))
    ])
 
 
# ── LightGBM MLTP (comparison model) ────────────────────────────────────
def build_lgbm_mltp(
    X: pd.DataFrame,
    class_weights: Optional[dict[str, float]] = None,
) -> Pipeline:
    """LightGBM multi-label model for benchmarking against XGBoost MLTP."""
    avg_weight = float(np.mean(list(class_weights.values()))) if class_weights else 2.0
 
    base_lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        scale_pos_weight=avg_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
 
    return Pipeline([
        ('pre', _make_preprocessor(X)),
        ('clf', MultiOutputClassifier(base_lgbm, n_jobs=-1))
    ])
 
 
# ── Hyperparameter tuning ─────────────────────────────────────────────────
# Default grid for XGBoost single-label or MLTP base estimator.
XGB_PARAM_GRID = {
    'clf__estimator__n_estimators':  [200, 400],
    'clf__estimator__max_depth':     [4, 6],
    'clf__estimator__learning_rate': [0.05, 0.1],
    'clf__estimator__subsample':     [0.8, 1.0],
}
 
# Logistic Regression grid
LR_PARAM_GRID = {
    'clf__C': [0.01, 0.1, 1.0, 10.0],
    'clf__solver': ['lbfgs', 'liblinear'],
}
 
def tune_hyperparameters(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int = CV_FOLDS,
    scoring: str = 'recall',
) -> GridSearchCV:
    """
    Run GridSearchCV on a Pipeline.
 
    Args:
        pipeline:   Sklearn Pipeline with 'pre' and 'clf' steps.
        X_train:    Feature matrix.
        y_train:    Single label Series (for single-label models).
                    For MLTP use the wrapper in train_mltp() instead.
        param_grid: Dict of parameter names → list of values to try.
        cv:         Number of CV folds (default: CV_FOLDS from config).
        scoring:    Primary scoring metric. 'recall' prioritises finding
                    at-risk children (false negatives are costly in healthcare).
 
    Returns:
        Fitted GridSearchCV object.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                                   random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    log.info(f'Best params: {gs.best_params_}')
    log.info(f'Best {scoring}: {gs.best_score_:.4f}')
    return gs
 
 
# ── Model persistence ────────────────────────────────────────────────────
def save_model(model, name: str, subdir: str = '') -> Path:
    """
    Save a fitted model to models/ using joblib.
 
    Args:
        model:  Any sklearn-compatible fitted estimator.
        name:   Filename without extension (e.g. 'mltp_xgb_v1').
        subdir: Optional subdirectory inside models/.
 
    Returns:
        Path where the model was saved.
    """
    save_dir = MODELS_DIR / subdir if subdir else MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f'{name}.pkl'
    joblib.dump(model, path)
    size_mb = path.stat().st_size / 1e6
    log.info(f'Model saved: {path}  ({size_mb:.1f} MB)')
    clog.log(
        dataset='model_artefacts', step='save_model',
        column_affected='N/A',
        issue_found='N/A — model save step',
        action_taken=f'joblib.dump to {path}',
        rows_affected=-1, validation_result='INFO',
        analyst_notes=f'Size: {size_mb:.1f} MB',
    )
    return path
 
 
def load_model(path: Path):
    """Load a model saved by save_model()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}')
    model = joblib.load(path)
    log.info(f'Model loaded: {path}')
    return model
 
 
# ── Class weights loader ──────────────────────────────────────────────────
def load_class_weights() -> dict[str, float]:
    """
    Load class weights from reports/tables/class_weights.json.
    Created by notebooks/01_data_loading.ipynb (Step 16).
    """
    path = TABLES_DIR / 'class_weights.json'
    if not path.exists():
        log.warning(f'class_weights.json not found at {path}. Using default weight 2.0.')
        return {c: 2.0 for c in TARGET_COLS}
    with open(path) as f:
        weights = json.load(f)
    log.info(f'Class weights loaded: {weights}')
    return weights
 
