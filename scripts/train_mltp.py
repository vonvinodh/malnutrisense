"""
scripts/train_mltp.py — Train the Multi-Label Trajectory Predictor (MLTP).
 
Trains XGBoost via MultiOutputClassifier on all 3 phenotypes simultaneously.
Saves fitted model to models/mltp_xgb_v1.pkl.
Prints benchmark comparing MLTP vs all baselines.
 
Usage: python3 scripts/train_mltp.py
"""
 
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import TRAIN_TEST_DIR, TARGET_COLS, TABLES_DIR, validate_environment
from src.model import build_mltp, build_lgbm_mltp, save_model, load_class_weights
from src.evaluation import (
    evaluate_multilabel, build_benchmark_table,
    save_benchmark, print_benchmark, log_to_mlflow,
)
from src.logger import get_console_logger
from src.utils import timer
 
log = get_console_logger(__name__)
 
 
def main() -> None:
    print('='*65)
    print('MalnutriSense — MLTP Training')
    print('='*65)
 
    validate_environment(require_data=False)
 
    # Load splits
    X_train = pd.read_csv(TRAIN_TEST_DIR / 'X_train.csv')
    X_test  = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
    y_train = pd.read_csv(TRAIN_TEST_DIR / 'y_train.csv')
    y_test  = pd.read_csv(TRAIN_TEST_DIR / 'y_test.csv')
    log.info(f'Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows')
 
    class_weights = load_class_weights()
    log.info(f'Class weights: {class_weights}')
 
    all_results: dict = {}
 
    # ── Train MLTP (XGBoost) ──────────────────────────────────────────
    print('\n[1/2] Training XGBoost MLTP...')
    with timer('XGBoost MLTP fit'):
        mltp_xgb = build_mltp(X_train, class_weights)
        mltp_xgb.fit(X_train, y_train[TARGET_COLS].fillna(0).astype(int))
 
    xgb_metrics = evaluate_multilabel(mltp_xgb, X_test, y_test[TARGET_COLS])
    all_results['mltp_xgb'] = xgb_metrics
 
    print('  XGBoost MLTP results:')
    for label in TARGET_COLS:
        m = xgb_metrics.get(label, {})
        print(f'  {label:<15} recall={m.get("recall",0):.3f}  '
              f'f1={m.get("f1",0):.3f}  roc_auc={m.get("roc_auc",0):.3f}')
    print(f'  macro_avg recall={xgb_metrics["macro_avg"]["recall"]:.3f}')
 
    # Save the XGBoost MLTP
    model_path = save_model(mltp_xgb, 'mltp_xgb_v1')
    print(f'  Model saved: {model_path}')
 
    log_to_mlflow('mltp_xgb', xgb_metrics,
        params={'model_type': 'MultiOutputClassifier(XGBoost)',
                'class_weights': str(class_weights)})
 
    # ── Train MLTP (LightGBM) — comparison ───────────────────────────
    print('\n[2/2] Training LightGBM MLTP (comparison)...')
    with timer('LightGBM MLTP fit'):
        mltp_lgbm = build_lgbm_mltp(X_train, class_weights)
        mltp_lgbm.fit(X_train, y_train[TARGET_COLS].fillna(0).astype(int))
 
    lgbm_metrics = evaluate_multilabel(mltp_lgbm, X_test, y_test[TARGET_COLS])
    all_results['mltp_lgbm'] = lgbm_metrics
    save_model(mltp_lgbm, 'mltp_lgbm_v1')
 
    # ── Load baseline results if available ────────────────────────────
    baseline_path = TABLES_DIR / 'baseline_benchmark.csv'
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        # Reconstruct baseline results dict from CSV for benchmark table
        for model_name in baseline_df['model'].unique():
            model_rows = baseline_df[baseline_df['model']==model_name]
            all_results[model_name] = {
                row['label']: {k: row[k] for k in ['recall','precision','f1','roc_auc','n_test']}
                for _, row in model_rows.iterrows()
            }
 
    # ── Full benchmark: MLTP vs all baselines ────────────────────────
    full_bench = build_benchmark_table(all_results)
    bench_path = save_benchmark(full_bench, 'full_benchmark')
    print_benchmark(full_bench)
 
    # ── Validation: MLTP recall must beat logistic baseline ──────────
    mltp_recall = xgb_metrics['macro_avg']['recall']
    if baseline_path.exists():
        lr_rows = baseline_df[baseline_df['model']=='logistic']
        lr_recall = lr_rows['recall'].mean()
        if mltp_recall > lr_recall:
            print(f'\n[PASS] MLTP macro recall {mltp_recall:.3f} > logistic {lr_recall:.3f}')
        else:
            print(f'\n[WARN] MLTP macro recall {mltp_recall:.3f} <= logistic {lr_recall:.3f}')
            print('       Consider tuning n_estimators or learning_rate in build_mltp()')
 
    print('\n' + '='*65)
    print(f'MLTP model: models/mltp_xgb_v1.pkl')
    print(f'Full benchmark: {bench_path}')
    print('Proceed to notebooks/02_modeling.ipynb for interactive analysis.')
    print('='*65)
 
 
if __name__ == '__main__':
    main()
 
