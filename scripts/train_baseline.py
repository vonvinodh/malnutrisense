"""
scripts/train_baseline.py — Train and evaluate all baseline classifiers.
 
Trains 3 baselines × 3 phenotypes = 9 models.
Saves comparison metrics to reports/tables/baseline_benchmark.csv.
 
Usage: python3 scripts/train_baseline.py
"""
 
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import TRAIN_TEST_DIR, TARGET_COLS, validate_environment
from src.model import (
    build_logistic_baseline, build_rf_baseline, build_xgb_single,
    load_class_weights,
)
from src.evaluation import (
    evaluate_single_label, build_benchmark_table,
    save_benchmark, print_benchmark, log_to_mlflow,
)
from src.logger import get_console_logger
from src.utils import timer
 
log = get_console_logger(__name__)
 
 
def main() -> None:
    print('='*65)
    print('MalnutriSense — Baseline Training')
    print('='*65)
 
    validate_environment(require_data=False)
 
    # Load splits
    log.info('Loading train/test splits...')
    X_train = pd.read_csv(TRAIN_TEST_DIR / 'X_train.csv')
    X_test  = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
    y_train = pd.read_csv(TRAIN_TEST_DIR / 'y_train.csv')
    y_test  = pd.read_csv(TRAIN_TEST_DIR / 'y_test.csv')
    log.info(f'Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows')
 
    class_weights = load_class_weights()
 
    # Builders indexed by name
    BUILDERS = {
        'logistic': lambda label: build_logistic_baseline(X_train),
        'random_forest': lambda label: build_rf_baseline(X_train),
        'xgb_single': lambda label: build_xgb_single(
            X_train, scale_pos_weight=class_weights.get(label, 2.0)
        ),
    }
 
    all_results: dict = {}
 
    for model_name, builder in BUILDERS.items():
        print(f'\n--- {model_name} ---')
        label_results: dict = {}
 
        for label in TARGET_COLS:
            y_tr = y_train[label].dropna().astype(int)
            X_tr_aligned = X_train.loc[y_tr.index]
 
            with timer(f'{model_name} / {label}'):
                pipe = builder(label)
                pipe.fit(X_tr_aligned, y_tr)
 
            metrics = evaluate_single_label(
                pipe, X_test, y_test[label], label
            )
            label_results[label] = metrics
            print(f'  {label:<15} recall={metrics["recall"]:.3f}  '
                  f'f1={metrics["f1"]:.3f}  roc_auc={metrics["roc_auc"]:.3f}')
 
            log_to_mlflow(f'{model_name}_{label}', {label: metrics},
                          params={'model': model_name, 'label': label})
 
        all_results[model_name] = label_results
 
    # Build and save benchmark table
    benchmark_df = build_benchmark_table(all_results)
    path = save_benchmark(benchmark_df, 'baseline_benchmark')
    print_benchmark(benchmark_df)
 
    print('\n' + '='*65)
    print(f'Baseline benchmark saved: {path}')
    print('Run scripts/train_mltp.py to train the primary MLTP model.')
    print('='*65)
 
 
if __name__ == '__main__':
    main()
 
