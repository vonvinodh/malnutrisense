"""
tests/test_model.py — Unit tests for src/model.py and src/evaluation.py.
 
All tests use synthetic in-memory datasets — no data files required.
Run: pytest tests/test_model.py -m unit -v
"""
 
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
 
from src.model import (
    build_logistic_baseline, build_rf_baseline,
    build_xgb_single, build_mltp, build_lgbm_mltp,
    save_model, load_model, XGB_PARAM_GRID,
)
from src.evaluation import (
    evaluate_single_label, evaluate_multilabel,
    build_benchmark_table, print_benchmark,
)
from src.config import MODELS_DIR, TARGET_COLS
 
 
# ── Shared fixtures ───────────────────────────────────────────────────────
@pytest.fixture
def synthetic_data():
    """100-row dataset with numeric and categorical features."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        'HAZ':   rng.normal(-1.8, 1.2, n),
        'WAZ':   rng.normal(-1.7, 1.1, n),
        'WHZ':   rng.normal(-0.5, 1.0, n),
        'HV270': rng.integers(1, 6, n).astype(float),
        'V025':  rng.choice(['urban', 'rural'], n),
        'V106':  rng.choice(['no_education','primary','secondary','higher'], n),
        'sdi':   rng.uniform(0, 1, n),
    })
    y_single = pd.Series((rng.random(n) < 0.35).astype(int), name='stunted')
    y_multi  = pd.DataFrame({
        'stunted':     (rng.random(n) < 0.35).astype(int),
        'underweight': (rng.random(n) < 0.32).astype(int),
        'wasted':      (rng.random(n) < 0.19).astype(int),
    })
    # Use 160/40 train/test split
    X_train, X_test   = X.iloc[:160], X.iloc[160:]
    y_single_tr, y_single_te = y_single.iloc[:160], y_single.iloc[160:]
    y_multi_tr, y_multi_te   = y_multi.iloc[:160], y_multi.iloc[160:]
    return X_train, X_test, y_single_tr, y_single_te, y_multi_tr, y_multi_te
 
 
# ── Tests: model build functions ─────────────────────────────────────────
class TestModelBuilders:
 
    @pytest.mark.unit
    def test_logistic_returns_pipeline(self, synthetic_data):
        X_train, *_ = synthetic_data
        pipe = build_logistic_baseline(X_train)
        assert isinstance(pipe, Pipeline)
        assert 'pre' in pipe.named_steps
        assert 'clf' in pipe.named_steps
 
    @pytest.mark.unit
    def test_rf_returns_pipeline(self, synthetic_data):
        X_train, *_ = synthetic_data
        pipe = build_rf_baseline(X_train)
        assert isinstance(pipe, Pipeline)
 
    @pytest.mark.unit
    def test_xgb_single_returns_pipeline(self, synthetic_data):
        X_train, *_ = synthetic_data
        pipe = build_xgb_single(X_train, scale_pos_weight=1.84)
        assert isinstance(pipe, Pipeline)
 
    @pytest.mark.unit
    def test_mltp_uses_multioutput_classifier(self, synthetic_data):
        X_train, *_ = synthetic_data
        pipe = build_mltp(X_train, {'stunted':1.84,'underweight':2.15,'wasted':4.24})
        assert isinstance(pipe, Pipeline)
        from sklearn.multioutput import MultiOutputClassifier
        assert isinstance(pipe.named_steps['clf'], MultiOutputClassifier)
 
    @pytest.mark.unit
    def test_lgbm_returns_pipeline(self, synthetic_data):
        X_train, *_ = synthetic_data
        pipe = build_lgbm_mltp(X_train)
        assert isinstance(pipe, Pipeline)
 
 
# ── Tests: model fit and predict ─────────────────────────────────────────
class TestModelFitPredict:
 
    @pytest.mark.unit
    def test_single_label_fit_predict(self, synthetic_data):
        X_train, X_test, y_single_tr, y_single_te, *_ = synthetic_data
        pipe = build_logistic_baseline(X_train)
        pipe.fit(X_train, y_single_tr)
        y_pred = pipe.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({0, 1})
 
    @pytest.mark.unit
    def test_single_label_predict_proba_shape(self, synthetic_data):
        X_train, X_test, y_single_tr, *_ = synthetic_data
        pipe = build_logistic_baseline(X_train)
        pipe.fit(X_train, y_single_tr)
        proba = pipe.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)  # (n_samples, n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
 
    @pytest.mark.unit
    def test_mltp_predict_shape(self, synthetic_data):
        X_train, X_test, _, _, y_multi_tr, _ = synthetic_data
        pipe = build_mltp(X_train)
        pipe.fit(X_train, y_multi_tr)
        y_pred = pipe.predict(X_test)
        # MultiOutputClassifier returns (n_samples, n_labels)
        assert y_pred.shape == (len(X_test), 3)
 
    @pytest.mark.unit
    def test_mltp_predict_proba_returns_list(self, synthetic_data):
        X_train, X_test, _, _, y_multi_tr, _ = synthetic_data
        pipe = build_mltp(X_train)
        pipe.fit(X_train, y_multi_tr)
        proba_list = pipe.predict_proba(X_test)
        # MultiOutputClassifier returns a list of arrays, one per label
        assert isinstance(proba_list, list)
        assert len(proba_list) == 3
        for p in proba_list:
            assert p.shape == (len(X_test), 2)
 
    @pytest.mark.unit
    def test_mltp_labels_are_binary(self, synthetic_data):
        X_train, X_test, _, _, y_multi_tr, _ = synthetic_data
        pipe = build_mltp(X_train)
        pipe.fit(X_train, y_multi_tr)
        y_pred = pipe.predict(X_test)
        assert set(y_pred.flatten()).issubset({0, 1})
 
 
# ── Tests: evaluation module ──────────────────────────────────────────────
class TestEvaluation:
 
    @pytest.fixture
    def fitted_logistic(self, synthetic_data):
        X_train, _, y_single_tr, *_ = synthetic_data
        pipe = build_logistic_baseline(X_train)
        pipe.fit(X_train, y_single_tr)
        return pipe
 
    @pytest.fixture
    def fitted_mltp(self, synthetic_data):
        X_train, _, _, _, y_multi_tr, _ = synthetic_data
        pipe = build_mltp(X_train)
        pipe.fit(X_train, y_multi_tr)
        return pipe
 
    @pytest.mark.unit
    def test_evaluate_single_label_keys(self, synthetic_data, fitted_logistic):
        _, X_test, _, y_single_te, *_ = synthetic_data
        m = evaluate_single_label(fitted_logistic, X_test, y_single_te, 'stunted')
        assert all(k in m for k in ['label','recall','precision','f1','roc_auc','n_test'])
 
    @pytest.mark.unit
    def test_recall_in_valid_range(self, synthetic_data, fitted_logistic):
        _, X_test, _, y_single_te, *_ = synthetic_data
        m = evaluate_single_label(fitted_logistic, X_test, y_single_te, 'stunted')
        assert 0.0 <= m['recall'] <= 1.0
        assert 0.0 <= m['roc_auc'] <= 1.0
        assert 0.0 <= m['f1'] <= 1.0
 
    @pytest.mark.unit
    def test_evaluate_multilabel_returns_all_labels(self, synthetic_data, fitted_mltp):
        _, X_test, _, _, _, y_multi_te = synthetic_data
        results = evaluate_multilabel(fitted_mltp, X_test, y_multi_te)
        assert 'stunted' in results
        assert 'underweight' in results
        assert 'wasted' in results
        assert 'macro_avg' in results
 
    @pytest.mark.unit
    def test_macro_avg_is_mean_of_per_label(self, synthetic_data, fitted_mltp):
        _, X_test, _, _, _, y_multi_te = synthetic_data
        results = evaluate_multilabel(fitted_mltp, X_test, y_multi_te)
        per_label_recalls = [results[l]['recall'] for l in TARGET_COLS if l in results]
        expected_macro = round(float(np.mean(per_label_recalls)), 4)
        assert abs(results['macro_avg']['recall'] - expected_macro) < 0.001
 
    @pytest.mark.unit
    def test_build_benchmark_table_shape(self, synthetic_data, fitted_mltp, fitted_logistic):
        _, X_test, _, y_single_te, _, y_multi_te = synthetic_data
        m_lr = {'stunted': evaluate_single_label(fitted_logistic, X_test, y_single_te, 'stunted')}
        m_ml = evaluate_multilabel(fitted_mltp, X_test, y_multi_te)
        df = build_benchmark_table({'logistic': m_lr, 'mltp_xgb': m_ml})
        assert 'model' in df.columns
        assert 'recall' in df.columns
        # logistic has 1 label, mltp has 3 labels (excluding macro_avg) = 4 rows
        assert len(df) == 4
 
 
# ── Tests: model persistence ─────────────────────────────────────────────
class TestModelPersistence:
 
    @pytest.mark.unit
    def test_save_and_load_roundtrip(self, synthetic_data, tmp_path, monkeypatch):
        X_train, X_test, y_single_tr, y_single_te, *_ = synthetic_data
        pipe = build_logistic_baseline(X_train)
        pipe.fit(X_train, y_single_tr)
 
        # Monkeypatch MODELS_DIR to use tmp_path so we don't pollute models/
        monkeypatch.setattr('src.model.MODELS_DIR', tmp_path)
        saved_path = save_model(pipe, 'test_model')
        assert saved_path.exists()
 
        loaded = load_model(saved_path)
        y_orig   = pipe.predict(X_test)
        y_loaded = loaded.predict(X_test)
        assert np.array_equal(y_orig, y_loaded)
 
    @pytest.mark.unit
    def test_load_model_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / 'does_not_exist.pkl')
 
