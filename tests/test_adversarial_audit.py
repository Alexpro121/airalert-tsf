"""Behavioral regression guards for past audit findings."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import generate_synthetic
from src.evaluate import safe_pr_auc, safe_roc_auc
from src.explain import save_importance
from src.feature_engineering import make_features
from src.models import tune_threshold
from src.text_signals import ALL_CLEAR_KEYWORDS, build_hourly_flags


def _idx(n=50):
    return pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")


def test_01_all_clear_not_threat():
    idx = _idx()
    ev = pd.DataFrame([{"timestamp": idx[30], "text": "Відбій загрози"}])
    assert build_hourly_flags(ev, idx).loc[idx[30]] == 0
    assert build_hourly_flags(ev, idx, keywords=ALL_CLEAR_KEYWORDS).loc[idx[30]] == 1


def test_02_target_excluded_from_neighbors():
    # Canonical coverage lives in test_data_leakage.test_neighbors_exclude_target;
    # here only the Sumy-specific regression (target inside default neighbors).
    idx = _idx()
    grid = pd.DataFrame({"Sumy": [1] * 50, "Kyiv": [0] * 50}, index=idx)
    X, _ = make_features(grid, "Sumy", ["Sumy", "Kyiv"], horizon=3)
    assert "nbr_Sumy" not in X.columns


def test_03_single_class_auc_safe():
    # Canonical NaN-behavior lives in test_data_leakage.test_single_class_metrics_return_nan;
    # here only the safe_* helper contract.
    y0 = np.zeros(20)
    p = np.linspace(0, 1, 20)
    assert np.isnan(safe_pr_auc(y0, p)) and np.isnan(safe_roc_auc(y0, p))


def test_04_oof_embargo_gap_behavioral():
    # Canonical embargo proof lives in
    # test_data_leakage.test_oof_folds_keep_horizon_gap; here only that the
    # helper is importable and covers validation blocks.
    from main import oof_predict

    idx = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.random(120)}, index=idx)
    y = pd.Series(rng.integers(0, 2, 120), index=idx)

    def factory(a, b):
        class M:
            def predict_proba(self, Xv):
                z = np.zeros(len(Xv))
                return np.vstack([1 - z, z]).T

        return M()

    oof = oof_predict(factory, lambda m, Xv: m.predict_proba(Xv)[:, 1],
                      X, y, horizon_h=3, n_splits=3)
    assert oof.notna().sum() > 0


def test_05_hgb_factory_disables_early_stopping():
    from main import train_model
    from src.config import Settings

    idx = _idx(200)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, 3)), columns=list("abc"), index=idx)
    y = pd.Series(rng.integers(0, 2, 200), index=idx)
    m = train_model(X, y, Settings(), backend="hgb")
    assert m.get_params()["early_stopping"] is False
    assert m.get_params()["class_weight"] is None


def test_05b_oserror_on_import_falls_back(monkeypatch):
    import main as main_mod

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if isinstance(name, str) and name.split(".")[0] == "lightgbm":
            raise OSError("libomp missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert main_mod.detect_backend() == "hgb"


def test_06_threshold_grid_top_reachable():
    # Canonical grid-top proof lives in
    # test_data_leakage.test_threshold_range_reaches_grid_top; here only that
    # tune_threshold exposes the used_fallback flag.
    y = pd.Series([0] * 90 + [1] * 10)
    p = np.array([0.8] * 90 + [0.9] * 10)
    thr, fb = tune_threshold(y, p)
    assert thr == 0.9 and fb is False


def test_07_synthetic_deterministic(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    generate_synthetic(a, days=60, seed=42)
    generate_synthetic(b, days=60, seed=42)
    dfa = pd.read_csv(a, parse_dates=["start_time", "end_time"])
    dfb = pd.read_csv(b, parse_dates=["start_time", "end_time"])
    pd.testing.assert_frame_equal(dfa, dfb)


def test_08_gain_fallback_writes_plot(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier

    idx = _idx(300)
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((300, 4)), columns=list("abcd"), index=idx)
    y = pd.Series(rng.integers(0, 2, 300), index=idx)
    m = HistGradientBoostingClassifier(max_iter=5).fit(X, y)
    out = str(tmp_path / "imp.png")
    save_importance(m, X, y, out)
    assert Path(str(tmp_path / "imp_fallback.png")).exists()
    assert not Path(out).exists()


def test_08b_waterfall_fn_branch(tmp_path):
    from lightgbm import LGBMClassifier

    from src.explain import top_episode_waterfall

    idx = _idx(120)
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((120, 3)), columns=list("abc"), index=idx)
    y = pd.Series([0] * 100 + [1] * 20, index=idx)
    m = LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
    outs = top_episode_waterfall(X, y, m, str(tmp_path / "shap_summary.png"), threshold=0.99)
    assert any("fn_" in o for o in outs)
    for o in outs:
        assert Path(o).exists()


def test_09_coverage_guard_nonempty():
    idx = _idx()
    from src.grid_builder import build_hourly_grid

    raw = pd.DataFrame([{
        "region": "Kyiv",
        "start_time": idx[10] + pd.Timedelta(minutes=5),
        "end_time": idx[10] + pd.Timedelta(minutes=25),
        "alert_type": "air_raid",
    }])
    g = build_hourly_grid(raw, ["Kyiv"])
    slot = [h for h in g.index if h < raw["end_time"][0] and h + pd.Timedelta(hours=1) > raw["start_time"][0]]
    assert len(slot) > 0
    assert (g.loc[slot, "Kyiv"] == 1).all()
