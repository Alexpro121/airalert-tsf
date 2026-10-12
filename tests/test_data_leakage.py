import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.feature_engineering import make_features


def _toy_grid(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Kyiv": rng.integers(0, 2, n),
            "Sumy": rng.integers(0, 2, n),
            "Chernihiv": rng.integers(0, 2, n),
        },
        index=idx,
    )


def test_no_future_in_features():
    grid = _toy_grid()
    X, y = make_features(grid, "Kyiv", ["Sumy", "Chernihiv"], horizon=3)
    for t in X.index:
        i = grid.index.get_loc(t)
        assert X.loc[t, "lag_1"] == grid["Kyiv"].iloc[i - 1]
        assert X.loc[t, "nbr_Sumy"] == grid["Sumy"].iloc[i - 1]
        assert X.loc[t, "roll_sum_6h"] == grid["Kyiv"].iloc[i - 6 : i].sum()
        assert y.loc[t] == grid["Kyiv"].iloc[i : i + 3].max()


def test_oof_output_covers_validation_blocks():
    from main import oof_predict

    idx = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.random(120)}, index=idx)
    y = pd.Series(rng.integers(0, 2, 120), index=idx)

    def factory(a, b):
        class M:
            def predict_proba(self, Xv):
                p = np.zeros(len(Xv))
                return np.vstack([1 - p, p]).T

        return M()


    oof = oof_predict(factory, lambda m, Xv: m.predict_proba(Xv)[:, 1],
                      X, y, horizon_h=3, n_splits=3)
    covered = set()
    for _, va_idx in TimeSeriesSplit(n_splits=3, gap=3).split(X):
        covered.update(va_idx)
    assert set(oof[oof.notna()].index) == set(X.index[list(covered)])


def test_country_ratio_excludes_target():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    grid = pd.DataFrame({"Kyiv": [1] * 50, "Sumy": [0] * 50}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=3)
    assert (X["country_ratio"] == 0).all()
    assert (X["lag_1"] == 1).all()


def test_oof_folds_keep_horizon_gap():
    from main import oof_predict

    idx = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.random(120)}, index=idx)
    y = pd.Series(rng.integers(0, 2, 120), index=idx)
    train_maxes = []

    def factory(a, b):
        train_maxes.append(a.index.max())

        class M:
            def predict_proba(self, Xv):
                p = np.zeros(len(Xv))
                return np.vstack([1 - p, p]).T

        return M()


    oof_predict(factory, lambda m, Xv: m.predict_proba(Xv)[:, 1],
                X, y, horizon_h=3, n_splits=3)
    expected = [(X.index[tr].max(), X.index[va].min())
                for tr, va in TimeSeriesSplit(n_splits=3, gap=3).split(X)]
    assert len(train_maxes) == 3
    for recorded, (exp_tr_max, exp_va_min) in zip(train_maxes, expected):
        assert recorded == exp_tr_max
        assert recorded + pd.Timedelta(hours=3) < exp_va_min


def test_single_class_metrics_return_nan():
    from src.evaluate import compute_metrics

    m = compute_metrics(np.zeros(10), np.linspace(0, 1, 10))
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])


def test_threshold_range_reaches_grid_top():
    from src.models import tune_threshold

    y = pd.Series([0] * 90 + [1] * 10)
    p = np.array([0.8] * 90 + [0.9] * 10)
    assert tune_threshold(y, p) == (0.9, False)


def test_operating_point_max_precision_at_recall():
    from src.models import select_operating_point

    y = pd.Series([0] * 70 + [1] * 30)
    p = np.array([0.1] * 70 + [0.5] * 20 + [0.9] * 10)
    thr, missed = select_operating_point(y, p, min_recall=0.85)
    assert missed is False
    pred = (p >= thr).astype(int)
    from sklearn.metrics import recall_score
    assert recall_score(y, pred) >= 0.85
    assert thr > 0.1


def test_operating_point_missed_flag():
    from src.models import select_operating_point

    y = pd.Series([0] * 90 + [1] * 10)
    p = np.array([0.0] * 100)
    thr, missed = select_operating_point(y, p, min_recall=0.85)
    assert missed is True
    assert thr == 0.9


def test_threshold_guardrail_blocks_panic():
    from src.models import tune_threshold

    y = pd.Series([0] * 70 + [1] * 30)
    p = np.array([0.11] * 70 + [0.11] * 20 + [0.9] * 10)
    assert tune_threshold(y, p, min_precision=0.0) == (0.11, False)
    assert tune_threshold(y, p, min_precision=0.4)[0] > 0.11


def test_onset_cont_split():
    from src.evaluate import compute_metrics

    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.3, 0.9, 0.4, 0.1])
    prev = np.array([0, 0, 1, 1, 0, 1])
    m = compute_metrics(y, p, 0.5, prev=prev)
    assert m["onset"]["n"] == 3 and m["cont"]["n"] == 3
    assert m["onset"]["pos_rate"] == round(2 / 3, 4)


def test_neighbors_exclude_target():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    grid = pd.DataFrame({"Sumy": [1] * 50, "Kyiv": [0] * 50}, index=idx)
    X, _ = make_features(grid, "Sumy", ["Sumy", "Kyiv"], horizon=3)
    assert "nbr_Sumy" not in X.columns
    assert "nbr_Kyiv" in X.columns
    assert (X["country_ratio"] == 0).all()


def test_manual_tiny_grid():
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    a = [0] * 30
    a[26] = 1
    grid = pd.DataFrame({"Kyiv": a, "Sumy": [0] * 30}, index=idx)
    X, y = make_features(grid, "Kyiv", ["Sumy"], horizon=3, lags=[1, 2], windows=[6])
    assert X.loc[idx[27], "lag_1"] == 1
    assert X.loc[idx[27], "lag_2"] == 0
    assert X.loc[idx[27], "roll_sum_6h"] == 1
    assert list(y.loc[[idx[24], idx[25], idx[26], idx[27]]]) == [1, 1, 1, 0]


def test_nan_coverage_drops_rows_and_flags():
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    a = [0] * 60
    a[30] = 1
    kv = [float(x) for x in a]
    kv[10:15] = [float("nan")] * 5
    grid = pd.DataFrame({"Kyiv": kv, "Sumy": [0] * 60}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=3, lags=[1, 2], windows=[6])
    assert X.loc[idx[16]:idx[20], "has_history"].eq(1).all()
    assert (X["hours_since_last"] <= 168.0).all()


def test_lags_require_lag_1():
    import pytest

    from src.config import Settings

    with pytest.raises(ValueError, match="lag_1"):
        Settings(lags=[2, 3])


def test_neighbor_onset_flag():
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    s = [0] * 60
    s[30] = 1
    grid = pd.DataFrame({"Kyiv": [0] * 60, "Sumy": s}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=3, lags=[1, 2], windows=[6])
    assert X.loc[idx[31], "nbr_new_onset"] == 1
    assert X.loc[idx[32], "nbr_new_onset"] == 0
    assert "war_week" not in X.columns


def test_tail_with_unknown_future_excluded():
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    grid = pd.DataFrame({"Kyiv": [0] * 30, "Sumy": [0] * 30}, index=idx)
    X, y = make_features(grid, "Kyiv", ["Sumy"], horizon=3)
    assert X.index.max() == idx[27]
    assert y.index.max() == idx[27]


def test_threat_path_distance_shrinks():
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    z = [0] * 60
    su, ch, ko = [0] * 60, [0] * 60, [0] * 60
    su[27], ch[28], ko[29] = 1, 1, 1
    grid = pd.DataFrame({"Kyiv": z, "Sumy": su, "Chernihiv": ch,
                         "Kyiv_Oblast": ko, "Poltava": z}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy", "Chernihiv", "Kyiv_Oblast", "Poltava"],
                         horizon=3, lags=[1, 2], windows=[6])
    assert X.loc[idx[28], "path_northeast_dist"] == 3
    assert X.loc[idx[29], "path_northeast_dist"] == 2
    assert X.loc[idx[30], "path_northeast_dist"] == 1
    assert X.loc[idx[30], "path_northeast_count"] == 1
    assert X.loc[idx[30], "min_threat_dist"] == 1


def test_forecast_moment_variant_a():
    """X[t] uses only A[<=t-1]; y[t] = max(A[t..t+2]); no dead hour."""
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    a = [0] * 60
    a[30] = 1
    grid = pd.DataFrame({"Kyiv": a, "Sumy": [0] * 60}, index=idx)
    X, y = make_features(grid, "Kyiv", ["Sumy"], horizon=3)
    assert X.loc[idx[31], "lag_1"] == 1
    assert X.loc[idx[30], "lag_1"] == 0
    assert y.loc[idx[28]] == 1 and y.loc[idx[30]] == 1
    assert y.loc[idx[31]] == 0


def test_hours_since_last_capped_before_first_alert():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    vals = [0] * 40 + [1] + [0] * 9
    grid = pd.DataFrame({"Kyiv": vals, "Sumy": [0] * 50}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=3)
    assert (X["hours_since_last"].loc[idx[24:40]] == 168.0).all()
    assert (X["has_history"].loc[idx[24:40]] == 0).all()
    assert X["hours_since_last"].loc[idx[41]] == 0.0
    assert X["has_history"].loc[idx[41]] == 1


def test_hours_since_last_clipped_after_long_calm():
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    vals = [1] + [0] * 299
    grid = pd.DataFrame({"Kyiv": vals, "Sumy": [0] * 300}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=3)
    assert X["hours_since_last"].max() == 168.0
    assert (X["has_history"] == 1).all()
