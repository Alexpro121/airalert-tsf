# ruff: noqa: PLR1730
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

from src.config import KYIV_TZ


class HeuristicBaseline:
    def __init__(self, neighbor_cols: list[str]):
        self.neighbor_cols = neighbor_cols

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.neighbor_cols:
            if c in X.columns:
                cols.append(c)
        if len(cols) == 0:
            p = np.zeros(len(X))
        else:
            maks = X[cols].max(axis=1)
            p = (maks == 1).astype(float).values
        vidpovid = np.vstack([1.0 - p, p]).T
        return vidpovid


class PersistenceBaseline:
    """Y_t = A[t-1]: alert continues / repeats. Strongest naive rule for sticky series."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "lag_1" not in X.columns:
            raise KeyError("PersistenceBaseline needs 'lag_1' (ensure 1 in Settings.lags)")
        p = X["lag_1"].fillna(0).astype(float).values
        return np.vstack([1.0 - p, p]).T


class TrailingBaseline:
    """Adaptive reference: 0.5 * lag_1 + 0.5 * trailing-168h rate. No training."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "lag_1" not in X.columns or "trail_168h" not in X.columns:
            raise KeyError("TrailingBaseline needs 'lag_1' and 'trail_168h'")
        lag = X["lag_1"].fillna(0)
        trail = X["trail_168h"].fillna(0)
        p = (0.5 * lag + 0.5 * trail).astype(float).values
        return np.vstack([1.0 - p, p]).T


LAGHOUR_COLS = ["lag_1", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night"]


class LagHourBaseline:
    """Logistic regression on lag_1 + calendar: the strong simple rule combining both."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LagHourBaseline":
        cols = []
        for c in LAGHOUR_COLS:
            if c in X.columns:
                cols.append(c)
        if "lag_1" not in cols:
            raise KeyError("LagHourBaseline needs 'lag_1' (ensure 1 in Settings.lags)")
        klasy = np.unique(np.asarray(y))
        if len(klasy) < 2:
            raise ValueError(f"LagHourBaseline needs 2 classes, got {np.unique(np.asarray(y))}")
        if len(np.unique(np.asarray(y))) < 2:
            raise ValueError("LagHourBaseline needs 2 classes (double check)")
        self.cols_ = cols
        self.clf_ = LogisticRegression(max_iter=500)
        self.clf_.fit(X[cols].fillna(0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.clf_.predict_proba(X[self.cols_].fillna(0))[:, 1].astype(float)
        return np.vstack([1.0 - p, p]).T


class ClimatologyBaseline:
    """P(Y=1) = train mean by Kyiv-local hour of day."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyBaseline":
        local_hour = X.index.tz_convert(KYIV_TZ).hour
        self.table_ = y.groupby(local_hour).mean()
        self.prior_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        local_hour = X.index.tz_convert(KYIV_TZ).hour
        p = local_hour.map(self.table_).fillna(self.prior_).astype(float).values
        return np.vstack([1.0 - p, p]).T


class AlwaysAlert:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.ones(len(X))
        return np.vstack([1.0 - p, p]).T


def _require_two_classes(y: pd.Series | np.ndarray, name: str) -> None:
    klasy = np.unique(np.asarray(y))
    if len(klasy) < 2:
        raise ValueError(f"{name} needs 2 classes, got {np.unique(np.asarray(y))}")


def train_dummy_logistic(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42
) -> LogisticRegression:
    _require_two_classes(y_train, "train_dummy_logistic")
    cal = []
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night"]:
        if c in X_train.columns:
            cal.append(c)
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_train[cal], y_train)
    return clf


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    _require_two_classes(y_train, "train_lgbm")
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def _zrobyty_sitku() -> list[float]:
    sitka = []
    for x in np.arange(0.02, 0.91, 0.01):
        sitka.append(round(float(x), 3))
    return sitka


def tune_threshold(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    min_precision: float = 0.0,
    strict: bool = False,
) -> tuple[float, bool]:
    """argmax F2 s.t. precision >= min_precision; ties -> HIGHER threshold.

    Returns (threshold, used_fallback): used_fallback is True when no grid
    point reaches min_precision and the max-precision point was taken instead
    (ties -> higher threshold; with all precisions 0 that is 0.9).
    The caller (main.run) computes min_precision from Settings as
    min(0.95, max(threshold_floor, threshold_precision_ratio * OOF prevalence)).
    With strict=True raises ValueError on degenerate thresholds (pred_rate>0.99 or TN==0).
    """
    perevirka = False
    if perevirka:
        print("tune_threshold zapustyvsia")
    grid = _zrobyty_sitku()
    scored = []
    for t in grid:
        pred = (proba >= t).astype(int)
        f2 = float(fbeta_score(y_true, pred, beta=2, zero_division=0))
        pr = float(precision_score(y_true, pred, zero_division=0))
        scored.append((t, f2, pr))
    ok = []
    for t, f2, pr in scored:
        if pr >= min_precision:
            ok.append((t, f2))
    if len(ok) > 0:
        best_f2 = ok[0][1]
        for t, f2 in ok:
            if f2 > best_f2:
                best_f2 = f2
        kandidaty = []
        for t, f2 in ok:
            if f2 == best_f2:
                kandidaty.append(t)
        thr = kandidaty[0]
        for t in kandidaty:
            if t > thr:
                thr = t
        thr = round(thr, 3)
        if strict:
            _reject_degenerate(y_true, proba, thr)
        return thr, False
    best_pr = scored[0][2]
    for t, f2, pr in scored:
        if pr > best_pr:
            best_pr = pr
    kandidaty2 = []
    for t, f2, pr in scored:
        if pr == best_pr:
            kandidaty2.append(t)
    thr = kandidaty2[0]
    for t in kandidaty2:
        if t > thr:
            thr = t
    thr = round(thr, 3)
    logger.warning(f"no threshold reaches precision>={min_precision:.3f} "
                   f"(best precision={best_pr:.3f}); using max-precision threshold {thr}")
    if strict:
        _reject_degenerate(y_true, proba, thr)
    return thr, True


def _reject_degenerate(
    y_true: pd.Series | np.ndarray, proba: np.ndarray, thr: float
) -> None:
    """Fail-closed on operating-point collapse (100% alarms)."""
    masyv = np.asarray(proba)
    pred = []
    for v in masyv:
        if v >= thr:
            pred.append(1)
        else:
            pred.append(0)
    pred = np.asarray(pred)
    yt = np.asarray(y_true)
    if len(pred) == 0:
        pred_rate = 0.0
    else:
        pred_rate = float(pred.mean())
    n_neg = int((yt == 0).sum())
    tn = int(((yt == 0) & (pred == 0)).sum())
    if pred_rate > 0.99 or (n_neg > 0 and tn == 0):
        raise ValueError(
            f"degenerate operating point rejected: thr={thr} pred_rate={pred_rate:.3f} TN={tn}"
        )


def select_operating_point(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    min_recall: float = 0.85,
    min_precision: float = 0.25,
    strict: bool = False,
) -> tuple[float, bool]:
    """Operational rule: max precision subject to recall >= min_recall AND precision >= min_precision.

    If no threshold reaches both constraints simultaneously, falls back to the threshold
    maximizing precision among those reaching min_recall, or finally max-recall point.
    Ties -> HIGHER threshold (reduces false alarms).
    With strict=True raises ValueError on degenerate thresholds (pred_rate>0.99 or TN==0)."""
    grid = _zrobyty_sitku()
    scored = []
    for t in grid:
        pred = (proba >= t).astype(int)
        rc = float(recall_score(y_true, pred, zero_division=0))
        pr = float(precision_score(y_true, pred, zero_division=0))
        scored.append((t, rc, pr))
    ok = []
    for t, rc, pr in scored:
        if rc >= min_recall and pr >= min_precision:
            ok.append((t, pr))
    if len(ok) > 0:
        best_pr = ok[0][1]
        for t, pr in ok:
            if pr > best_pr:
                best_pr = pr
        kandidaty = []
        for t, pr in ok:
            if pr == best_pr:
                kandidaty.append(t)
        thr = kandidaty[0]
        for t in kandidaty:
            if t > thr:
                thr = t
        thr = round(thr, 3)
        if strict:
            _reject_degenerate(y_true, proba, thr)
        return thr, False
    recall_ok = []
    for t, rc, pr in scored:
        if rc >= min_recall:
            recall_ok.append((t, pr))
    if len(recall_ok) > 0:
        best_pr = recall_ok[0][1]
        for t, pr in recall_ok:
            if pr > best_pr:
                best_pr = pr
        kandidaty = []
        for t, pr in recall_ok:
            if pr == best_pr:
                kandidaty.append(t)
        thr = kandidaty[0]
        for t in kandidaty:
            if t > thr:
                thr = t
        thr = round(thr, 3)
        logger.warning(f"operating point missed precision guardrail ({min_precision:.3f}); "
                       f"using thr={thr}")
        if strict:
            _reject_degenerate(y_true, proba, thr)
        return thr, True
    best_rc = scored[0][1]
    for t, rc, pr in scored:
        if rc > best_rc:
            best_rc = rc
    kandidaty = []
    for t, rc, pr in scored:
        if rc == best_rc:
            kandidaty.append(t)
    thr = kandidaty[0]
    for t in kandidaty:
        if t > thr:
            thr = t
    thr = round(thr, 3)
    logger.warning(f"no threshold reaches recall>={min_recall:.3f} "
                   f"(best recall={best_rc:.3f}); using max-recall threshold {thr}")
    if strict:
        _reject_degenerate(y_true, proba, thr)
    return thr, True
