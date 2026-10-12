from itertools import pairwise
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BootstrapResult(TypedDict):
    point: float
    mean: float
    lo: float
    hi: float
    n_boot: int


def safe_roc_auc(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(np.asarray(y_true))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def safe_pr_auc(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(np.asarray(y_true))) < 2:
        return float("nan")
    return float(average_precision_score(y_true, proba))


def brier_skill_score(
    y_true: pd.Series | np.ndarray, proba: np.ndarray, ref: float | None = None
) -> float:
    """BSS vs a constant reference. ref must be TRAIN prevalence (never test labels)."""
    yt = np.asarray(y_true, dtype=float)
    ref = float(yt.mean()) if ref is None else float(ref)
    denom = float(((yt - ref) ** 2).mean())
    if denom == 0:
        return float("nan")
    return float(1.0 - float(((yt - np.asarray(proba)) ** 2).mean()) / denom)


def expected_calibration_error(
    y_true: pd.Series | np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    yt, pr = np.asarray(y_true, dtype=float), np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (pr > lo) & (pr <= hi) if lo > 0 else (pr >= lo) & (pr <= hi)
        if m.sum():
            ece += m.mean() * abs(yt[m].mean() - pr[m].mean())
    return float(ece)


def budget_metrics(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> dict:
    """Operating points at fixed alert budgets + precision at recall 0.9."""
    yt, pr = np.asarray(y_true, dtype=int), np.asarray(proba, dtype=float)
    out: dict = {}
    for budget in (0.3, 0.5):
        thr = float(np.quantile(pr, 1.0 - budget))
        pred = (pr >= thr).astype(int)
        out[f"budget_{int(budget * 100)}"] = {
            "threshold": round(thr, 4),
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "precision": float(precision_score(yt, pred, zero_division=0)),
        }
    order = np.argsort(-pr, kind="stable")
    ranked = yt[order]
    cum_recall = np.cumsum(ranked) / max(ranked.sum(), 1)
    k = int(np.searchsorted(cum_recall, 0.9)) + 1
    out["precision_at_recall_09"] = round(float(ranked[:k].mean()), 4)
    return out


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = np.arange(0, n, block)
    full = starts[starts + block <= n]
    pool = full if len(full) else np.array([0])
    size = len(starts)
    picked = rng.choice(pool, size)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in picked])[:n]


def _summarize(vals: list[float], point: float) -> dict:
    if not vals:
        return {"point": round(float(point), 4), "mean": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    return {
        "point": round(float(point), 4),
        "mean": round(float(np.mean(vals)), 4),
        "lo": round(float(np.percentile(vals, 2.5)), 4),
        "hi": round(float(np.percentile(vals, 97.5)), 4),
        "n_boot": len(vals),
    }


def block_bootstrap_delta_pr_auc(
    y_true: pd.Series | np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    block: int = 24,
    n_boot: int = 1000,
    seed: int = 42,
    mask: np.ndarray | None = None,
) -> BootstrapResult:
    """Paired CI for (A minus B) PR-AUC on identical block resamples.

    Blocks are drawn from the FULL series to preserve autocorrelation;
    mask (e.g. onset) is applied inside each iteration.
    """
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    pa, pb = np.asarray(proba_a), np.asarray(proba_b)
    n = len(yt)
    m = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    vals = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        sub = idx[m[idx]]
        if len(sub) < 2:
            continue
        da = safe_pr_auc(yt[sub], pa[sub])
        db = safe_pr_auc(yt[sub], pb[sub])
        if not (np.isnan(da) or np.isnan(db)):
            vals.append(da - db)
    n_masked = int(m.sum())
    base = (safe_pr_auc(yt[m], pa[m]) - safe_pr_auc(yt[m], pb[m])) if n_masked >= 2 else float("nan")
    return _summarize(vals, base)

def block_bootstrap_metric(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    metric: str = "pr_auc",
    threshold: float = 0.5,
    block: int = 24,
    n_boot: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Block-bootstrap CI for pr_auc, roc_auc, f2, precision or recall."""
    fns = {
        "pr_auc": lambda yt, pr: safe_pr_auc(yt, pr),
        "roc_auc": lambda yt, pr: safe_roc_auc(yt, pr),
        "f2": lambda yt, pr: float(fbeta_score(yt, (pr >= threshold).astype(int),
                                               beta=2, zero_division=0)),
        "precision": lambda yt, pr: float(precision_score(yt, (pr >= threshold).astype(int),
                                                           zero_division=0)),
        "recall": lambda yt, pr: float(recall_score(yt, (pr >= threshold).astype(int),
                                                    zero_division=0)),
    }
    fn = fns[metric]
    rng = np.random.default_rng(seed)
    yt, pr = np.asarray(y_true), np.asarray(proba)
    n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        v = fn(yt[idx], pr[idx])
        if not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    base = fn(yt, pr)
    base = float("nan") if (isinstance(base, float) and np.isnan(base)) else base
    return _summarize(vals, base)


def block_bootstrap_pr_auc(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    block: int = 24,
    n_boot: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    yt, pr = np.asarray(y_true), np.asarray(proba)
    n = len(yt)
    vals = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        v = safe_pr_auc(yt[idx], pr[idx])
        if not np.isnan(v):
            vals.append(v)
    return _summarize(vals, safe_pr_auc(yt, pr))


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
    prev: pd.Series | np.ndarray | None = None,
    ref: float | None = None,
) -> dict:
    pred = (proba >= threshold).astype(int)
    out = {
        "pr_auc": safe_pr_auc(y_true, proba),
        "roc_auc": safe_roc_auc(y_true, proba),
        "brier": float(brier_score_loss(y_true, proba)),
        "bss": brier_skill_score(y_true, proba, ref=ref),
        "ece": expected_calibration_error(y_true, proba),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "threshold": float(threshold),
    }
    if prev is not None:
        pv = np.asarray(prev)
        yt, pr = np.asarray(y_true), np.asarray(proba)
        for name, mask in (("onset", pv == 0), ("cont", pv == 1)):
            if mask.sum() == 0:
                continue
            pa = safe_pr_auc(yt[mask], pr[mask])
            out[name] = {
                "n": int(mask.sum()),
                "pos_rate": round(float(yt[mask].mean()), 4),
                "pr_auc": pa,
                "lift": round(float(pa - yt[mask].mean()), 4) if not np.isnan(pa) else float("nan"),
                "f2": float(
                    fbeta_score(yt[mask], (pr[mask] >= threshold).astype(int), beta=2, zero_division=0)
                ),
            }
    return out


def save_plots(
    curves: dict[str, tuple],
    prevalence: float,
    threshold_point: tuple[float, float] | None,
    out_dir: str,
) -> None:
    """curves: name -> (y_true, proba). PR curves of all models + prevalence line."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    try:
        for name, (yt, pr) in curves.items():
            prec, rec, _ = precision_recall_curve(yt, pr)
            plt.plot(rec, prec, label=name)
        plt.axhline(prevalence, linestyle="--", label=f"prevalence {prevalence:.3f}")
        if threshold_point is not None:
            plt.plot([threshold_point[0]], [threshold_point[1]], "o", label="operating point")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curves")
        plt.legend()
        plt.savefig(f"{out_dir}/pr_curve.png", dpi=120)
    finally:
        plt.close()


def save_calibration(
    curves: dict[str, tuple],
    out_dir: str,
) -> None:
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    try:
        for name, (yt, pr) in curves.items():
            if len(np.unique(np.asarray(pr))) < 3:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(yt, pr, n_bins=10)
            except ValueError:
                continue
            plt.plot(mean_pred, frac_pos, marker="o", label=name)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted proba")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration plot")
        plt.legend()
        plt.savefig(f"{out_dir}/calibration.png", dpi=120)
    finally:
        plt.close()


def _sanitize(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def save_metrics_json(payload: dict, path: str) -> None:
    import json
    import sys
    from pathlib import Path

    import pandas
    import sklearn

    try:
        import lightgbm

        lgbm_v = lightgbm.__version__
    except ImportError:
        lgbm_v = "not installed"

    payload = dict(payload)
    payload["versions"] = {
        "sklearn": sklearn.__version__,
        "lightgbm": lgbm_v,
        "pandas": pandas.__version__,
        "python": sys.version.split()[0],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_sanitize(payload), indent=2, default=str, allow_nan=False)
    Path(path).write_text(text)
    json.loads(text)


def save_model_bundle(model, feature_names: list[str], threshold: float, path: str) -> None:
    import sys
    from pathlib import Path

    import joblib
    import sklearn

    try:
        import lightgbm

        lgbm_v = lightgbm.__version__
    except ImportError:
        lgbm_v = None

    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "threshold": float(threshold),
        "metadata": {
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_v,
            "python_version": sys.version.split()[0],
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
