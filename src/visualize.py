"""Extra figures for AirAlert-TSF. Writes PNGs to <root>/reports/figures/."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score

from src.config import Settings
from src.data_loader import load_raw
from src.feature_engineering import make_features

FIGURES = [
    "fig_hour_weekday.png",
    "fig_durations.png",
    "fig_xcorr.png",
    "fig_models.png",
    "fig_threshold.png",
    "fig_proba_hist.png",
    "fig_confusion.png",
]


def visualize(root: str | Path, target: str = "Kyiv") -> list[str]:

    root = Path(root)
    grid_path = root / "data" / "processed" / "grid.csv"
    raw_path = root / "data" / "raw" / "alerts.csv"
    metrics_path = root / "reports" / "metrics.json"
    bundle_path = root / "reports" / "model.pkl"
    missing = [str(p) for p in (grid_path, raw_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing} (run main.py first)")
    out = root / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = Settings()
    from src.config import TRANSIT_REGIONS

    neighbors = [n for n in TRANSIT_REGIONS.get(target, cfg.neighbors) if n != target]
    grid = pd.read_csv(grid_path, parse_dates=["timestamp"], index_col="timestamp")
    grid.index = pd.to_datetime(grid.index, utc=True)
    raw = load_raw(raw_path)
    raw = raw[raw["region"] == target]

    # 1. Kyiv-local hour x weekday heatmap
    local = grid.index.tz_convert("Europe/Kyiv")
    piv = grid[target].groupby([local.hour, local.dayofweek]).mean().unstack()
    plt.figure(figsize=(10, 5))
    plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.colorbar(label="alert rate")
    plt.xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xlabel("weekday (Kyiv)")
    plt.ylabel("hour (Kyiv)")
    plt.title(f"{target} alert rate by hour and weekday")
    plt.tight_layout()
    plt.savefig(out / "fig_hour_weekday.png", dpi=120)
    plt.close()

    # 2. Alert durations
    dur = (raw["end_time"] - raw["start_time"]).dt.total_seconds() / 60
    plt.figure()
    plt.hist(np.clip(dur, 0, 600), bins=60)
    plt.xlabel("duration, min (clipped at 600)")
    plt.ylabel("alerts")
    plt.title(f"Alert durations (n={len(dur)}, median={dur.median():.0f} min)")
    plt.tight_layout()
    plt.savefig(out / "fig_durations.png", dpi=120)
    plt.close()

    # 3. Neighbor -> target cross-correlation by lag (correlation only, not proof of lead)
    plt.figure()
    for n in neighbors:
        if n in grid.columns:
            plt.plot(range(7), [grid[target].corr(grid[n].shift(L)) for L in range(7)],
                     marker="o", label=n)
    plt.xlabel("lag, hours")
    plt.ylabel(f"corr with {target}")
    plt.title(f"Neighbor correlation with {target} (correlation, not proof of lead)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_xcorr.png", dpi=120)
    plt.close()

    # 4. Model PR-AUC comparison from metrics.json
    import json

    metrics_path = root / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path} (run main.py first)")
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)
    names = list(m["results"])
    vals = [m["results"][k]["pr_auc"] for k in names]
    prev = m["prevalence"]["test"]
    data_source = m.get("data_source", "unknown")
    thr = float(m["results"]["main"]["threshold"])
    plt.figure(figsize=(9, 4))
    bars = plt.bar(names, vals)
    plt.axhline(prev, linestyle="--", label=f"prevalence {prev:.3f}")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("PR-AUC (holdout)")
    plt.title(f"Model comparison ({data_source} data)")
    plt.legend()
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}",
                 ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "fig_models.png", dpi=120)
    plt.close()

    # 5-7. Reuse the saved model bundle instead of retraining
    import joblib

    bundle_path = root / "reports" / "model.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing {bundle_path} (run main.py first)")
    bundle = joblib.load(bundle_path)
    model, thr = bundle["model"], float(bundle["threshold"])
    target = m.get("target", target)
    neighbors = [n for n in TRANSIT_REGIONS.get(target, cfg.neighbors) if n != target]
    X, y = make_features(grid, target, neighbors)
    cut = X.index.max() - pd.Timedelta(days=cfg.test_days)
    gap = cut - pd.Timedelta(hours=cfg.horizon_h)
    Xte = X[(X.index > cut)][bundle["feature_names"]]
    yte = y[y.index > cut]
    _gap_hours = (Xte.index.min() - X[X.index <= gap].index.max()).total_seconds() / 3600
    logger.info(f"visualize: embargo gap to test = {_gap_hours:.1f}h")
    proba = model.predict_proba(Xte)[:, 1]
    yv = yte.values
    logger.info(f"visualize: bundle thr={thr:.2f}, test_hours={len(yv)}")

    ths = np.arange(0.05, 0.96, 0.05)
    f2 = [fbeta_score(yv, (proba >= t).astype(int), beta=2, zero_division=0) for t in ths]
    pr = [precision_score(yv, (proba >= t).astype(int), zero_division=0) for t in ths]
    rc = [recall_score(yv, (proba >= t).astype(int), zero_division=0) for t in ths]
    plt.figure()
    plt.plot(ths, f2, label="F2")
    plt.plot(ths, pr, label="precision")
    plt.plot(ths, rc, label="recall")
    plt.xlabel("threshold")
    backend = m.get("backend", "model")
    plt.title(f"Threshold sweep ({backend}, test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_threshold.png", dpi=120)
    plt.close()

    plt.figure()
    plt.hist(proba[yv == 0], bins=40, alpha=0.6, label="calm")
    plt.hist(proba[yv == 1], bins=40, alpha=0.6, label="alert")
    plt.xlabel("predicted probability")
    plt.ylabel("hours")
    plt.title("Probability distribution by true class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_proba_hist.png", dpi=120)
    plt.close()

    cm = confusion_matrix(yv, (proba >= thr).astype(int))
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", fontsize=14)
    plt.xticks([0, 1], ["pred calm", "pred alert"])
    plt.yticks([0, 1], ["true calm", "true alert"])
    plt.title(f"Confusion matrix @ thr={thr:.2f}")
    plt.tight_layout()
    plt.savefig(out / "fig_confusion.png", dpi=120)
    plt.close()

    done = sorted(p.name for p in out.glob("*.png"))
    logger.info(f"figures: {done}")
    return done


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
    visualize(root)
