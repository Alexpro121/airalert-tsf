"""Build reports/index.html: offline dashboard with data embedded as JSON."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import Settings
from src.data_loader import load_raw
from src.feature_engineering import make_features
from src.grid_builder import build_hourly_grid
from src.models import (
    AlwaysAlert,
    ClimatologyBaseline,
    HeuristicBaseline,
    LagHourBaseline,
    PersistenceBaseline,
    train_dummy_logistic,
)

ROOT = Path(__file__).parent.parent
COLORS = {
    "main": "#e94560",
    "persistence": "#0f3460",
    "climatology": "#16213e",
    "heuristic": "#533483",
    "always": "#ff6b6b",
    "dummy_logistic": "#feca57",
    "laghour": "#00adb5",
}


def ftype(name: str) -> str:
    if name.startswith("lag_"):
        return "lags"
    if name.startswith("roll_"):
        return "windows"
    if name.startswith("nbr_") or name == "country_ratio":
        return "neighbors"
    if name.startswith("has_"):
        return "text"
    return "time"


def _pos_proba(model, proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    classes = list(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:
        if classes[0] == 1:
            return np.full(proba.shape[0], 1.0)
        return np.full(proba.shape[0], 0.0)
    if proba.ndim == 1:
        return proba.astype(float)
    if proba.shape[1] == 1:
        if classes[0] == 1:
            return proba[:, 0].astype(float)
        return (1.0 - proba[:, 0]).astype(float)
    surs_pozytsiya = list(classes).index(1)
    return proba[:, surs_pozytsiya].astype(float)


def main() -> None:
    import joblib

    from src.grid_builder import coverage_mask

    bundle_path = ROOT / "reports" / "model.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing {bundle_path}; run main.py first")
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    with open(ROOT / "reports" / "metrics.json", encoding="utf-8") as _f:
        _meta0 = json.load(_f)
    _mcfg = _meta0.get("config", {})
    cfg = Settings(
        target_region=_meta0.get("target", "Kyiv"),
        neighbors=_mcfg.get("neighbors", Settings().neighbors),
        horizon_h=_meta0.get("horizon_h", 3),
        lags=_mcfg.get("lags", [1, 2, 3, 6, 24]),
        windows=_mcfg.get("windows", [6, 12, 24]),
        test_days=_meta0.get("test_days", 30),
    )
    df_raw = load_raw(ROOT / cfg.raw_path)
    regions = [cfg.target_region, *[n for n in cfg.neighbors if n != cfg.target_region]]
    grid = build_hourly_grid(df_raw, regions, freq=cfg.freq)
    grid = grid.where(coverage_mask(df_raw, regions, grid.index))
    if cfg.tail_cutoff_hours:
        grid = grid.loc[: grid.index.max() - pd.Timedelta(hours=cfg.tail_cutoff_hours)]
    else:
        _dur = (df_raw["end_time"] - df_raw["start_time"]).dt.total_seconds() / 3600
        _auto = int(max(1, _dur.quantile(0.99)))
        grid = grid.loc[: grid.index.max() - pd.Timedelta(hours=_auto)]
    X, y = make_features(
        grid, cfg.target_region, cfg.neighbors,
        horizon=cfg.horizon_h, lags=cfg.lags, windows=cfg.windows,
    )
    cut = X.index.max() - pd.Timedelta(days=cfg.test_days)
    gap = cut - pd.Timedelta(hours=cfg.horizon_h)
    Xtr, Xte = X[X.index <= gap], X[X.index > cut]
    feat_names: list[str] = list(bundle["feature_names"])
    missing = [c for c in feat_names if c not in Xte.columns]
    if missing:
        raise ValueError(f"dashboard feature mismatch: bundle needs {missing} not built by make_features")
    Xte_model = Xte[feat_names]
    ytr, yte = y[y.index <= gap], y[y.index > cut]

    nbr_cols = [f"nbr_{n}" for n in cfg.neighbors if n != cfg.target_region]
    makers = {
        "heuristic": (HeuristicBaseline(nbr_cols), None),
        "persistence": (PersistenceBaseline(), None),
        "always": (AlwaysAlert(), None),
        "climatology": (ClimatologyBaseline().fit(Xtr, ytr), None),
        "laghour": (LagHourBaseline().fit(Xtr, ytr), None),
        "dummy_logistic": (train_dummy_logistic(Xtr, ytr, cfg.seed), "cal"),
        "main": (model, "bundle"),
    }
    with open(ROOT / "reports" / "metrics.json", encoding="utf-8") as f:
        meta = json.load(f)
    out_models = {}
    for name, (m, kind) in makers.items():
        if kind == "bundle":
            raw_p = m.predict_proba(Xte_model)
        elif kind == "cal":
            raw_p = m.predict_proba(Xte[m.feature_names_in_])
        else:
            raw_p = m.predict_proba(Xte)
        proba = _pos_proba(m, raw_p)
        prec, rec, _ = precision_recall_curve(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        step = max(1, len(fpr) // 200)
        try:
            fp_, mp_ = calibration_curve(yte, proba, n_bins=10)
            cal = {"x": [round(float(v), 4) for v in mp_], "y": [round(float(v), 4) for v in fp_]}
        except ValueError:
            cal = {"x": [], "y": []}
        out_models[name] = {
            "proba": [round(float(v), 4) for v in proba],
            "pr": {
                "x": [round(float(v), 4) for v in rec.tolist()],
                "y": [round(float(v), 4) for v in prec.tolist()],
            },
            "roc": {
                "x": [round(float(v), 4) for v in fpr[::step].tolist()],
                "y": [round(float(v), 4) for v in tpr[::step].tolist()],
            },
            "cal": cal,
            "metrics": meta["results"][name],
        }
    if hasattr(model, "booster_"):
        gain = model.booster_.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importances_"):
        gain = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        gain = np.abs(np.asarray(model.coef_).ravel()).astype(float)
    else:
        try:
            from sklearn.inspection import permutation_importance

            r = permutation_importance(model, Xte_model, yte, n_repeats=5, random_state=0)
            gain = np.asarray(r.importances_mean, dtype=float)
        except Exception:  # noqa: BLE001 - dashboard must not fail on importance
            gain = np.zeros(len(Xte_model.columns), dtype=float)
    surs_pary = []
    surs_spysok = list(Xte_model.columns)
    for i in range(len(surs_spysok)):
        surs_pary.append((surs_spysok[i], float(gain[i])))
    surs_pary.sort(key=lambda t: t[1], reverse=True)
    imp = surs_pary[:15]
    payload = {
        "colors": COLORS,
        "meta": {
            k: meta[k]
            for k in (
                "target",
                "horizon_h",
                "seed",
                "backend",
                "test_days",
                "data_source",
                "use_text_signals",
                "threshold_rule",
            )
            if k in meta
        }
        | {"prevalence": meta["prevalence"], "versions": meta["versions"]},
        "models": out_models,
        "features": [{"name": n, "gain": round(g, 1), "type": ftype(n)} for n, g in imp],
        "series": {
            "t": [t.isoformat() for t in Xte.index],
            "y": [int(v) for v in yte.values],
            "threshold": meta["results"]["main"]["threshold"],
        },
    }
    tpl = (ROOT / "src" / "dashboard_template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (ROOT / "reports" / "index.html").write_text(html, encoding="utf-8")
    print("index.html bytes:", len(html.encode()))


if __name__ == "__main__":
    main()
