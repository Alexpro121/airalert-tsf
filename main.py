"""AirAlert-TSF pipeline.

Forecast moment T = start of hour t. Known: A[tau], tau <= t-1.
Target: Y[t] = max(A[t..t+H-1]). Features use shift(>=1); no dead hour.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from src.config import TRANSIT_REGIONS, Settings
from src.data_loader import (
    fetch_real_alerts,
    generate_synthetic,
    load_raw,
    read_source,
    sha256_file,
    validate_regions,
)
from src.evaluate import (
    block_bootstrap_delta_pr_auc,
    block_bootstrap_metric,
    block_bootstrap_pr_auc,
    budget_metrics,
    compute_metrics,
    safe_pr_auc,
    safe_roc_auc,
    save_calibration,
    save_metrics_json,
    save_model_bundle,
    save_plots,
)
from src.explain import save_importance
from src.feature_engineering import make_features
from src.grid_builder import build_hourly_grid, coverage_mask
from src.models import (
    AlwaysAlert,
    ClimatologyBaseline,
    HeuristicBaseline,
    LagHourBaseline,
    PersistenceBaseline,
    TrailingBaseline,
    select_operating_point,
    train_dummy_logistic,
    train_lgbm,
)
from src.text_signals import ALL_CLEAR_KEYWORDS, build_hourly_flags, load_events


def resolve_paths(cfg: Settings) -> dict[str, Path]:
    root = Path(__file__).parent

    def _abs(p: str) -> Path:
        return root / p if not Path(p).is_absolute() else Path(p)

    return {
        "root": root,
        "raw": _abs(cfg.raw_path),
        "synthetic": _abs(cfg.synthetic_path),
        "grid": _abs(cfg.grid_path),
        "reports": _abs(cfg.report_dir),
        "airforce": _abs(cfg.airforce_path),
    }


def detect_backend() -> str:
    try:
        import lightgbm  # noqa: F401
        from lightgbm import LGBMClassifier

        X_probe = [[0.0] * 3 for _ in range(20)]
        LGBMClassifier(n_estimators=2, verbose=-1).fit(X_probe, [0, 1] * 10)
        return "lightgbm"
    except (ImportError, OSError) as e:
        logger.warning(f"lightgbm unavailable ({type(e).__name__}: {e}) -> HistGradientBoosting")
        return "hgb"
    except Exception as e:  # noqa: BLE001 - any native failure means fallback
        logger.warning(f"lightgbm smoke-fit failed ({e}) -> HistGradientBoosting")
        return "hgb"


def train_model(X: pd.DataFrame, y: pd.Series, cfg: Settings, backend: str):
    # Single rule for both backends: no class weights (keeps probabilities comparable).
    klasy = np.unique(np.asarray(y))
    if len(klasy) < 2:
        raise ValueError(f"train_model needs 2 classes, got {np.unique(np.asarray(y))}")
    if backend == "lightgbm":
        surs_rez = train_lgbm(X, y, cfg.resolved_lgbm_params())
        return surs_rez
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(random_state=cfg.seed, early_stopping=False)
    m.fit(X, y)
    return m


def oof_predict(train_fn, predict_fn, X_tr: pd.DataFrame, y_tr: pd.Series, horizon_h: int, n_splits: int):
    """Out-of-fold probabilities with a horizon gap between train and validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon_h)
    oof = pd.Series(float("nan"), index=X_tr.index)
    surs_nomer = 0
    for tr_idx, va_idx in tscv.split(X_tr):
        surs_nomer = surs_nomer + 1
        try:
            m = train_fn(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        except ValueError as e:
            from sklearn.dummy import DummyClassifier

            logger.warning(f"fold{surs_nomer}: single-class train ({e}) -> DummyClassifier(prior)")
            m = DummyClassifier(strategy="prior")
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pv = _proba_pos(m, m.predict_proba(X_tr.iloc[va_idx]))
            oof.iloc[va_idx] = pv
            continue
        pv = predict_fn(m, X_tr.iloc[va_idx])
        oof.iloc[va_idx] = pv
        logger.info(
            f"fold{surs_nomer}: PR-AUC={safe_pr_auc(y_tr.iloc[va_idx], pv):.4f} "
            f"ROC-AUC={safe_roc_auc(y_tr.iloc[va_idx], pv):.4f}"
        )
    return oof


def _proba_pos(model, proba: np.ndarray) -> np.ndarray:
    """Map predict_proba output to P(y=1), safe for single-class models."""
    proba = np.asarray(proba)
    classes = list(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:
        if classes[0] == 1:
            return np.full(proba.shape[0], 1.0)
        else:
            return np.full(proba.shape[0], 0.0)
    if proba.ndim == 1:
        return proba.astype(float)
    if proba.shape[1] == 1:
        if classes[0] == 1:
            return proba[:, 0].astype(float)
        else:
            return (1.0 - proba[:, 0]).astype(float)
    surs_pozytsiya = list(classes).index(1)
    return proba[:, surs_pozytsiya].astype(float)


def proba_full(model, X: pd.DataFrame) -> np.ndarray:
    surs_syri = model.predict_proba(X)
    return _proba_pos(model, surs_syri)


def proba_cal(model, X: pd.DataFrame) -> np.ndarray:
    surs_syri = model.predict_proba(X[model.feature_names_in_])
    return _proba_pos(model, surs_syri)


def load_dataset(cfg: Settings, paths: dict[str, Path], args) -> tuple[pd.DataFrame, str]:
    if args.synthetic:
        df = generate_synthetic(str(paths["synthetic"]), days=args.days, seed=cfg.seed)
        return df, "synthetic"
    if args.real:
        df = fetch_real_alerts(str(paths["raw"]), required=[cfg.target_region, *cfg.neighbors])
        return df, "real"
    if paths["raw"].exists():
        src = read_source(paths["raw"])
        if src == "unknown" and getattr(args, "strict_source", False):
            raise ValueError(f"{paths['raw']} source is unknown; refusing to run (see --strict-source)")
        if src == "unknown":
            logger.warning(f"{paths['raw']} source is unknown (no verified meta)")
        logger.info(f"using existing {paths['raw']} (source={src})")
        return load_raw(paths["raw"]), src
    raise FileNotFoundError(
        "no data file. Use --real to download or --synthetic to generate "
        f"(would write {paths['synthetic']}, never overwrites {paths['raw']})."
    )


def build_text_features(
    cfg: Settings, paths: dict[str, Path], grid: pd.DataFrame
) -> tuple[pd.Series | None, pd.Series | None, int]:
    """Returns (threat, all_clear, active_hours). Third value proves real usage."""
    if not cfg.use_text_signals:
        return None, None, 0
    msgs = load_events(paths["airforce"])
    if msgs.empty:
        logger.warning(f"text signals enabled but {paths['airforce']} missing/empty -> zeros")
        return None, None, 0
    threat = build_hourly_flags(msgs, grid.index, exclude_keywords=ALL_CLEAR_KEYWORDS)
    cleared = build_hourly_flags(msgs, grid.index, keywords=ALL_CLEAR_KEYWORDS)
    return threat, cleared, int((threat + cleared).astype(bool).sum())


def prepare_data(cfg: Settings, args) -> tuple[dict[str, Path], pd.DataFrame, str, pd.DataFrame]:
    paths = resolve_paths(cfg)
    df_raw, data_source = load_dataset(cfg, paths, args)
    validate_regions(df_raw, [cfg.target_region, *cfg.neighbors], cfg.min_events_per_region)
    regions = [cfg.target_region, *[n for n in cfg.neighbors if n != cfg.target_region]]
    grid = build_hourly_grid(df_raw, regions, freq=cfg.freq)
    cov = coverage_mask(df_raw, regions, grid.index)
    n_masked = int((~cov).sum().sum())
    if n_masked:
        logger.info(f"masking {n_masked} region-hours outside observation coverage as NaN")
    grid = grid.where(cov)
    if cfg.tail_cutoff_hours:
        grid = grid.loc[: grid.index.max() - pd.Timedelta(hours=cfg.tail_cutoff_hours)]
    else:
        dur = (df_raw["end_time"] - df_raw["start_time"]).dt.total_seconds() / 3600
        auto_tail = int(max(1, dur.quantile(0.99)))
        logger.info(f"auto tail cutoff from p99 duration: {auto_tail}h")
        grid = grid.loc[: grid.index.max() - pd.Timedelta(hours=auto_tail)]
    paths["grid"].parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(paths["grid"])
    return paths, df_raw, data_source, grid


def split_xy(X: pd.DataFrame, y: pd.Series, cfg: Settings):
    cutoff = X.index.max() - pd.Timedelta(days=cfg.test_days)
    gap_end = cutoff - pd.Timedelta(hours=cfg.horizon_h)
    X_tr, X_te = X[X.index <= gap_end], X[X.index > cutoff]
    y_tr, y_te = y[y.index <= gap_end], y[y.index > cutoff]
    logger.info(f"train={len(X_tr)} test={len(X_te)} cutoff={cutoff} embargo={cfg.horizon_h}h")
    return X_tr, X_te, y_tr, y_te, cutoff


def run(cfg: Settings, args) -> dict:
    paths, _df_raw, data_source, grid = prepare_data(cfg, args)
    events, cleared, text_hours = build_text_features(cfg, paths, grid)
    X, y = make_features(
        grid, cfg.target_region, cfg.neighbors, horizon=cfg.horizon_h,
        lags=cfg.lags, windows=cfg.windows, events=events, all_clear=cleared,
    )
    X_tr, X_te, y_tr, y_te, _cutoff = split_xy(X, y, cfg)
    train_prev, test_prev = float(y_tr.mean()), float(y_te.mean())
    logger.info(f"prevalence train={train_prev:.4f} test={test_prev:.4f}")
    trail30 = float(y_tr.iloc[-720:].mean()) if len(y_tr) >= 720 else train_prev
    logger.info(f"trailing-30d rate (BSS reference, known at forecast time): {trail30:.4f}")
    if events is not None:
        for name, s in (("threat", events), ("all_clear", cleared)):
            logger.info(f"text {name}: train-hours={int(s[X_tr.index].sum())} "
                        f"test-hours={int(s[X_te.index].sum())}")

    backend = detect_backend()
    logger.info(f"backend={backend}")
    cal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night"]
    if getattr(args, "no_calendar", False):
        drop = [c for c in cal_cols if c in X_tr.columns]
        Xtr_m, Xte_m = X_tr.drop(columns=drop), X_te.drop(columns=drop)
        logger.info(f"ablation: main model without calendar cols {drop}; baselines keep full X")
    else:
        Xtr_m, Xte_m = X_tr, X_te
    oof_main = oof_predict(
        lambda a, b: train_model(a, b, cfg, backend), proba_full,
        Xtr_m, y_tr, cfg.horizon_h, cfg.n_splits,
    )
    oof_dummy = oof_predict(
        lambda a, b: train_dummy_logistic(a, b, cfg.seed), proba_cal,
        X_tr, y_tr, cfg.horizon_h, cfg.n_splits,
    )
    def clim_factory(a, b):
        return ClimatologyBaseline().fit(a, b)

    def laghour_factory(a, b):
        return LagHourBaseline().fit(a, b)

    oof_clim = oof_predict(clim_factory, proba_full, X_tr, y_tr, cfg.horizon_h, cfg.n_splits)
    oof_laghour = oof_predict(laghour_factory, proba_full, X_tr, y_tr, cfg.horizon_h, cfg.n_splits)
    mask = oof_main.notna()
    logger.info(f"OOF PR-AUC={safe_pr_auc(y_tr[mask], oof_main[mask]):.4f}")

    def guardrail() -> float:
        return min(0.95, max(cfg.threshold_floor,
                             cfg.threshold_precision_ratio * float(y_tr[mask].mean())))

    gr = guardrail()
    logger.info(f"threshold guardrail precision>={gr:.3f} (ratio={cfg.threshold_precision_ratio})")
    op_rule = f"max precision s.t. recall>=0.85 and precision>={gr:.3f} on pooled OOF"

    def surs_vziaty_porig(surs_y, surs_oof, surs_nazva):
        try:
            return select_operating_point(
                surs_y, surs_oof, min_recall=0.85, min_precision=gr, strict=True)
        except ValueError as surs_pomylka:
            logger.error(f"{surs_nazva}: strict operating point degenerate "
                         f"({surs_pomylka}) -> non-strict")
            surs_thr, _ = select_operating_point(
                surs_y, surs_oof, min_recall=0.85, min_precision=gr, strict=False)
            return surs_thr, True

    thr_main, fb_main = surs_vziaty_porig(y_tr[mask], oof_main[mask].values, "main")
    thr_dummy, fb_dummy = surs_vziaty_porig(y_tr[mask], oof_dummy[mask].values, "dummy")
    thr_clim, fb_clim = surs_vziaty_porig(y_tr[mask], oof_clim[mask].values, "clim")
    thr_laghour, fb_laghour = surs_vziaty_porig(y_tr[mask], oof_laghour[mask].values, "laghour")
    fb_models = [n for n, f in (("main", fb_main), ("dummy", fb_dummy),
                                ("clim", fb_clim), ("laghour", fb_laghour)) if f]
    logger.info(f"operating point rule: {op_rule}; missed-recall fallback: "
                f"{fb_models if fb_models else 'none'}")

    prev = X_te["lag_1"].values if "lag_1" in X_te.columns else None
    model = train_model(Xtr_m, y_tr, cfg, backend)
    proba = proba_full(model, Xte_m)
    pred_rate = float((proba >= thr_main).mean())
    surs_vyrodzhenyi = False
    if pred_rate > 0.99:
        logger.error(f"degenerate operating point: thr={thr_main} flags {pred_rate:.1%} of test "
                     f"(precision≈prevalence={float(y_te.mean()):.3f}) -> fallback to 0.5")
        thr_main = 0.5
        fb_main = True
        surs_vyrodzhenyi = True
        pred_rate = float((proba >= thr_main).mean())
        if pred_rate > 0.99:
            raise ValueError(f"even fallback thr=0.5 is degenerate (pred_rate={pred_rate:.3f})")
    onset_mask_tr = (Xtr_m["lag_1"] == 0).values if "lag_1" in Xtr_m.columns else None
    onset_res: dict = {}
    if onset_mask_tr is not None and onset_mask_tr.sum() > 100 and prev is not None:
        m_on = train_model(Xtr_m[onset_mask_tr], y_tr[onset_mask_tr], cfg, backend)
        p_on = proba_full(m_on, Xte_m)
        mo = (prev == 0)
        onset_res = {
            "pr_auc": safe_pr_auc(np.asarray(y_te)[mo], p_on[mo]),
            "n_train_onset": int(onset_mask_tr.sum()),
        }
        logger.info(f"onset-only model: {onset_res}")
    ci = block_bootstrap_pr_auc(y_te, proba, seed=cfg.seed)
    logger.info(f"test PR-AUC CI(24h blocks): {ci}")
    ci_f2 = block_bootstrap_metric(y_te, proba, metric="f2", threshold=thr_main, seed=cfg.seed)
    ci_roc = block_bootstrap_metric(y_te, proba, metric="roc_auc", seed=cfg.seed)
    logger.info(f"test F2 CI={ci_f2} ROC-AUC CI={ci_roc}")
    pers_proba = PersistenceBaseline().predict_proba(X_te)[:, 1]
    laghour = LagHourBaseline().fit(X_tr, y_tr)
    laghour_proba = laghour.predict_proba(X_te)[:, 1]
    sens = {b: block_bootstrap_delta_pr_auc(y_te, proba, pers_proba, block=b, seed=cfg.seed)
            for b in (24, 72, 168)}
    logger.info(f"block-size sensitivity of delta: {sens}")
    delta = block_bootstrap_delta_pr_auc(y_te, proba, pers_proba, seed=cfg.seed)
    logger.info(f"paired delta PR-AUC (main-persistence): {delta}")
    always_proba = AlwaysAlert().predict_proba(X_te)[:, 1]
    delta_laghour = block_bootstrap_delta_pr_auc(y_te, proba, laghour_proba, seed=cfg.seed)
    logger.info(f"paired delta PR-AUC (main-laghour): {delta_laghour}")
    delta_always = block_bootstrap_delta_pr_auc(y_te, proba, always_proba, seed=cfg.seed)
    logger.info(f"paired delta PR-AUC (main-always): {delta_always}")
    fold_thrs = []
    for _, va_idx in TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.horizon_h).split(X_tr):
        t, _ = select_operating_point(y_tr.iloc[va_idx], oof_main.iloc[va_idx].values)
        fold_thrs.append(t)
    logger.info(f"per-fold OOF thresholds: {fold_thrs}")
    n_edge = sum(1 for t in fold_thrs if t in (0.02, 0.9))
    if n_edge:
        logger.warning(f"{n_edge}/{len(fold_thrs)} folds hit the grid edge (0.02/0.9); optimum may lie outside")
    if thr_main in (0.02, 0.9):
        logger.warning(f"pooled operating point thr={thr_main} hit the grid edge (0.02/0.9)")
    thr_std = float(np.std(fold_thrs))
    if thr_std > 0.1:
        logger.warning(f"high threshold instability: std={thr_std:.3f}")
    deltas_split = {}
    if prev is not None:
        yt, pm = np.asarray(y_te), np.asarray(proba)
        refs = {"laghour": np.asarray(laghour_proba),
                "persistence": np.asarray(pers_proba),
                "always": np.asarray(always_proba)}
        for name, m in (("onset", prev == 0), ("cont", prev == 1)):
            deltas_split[name] = {
                ref: block_bootstrap_delta_pr_auc(yt, pm, pr, seed=cfg.seed, mask=m)
                for ref, pr in refs.items()
            }
        logger.info(f"paired delta PR-AUC by split: {deltas_split}")
    oof_mean = float(oof_main[mask].mean())
    logger.info(f"mean-pred shift: OOF={oof_mean:.4f} holdout={float(proba.mean()):.4f} "
                f"(train_prev={train_prev:.4f} test_prev={test_prev:.4f})")

    nbr_cols = [f"nbr_{n}" for n in cfg.neighbors if n != cfg.target_region]
    heur_proba = HeuristicBaseline(nbr_cols).predict_proba(X_te)[:, 1]
    clim = ClimatologyBaseline().fit(X_tr, y_tr)
    clim_proba = clim.predict_proba(X_te)[:, 1]
    dummy = train_dummy_logistic(X_tr, y_tr, cfg.seed)
    dummy_proba = dummy.predict_proba(X_te[dummy.feature_names_in_])[:, 1]
    trail_proba = TrailingBaseline().predict_proba(X_te)[:, 1]
    results = evaluate_models(
        y_te, prev,
        {"heuristic": heur_proba, "persistence": pers_proba, "always": always_proba,
         "trailing": trail_proba,
         "climatology": clim_proba, "laghour": laghour_proba,
         "dummy_logistic": dummy_proba, "main": proba},
        {"climatology": thr_clim, "laghour": thr_laghour,
         "dummy_logistic": thr_dummy, "main": thr_main},
        trail30,
    )
    results["main"]["budget"] = budget_metrics(y_te, proba)
    results["main"]["pr_auc_ci"] = ci
    results["main"]["f2_ci"] = ci_f2
    results["main"]["roc_auc_ci"] = ci_roc
    results["main"]["block_sensitivity"] = sens
    results["main"]["pr_auc_delta_vs_persistence"] = delta
    results["main"]["pr_auc_delta_vs_laghour"] = delta_laghour
    results["main"]["pr_auc_delta_vs_always"] = delta_always
    results["main"]["pr_auc_delta_by_split"] = deltas_split
    results["main"]["threshold"] = thr_main
    if onset_res:
        results["main"]["onset_only_model"] = onset_res
    for k, v in results.items():
        logger.info(f"{k}: pr_auc={v['pr_auc']:.4f} brier={v['brier']:.4f} "
                    f"prec={v['precision']:.3f} rec={v['recall']:.3f} f2={v['f2']:.4f}")

    trail_proba = TrailingBaseline().predict_proba(X_te)[:, 1]
    curves = {
        "heuristic": (y_te, heur_proba),
        "persistence": (y_te, pers_proba),
        "trailing": (y_te, trail_proba),
        "always": (y_te, always_proba),
        "climatology": (y_te, clim_proba),
        "laghour": (y_te, laghour_proba),
        "dummy_logistic": (y_te, dummy_proba),
        "main": (y_te, proba),
    }
    from sklearn.metrics import precision_score, recall_score

    op_mask = proba >= thr_main
    op_point = (float(recall_score(y_te, op_mask, zero_division=0)),
                float(precision_score(y_te, op_mask, zero_division=0)))
    cal_gap = {k: round(float(np.mean(v[1]) - test_prev), 4) for k, v in curves.items()}

    meta = {
        "results": results,
        "data_source": data_source,
        "data_sha256": sha256_file(paths["raw"]) if data_source != "synthetic" else sha256_file(paths["synthetic"]),
        "use_text_signals": cfg.use_text_signals and text_hours > 0,
        "text_active_hours": text_hours,
        "target": cfg.target_region,
        "seed": cfg.seed,
        "horizon_h": cfg.horizon_h,
        "test_days": cfg.test_days,
        "backend": backend,
        "prevalence": {"train": round(train_prev, 4), "test": round(test_prev, 4)},
        "calibration_gap": cal_gap,
        "mean_pred_minus_prevalence": cal_gap,
        "threshold_rule": f"max precision s.t. recall>=0.85 on pooled OOF (guardrail precision>={gr:.3f})" + ("; DEGENERATE FALLBACK to 0.5" if surs_vyrodzhenyi else ""),
        "degenerate_fallback": surs_vyrodzhenyi,
        "fold_thresholds": fold_thrs,
        "threshold_stability": {
            "mean": round(float(np.mean(fold_thrs)), 4),
            "std": round(thr_std, 4),
            "min": round(float(min(fold_thrs)), 4),
            "max": round(float(max(fold_thrs)), 4),
        },
        "grid_range": [str(grid.index.min()), str(grid.index.max())],
        "split_sizes": {"train": len(X_tr), "test": len(X_te)},
        "config": cfg.model_dump(),
        "resolved_lgbm_params": cfg.resolved_lgbm_params() if backend == "lightgbm" else None,
        "resolved_hgb_params": {"random_state": cfg.seed, "early_stopping": False,
                                "max_iter": 100, "learning_rate": 0.1} if backend != "lightgbm" else None,
    }
    write_artifacts(paths, y_te, test_prev, curves, op_point, model, X_te, thr_main, meta)
    return results


def evaluate_models(
    y_te: pd.Series,
    prev,
    probas: dict[str, np.ndarray],
    thresholds: dict[str, float],
    trail30: float,
) -> dict:
    results = {}
    for k, p in probas.items():
        results[k] = compute_metrics(y_te, p, thresholds.get(k, 0.5), prev=prev, ref=trail30)
    for _binary in ("heuristic", "persistence", "always", "trailing"):
        if _binary in results:
            results[_binary]["bss"] = None
    return results


def write_artifacts(
    paths: dict[str, Path],
    y_te: pd.Series,
    test_prev: float,
    curves: dict[str, tuple],
    op_point: tuple[float, float],
    model,
    X_te: pd.DataFrame,
    thr_main: float,
    meta: dict,
) -> None:
    try:
        save_plots(curves, test_prev, op_point, str(paths["reports"]))
        save_calibration(curves, str(paths["reports"]))
    except Exception:  # noqa: BLE001 - plots must not fail metrics
        logger.exception("plot saving failed (metrics unaffected)")
    try:
        save_importance(model, X_te, y_te, str(paths["reports"] / "shap_summary.png"),
                        threshold=thr_main)
    except Exception:  # noqa: BLE001 - SHAP must not fail metrics
        logger.exception("SHAP failed (metrics unaffected)")
    try:
        save_model_bundle(model, list(X_te.columns), thr_main, str(paths["reports"] / "model.pkl"))
        if hasattr(model, "booster_"):
            model.booster_.save_model(str(paths["reports"] / "lgbm.txt"))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - model dump must not fail metrics
        logger.exception("model saving failed")
    save_metrics_json(meta, str(paths["reports"] / "metrics.json"))


def build_config(args) -> Settings:
    if args.region not in TRANSIT_REGIONS:
        raise ValueError(f"unknown --region {args.region!r}; valid: {sorted(TRANSIT_REGIONS)}")
    return Settings(
        target_region=args.region,
        neighbors=[n for n in TRANSIT_REGIONS[args.region] if n != args.region],
        use_text_signals=args.text_signals,
    )


def rolling_evaluate(cfg: Settings, args, n_windows: int = 3, window_days: int = 30) -> dict:
    """Rolling-origin: last N non-overlapping 30d windows, train strictly before each.

    Uses the same embargo logic as run() but no OOF/threshold tuning (fixed 0.5
    operating point for comparability); text signals are off for determinism.
    """
    paths = resolve_paths(cfg)
    if args.synthetic:
        df_raw, data_source = generate_synthetic(str(paths["synthetic"]), days=args.days, seed=cfg.seed), "synthetic"
    elif paths["raw"].exists():
        df_raw, data_source = load_raw(paths["raw"]), read_source(paths["raw"])
    else:
        raise FileNotFoundError("no data file (use --real/--synthetic first)")
    validate_regions(df_raw, [cfg.target_region, *cfg.neighbors], cfg.min_events_per_region)
    regions = [cfg.target_region, *[n for n in cfg.neighbors if n != cfg.target_region]]
    grid = build_hourly_grid(df_raw, regions, freq=cfg.freq)
    grid = grid.where(coverage_mask(df_raw, regions, grid.index))
    if cfg.tail_cutoff_hours:
        grid = grid.loc[: grid.index.max() - pd.Timedelta(hours=cfg.tail_cutoff_hours)]
    else:
        _dur = (df_raw["end_time"] - df_raw["start_time"]).dt.total_seconds() / 3600
        _auto = int(max(1, _dur.quantile(0.99)))
        logger.info(f"auto tail cutoff from p99 duration: {_auto}h")
        grid = grid.loc[: grid.index.max() - pd.Timedelta(hours=_auto)]
    X, y = make_features(grid, cfg.target_region, cfg.neighbors, horizon=cfg.horizon_h,
                         lags=cfg.lags, windows=cfg.windows)
    backend = detect_backend()
    sha = None
    try:
        sha = sha256_file(paths["raw"]) if data_source != "synthetic" else sha256_file(paths["synthetic"])
    except OSError:
        pass
    out = {"data_source": data_source, "backend": backend, "data_sha256": sha,
           "config": cfg.model_dump(), "windows": []}
    for i in range(n_windows):
        w_end = X.index.max() - pd.Timedelta(days=i * window_days)
        w_start = w_end - pd.Timedelta(days=window_days)
        tr_end = w_start - pd.Timedelta(hours=cfg.horizon_h)
        Xtr, Xte = X[X.index <= tr_end], X[(X.index > w_start) & (X.index <= w_end)]
        ytr, yte = y[y.index <= tr_end], y[(y.index > w_start) & (y.index <= w_end)]
        if len(Xte) == 0 or len(Xtr) == 0:
            continue
        m = train_model(Xtr, ytr, cfg, backend)
        pm = proba_full(m, Xte)
        lh = LagHourBaseline().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        row = {
            "window_end": str(w_end), "n_train": len(Xtr), "n_test": len(Xte),
            "test_prev": round(float(yte.mean()), 4),
            "main_pr_auc": round(float(safe_pr_auc(yte, pm)), 4),
            "laghour_pr_auc": round(float(safe_pr_auc(yte, lh)), 4),
        }
        logger.info(f"rolling {row}")
        out["windows"].append(row)
    if not out["windows"]:
        raise ValueError("rolling produced no windows (empty train/test splits)")
    vals = [w["main_pr_auc"] for w in out["windows"]]
    out["summary"] = {"median": round(float(np.median(vals)), 4), "min": min(vals), "max": max(vals)}
    paths["reports"].mkdir(parents=True, exist_ok=True)
    save_metrics_json(out, str(paths["reports"] / "rolling.json"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Kyiv")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--text-signals", action="store_true")
    ap.add_argument("--no-calendar", action="store_true",
                    help="ablation: drop calendar features for the main model")
    ap.add_argument("--strict-source", action="store_true",
                    help="refuse to run on data with unverified source")
    ap.add_argument("--rolling", type=int, default=0, help="rolling-origin windows (0=off)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--real", action="store_true")
    src.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = build_config(args)
    if args.rolling:
        rolling_evaluate(cfg, args, n_windows=args.rolling)
    else:
        run(cfg, args)


if __name__ == "__main__":
    main()
