import numpy as np
import pandas as pd
from loguru import logger


def top_episode_waterfall(
    X: pd.DataFrame, y: pd.Series, model: object, summary_path: str, threshold: float = 0.5
) -> list[str]:
    """Waterfalls for one TP, one FP and one FN (with true labels in filenames)."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    proba = np.asarray(model.predict_proba(X)[:, 1])  # type: ignore[attr-defined]
    pred = (proba >= threshold).astype(int)
    yt = np.asarray(y)
    picks = {}
    tp = np.where((pred == 1) & (yt == 1))[0]
    fp = np.where((pred == 1) & (yt == 0))[0]
    fn = np.where((pred == 0) & (yt == 1))[0]
    if len(tp):
        picks[f"tp_{X.index[tp[0]].strftime('%Y%m%dT%H')}"] = tp[0]
    if len(fp):
        picks[f"fp_{X.index[fp[0]].strftime('%Y%m%dT%H')}"] = fp[0]
    if len(fn):
        picks[f"fn_{X.index[fn[0]].strftime('%Y%m%dT%H')}"] = fn[0]
    if not picks:
        picks[f"top_{X.index[int(np.argmax(proba))].strftime('%Y%m%dT%H')}"] = int(np.argmax(proba))
    explainer = shap.TreeExplainer(model)
    out_paths = []
    for tag, k in picks.items():
        sv = explainer.shap_values(X.iloc[[k]])
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1] if len(base) == 2 else base[0]
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=np.asarray(sv).ravel(),
                base_values=float(np.asarray(base).ravel()[0]),
                data=X.iloc[k].values,
                feature_names=list(X.columns),
            ),
            show=False,
        )
        out = str(Path(summary_path).parent / f"shap_waterfall_{tag}.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        out_paths.append(out)
    return out_paths


def save_importance(
    model: object, X: pd.DataFrame, y: pd.Series, out_path: str, threshold: float = 0.5
) -> None:
    from pathlib import Path

    import matplotlib

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap_ok, balanced_ok, waterfall_ok = False, False, False
    supports_tree_shap = hasattr(model, "booster_") or type(model).__name__ in (
        "LGBMClassifier", "XGBClassifier", "CatBoostClassifier", "RandomForestClassifier",
    )
    if not supports_tree_shap:
        logger.info(f"{type(model).__name__} has no TreeExplainer support -> permutation importance")
    else:
        try:
            import shap

            rng = np.random.default_rng(42)
            natural_idx = rng.choice(X.index, size=min(500, len(X)), replace=False)
            explainer = shap.TreeExplainer(model)

            def _vals(sample: pd.DataFrame):
                v = explainer.shap_values(sample)
                if isinstance(v, list) and len(v) == 2:
                    return v[1]
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    return v[:, :, 1]
                return v

            sample = X.loc[natural_idx]
            shap.summary_plot(_vals(sample), sample, show=False, max_display=15)
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            shap_ok = True
        except Exception as e:  # noqa: BLE001 - summary isolated
            plt.close("all")
            logger.warning(f"SHAP summary failed ({e})")
    if shap_ok:
        try:
            import shap

            n0, n1 = y[y == 0].index, y[y == 1].index
            k = min(250, len(n0), len(n1))
            rng = np.random.default_rng(42)
            bal_idx = list(rng.choice(n0, k, replace=False)) + list(rng.choice(n1, k, replace=False))
            bal_sample = X.loc[bal_idx]
            explainer = shap.TreeExplainer(model)
            v = explainer.shap_values(bal_sample)
            if isinstance(v, list) and len(v) == 2:
                v = v[1]
            elif isinstance(v, np.ndarray) and v.ndim == 3:
                v = v[:, :, 1]
            shap.summary_plot(v, bal_sample, show=False, max_display=15)
            bal_path = str(out_path).replace(".png", "_balanced.png")
            plt.savefig(bal_path, dpi=120, bbox_inches="tight")
            plt.close()
            balanced_ok = True
        except Exception as e:  # noqa: BLE001 - balanced isolated
            plt.close("all")
            logger.warning(f"SHAP balanced failed ({e})")
    if shap_ok:
        try:
            wfs = top_episode_waterfall(X, y, model, out_path, threshold=threshold)
            waterfall_ok = len(wfs) > 0
        except Exception as e:  # noqa: BLE001 - waterfall isolated
            plt.close("all")
            logger.warning(f"SHAP waterfall failed ({e})")
    if shap_ok:
        logger.info(f"SHAP saved -> {out_path} (balanced={balanced_ok}, waterfall={waterfall_ok})")
        return
    fb_path = str(out_path).replace(".png", "_fallback.png")
    logger.warning(f"SHAP failed, fallback -> {fb_path}")
    plt.figure(figsize=(8, 6))
    if hasattr(model, "booster_"):
        gain = model.booster_.feature_importance(importance_type="gain")  # type: ignore[attr-defined]
        imp = pd.Series(gain, index=X.columns).sort_values(ascending=False)
        imp.head(15)[::-1].plot.barh(title="Top 15 importances (gain)")
    elif hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)  # type: ignore[attr-defined]
        imp.head(15)[::-1].plot.barh(title="Top 15 importances (split)")
    else:
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(42)
        sub = rng.choice(X.index, size=min(2000, len(X)), replace=False)
        r = permutation_importance(
            model, X.loc[sub], y.loc[sub], n_repeats=3, random_state=42,
            scoring="average_precision",
        )
        imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
        imp.head(15)[::-1].plot.barh(title="Top 15 permutation importances")
    plt.tight_layout()
    plt.savefig(fb_path, dpi=120)
    plt.close()
