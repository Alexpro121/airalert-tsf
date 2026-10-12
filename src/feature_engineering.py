"""Features are causal w.r.t. forecast moment T = start of hour t.

At wall time T only alert states A[tau] for tau <= t-1 are known (hour t has
just started). Features use shift(>=1); target covers the next H full hours
starting with the current one: Y[t] = max(A[t..t+H-1]). No hour is skipped.
"""

import numpy as np
import pandas as pd

from src.config import KYIV_TZ

HISTORY_CAP = 168.0

# Threat-approach paths: upstream regions ordered far -> Kyiv (target excluded).
# A wave like Sumy -> Chernihiv -> Kyiv_Oblast shows up as shrinking distance.
PATHS: dict[str, list[str]] = {
    "east": ["Kharkiv_Oblast", "Sumy", "Chernihiv", "Kyiv_Oblast"],
    "northeast": ["Sumy", "Chernihiv", "Kyiv_Oblast"],
    "southeast": ["Poltava", "Cherkasy_Oblast", "Kyiv_Oblast"],
    "south": ["Cherkasy_Oblast", "Kyiv_Oblast"],
    "west": ["Zhytomyr_Oblast", "Kyiv_Oblast"],
    "southwest": ["Vinnytsia_Oblast", "Kyiv_Oblast"],
}


def _check_hourly_regular(index: pd.DatetimeIndex) -> None:
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("grid index must be monotonic and unique")
    step = index.to_series().diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta(hours=1):
        raise ValueError(f"grid must be exactly 1h regular, got steps={step}")


def make_features(
    grid: pd.DataFrame,
    target: str,
    neighbors: list[str],
    horizon: int = 3,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    events: pd.Series | None = None,
    all_clear: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    lags = lags or [1, 2, 3, 6, 24]
    windows = windows or [3, 6, 12, 24]
    _check_hourly_regular(grid.index)
    df = grid.copy()
    a = df[target]
    fut = pd.concat([a.shift(-h) for h in range(horizon)], axis=1)
    unknown_future = fut.isna().any(axis=1)
    y = (fut.max(axis=1) == 1).astype(float)
    y[unknown_future] = np.nan
    y.name = "target"
    feats: dict[str, pd.Series] = {}
    local = df.index.tz_convert(KYIV_TZ)
    h = local.hour
    dow = local.dayofweek
    feats["hour_sin"] = np.sin(2 * np.pi * h / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * h / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["is_night"] = ((h >= 23) | (h < 6)).astype(int)
    for L in lags:
        feats[f"lag_{L}"] = a.shift(L)
    for w in windows:
        feats[f"roll_sum_{w}h"] = a.shift(1).rolling(w, min_periods=w).sum()
    feats["trail_168h"] = a.shift(1).rolling(168, min_periods=1).mean()
    feats["trail_720h"] = a.shift(1).rolling(720, min_periods=1).mean()
    s = a.shift(1)
    sok = s.notna()
    seen = ((s == 1) & sok).cummax()
    grp = ((s == 1) & sok).cumsum()
    cnt = s.groupby(grp).cumcount()
    feats["hours_since_last"] = (
        cnt.where(s == 0, 0.0).astype(float).where(seen & sok, HISTORY_CAP).clip(upper=HISTORY_CAP)
    )
    feats["hours_since_last"] = feats["hours_since_last"].where(sok, np.nan)
    feats["has_history"] = (seen & sok).astype("int8")
    nbr_cols = [c for c in neighbors if c in df.columns and c != target]
    for n in nbr_cols:
        feats[f"nbr_{n}"] = df[n].shift(1)
        feats[f"nbr_{n}_lag2"] = df[n].shift(2)
        feats[f"nbr_{n}_sum3"] = df[n].shift(1).rolling(3, min_periods=3).sum()
    if nbr_cols:
        started = pd.concat([(df[n].shift(1) == 1) & (df[n].shift(2) == 0) for n in nbr_cols], axis=1)
        feats["nbr_new_onset"] = started.any(axis=1).fillna(False).astype("int8")
    else:
        feats["nbr_new_onset"] = 0
    feats["country_ratio"] = (
        df[nbr_cols].shift(1).mean(axis=1) if nbr_cols else pd.Series(0.0, index=df.index)
    )
    dists = []
    for pname, nodes in PATHS.items():
        present = [n for n in nodes if n in df.columns]
        if not present:
            continue
        sub = df[present].shift(1)
        # distance from Kyiv: nearest upstream node = 1
        dist = pd.Series(len(present) + 1, index=df.index)
        for pos, n in enumerate(reversed(present), start=1):
            dist = dist.where(sub[n] != 1, pos)
        dist = dist.where(sub.notna().all(axis=1), np.nan)
        feats[f"path_{pname}_dist"] = dist.astype(float)
        feats[f"path_{pname}_count"] = sub.sum(axis=1)
        dists.append(feats[f"path_{pname}_dist"])
    if dists:
        feats["min_threat_dist"] = pd.concat(dists, axis=1).min(axis=1)
    for col, src in (("has_threat_mention", events), ("has_all_clear", all_clear)):
        if src is None:
            continue
        ev = src.reindex(df.index, fill_value=0)
        feats[col] = pd.Series(ev, index=df.index).shift(1).fillna(0).astype("int8")
    X = pd.DataFrame(feats, index=df.index)
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid].astype(int)
