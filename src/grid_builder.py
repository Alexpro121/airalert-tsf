import numpy as np
import pandas as pd
from loguru import logger

MAX_SPAN_H = 6 * 366 * 24


def _as_utc(s: pd.Series, col: str) -> pd.Series:
    s = pd.to_datetime(s, utc=False, errors="coerce")
    if s.isna().all():
        raise ValueError(f"{col}: No valid timestamps (all NaT)")
    if s.dt.tz is None:
        raise TypeError(f"{col} must be tz-aware (naive timestamps rejected)")
    return s.dt.tz_convert("UTC")


def coverage_mask(df_raw: pd.DataFrame, regions: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """True where a region is under observation: [first start, last end]."""
    mask = pd.DataFrame(False, index=index, columns=regions)
    for r in regions:
        sub = df_raw[df_raw["region"] == r]
        if sub.empty:
            continue
        lo = sub["start_time"].min().floor("h")
        hi = sub["end_time"].max().ceil("h")
        mask.loc[(mask.index >= lo) & (mask.index < hi), r] = True
    return mask


def build_hourly_grid(
    df_raw: pd.DataFrame, regions: list[str], freq: str = "1h"
) -> pd.DataFrame:
    if df_raw.empty:
        raise ValueError("Raw dataframe is empty")
    if df_raw["start_time"].isna().all() or df_raw["end_time"].isna().all():
        raise ValueError("No valid timestamps (all NaT)")
    starts = _as_utc(df_raw["start_time"], "start_time")
    ends = _as_utc(df_raw["end_time"], "end_time")
    in_scope = df_raw["region"].isin(regions)
    if not in_scope.any():
        raise ValueError(f"no events for requested regions {regions}")
    valid_dur = (ends >= starts).fillna(False)
    n_bad = int((~valid_dur).sum())
    if n_bad:
        logger.warning(f"dropping {n_bad} rows with end<start or NaT duration")
    starts = starts[valid_dur]
    ends = ends[valid_dur]
    in_scope = in_scope[valid_dur]
    if not in_scope.any():
        raise ValueError(f"no events for requested regions {regions} after dropping bad durations")
    start = starts[in_scope].min().floor(freq)
    end = ends[in_scope].max().ceil(freq)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("No valid timestamps (all NaT)")
    idx = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    if len(idx) == 0:
        span_h = 0
    else:
        span_h = (idx.max() - idx.min()).total_seconds() / 3600
    if span_h > MAX_SPAN_H:
        raise ValueError(f"grid span {span_h / 24:.0f}d exceeds guard ({MAX_SPAN_H / 24:.0f}d)")
    n = len(idx)
    grid = pd.DataFrame(index=idx)
    grid.index.name = "timestamp"
    # idx is 1h-regular by construction; positions below are hour slots
    base = idx.values.astype("datetime64[s]").astype(np.int64)
    df_garna = df_raw[valid_dur]
    for r in regions:
        sub = df_garna.loc[df_garna["region"] == r, ["start_time", "end_time"]]
        if sub.empty:
            logger.warning(f"region {r!r} has 0 events -> all-zero column")
            grid[r] = np.zeros(n, dtype="int8")
            continue
        s = _as_utc(sub["start_time"], "start_time").values.astype("datetime64[s]").astype(np.int64)
        e = _as_utc(sub["end_time"], "end_time").values.astype("datetime64[s]").astype(np.int64)
        i0 = np.clip(np.searchsorted(base, s, side="right") - 1, 0, n)
        i1 = np.clip(np.searchsorted(base, e, side="left"), 0, n)
        diff = np.zeros(n + 1, dtype=np.int32)
        maska_garna = i0 < i1
        np.add.at(diff, i0[maska_garna], 1)
        np.add.at(diff, i1[maska_garna], -1)
        grid[r] = (np.cumsum(diff)[:-1] > 0).astype("int8")
    logger.info("grid: {}h x {} regions", n, len(regions))
    return grid
