from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from main import load_dataset
from src.config import Settings
from src.data_loader import read_source
from src.grid_builder import build_hourly_grid

pytestmark = pytest.mark.real

RAW = Path(__file__).parent.parent / "data" / "raw" / "alerts.csv"
TARGETS = ["Kyiv", "Kyiv_Oblast", "Chernihiv", "Sumy", "Poltava"]


@pytest.fixture(scope="module")
def df():
    if not RAW.exists():
        pytest.skip("no data/raw/alerts.csv")
    if read_source(RAW) != "real":
        pytest.skip(f"source is '{read_source(RAW)}', not real data")
    d = pd.read_csv(RAW, parse_dates=["start_time", "end_time"])
    d["start_time"] = pd.to_datetime(d["start_time"], utc=True)
    d["end_time"] = pd.to_datetime(d["end_time"], utc=True)
    return d


def test_real_schema_and_durations(df):
    assert {"region", "start_time", "end_time", "alert_type"} <= set(df.columns)
    assert len(df) > 1000
    assert (df["end_time"] > df["start_time"]).all()
    assert (df["alert_type"] == "air_raid").all()
    assert set(TARGETS) <= set(df["region"].unique())


def test_real_timestamps_carry_utc_offset(df):
    s = df["start_time"].astype(str)
    assert s.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$").all()


def test_night_peak_in_kyiv_local_time(df):
    g = build_hourly_grid(df[df["region"].isin(TARGETS)], TARGETS)
    loc = g.index.tz_convert("Europe/Kyiv")
    r = g["Kyiv"].groupby(loc.hour).mean()
    assert r.loc[[23, 0, 1, 2, 3, 4, 5]].mean() > r.loc[7:22].mean()


def test_no_silent_synthetic_fallback(tmp_path):
    cfg = Settings(raw_path=str(tmp_path / "missing.csv"))
    paths = {"raw": tmp_path / "missing.csv", "synthetic": tmp_path / "s.csv"}
    with pytest.raises(FileNotFoundError, match="--synthetic"):
        load_dataset(cfg, paths, SimpleNamespace(synthetic=False, real=False))


def test_real_grid_coverage(df):
    sample = df[df["region"].isin(TARGETS)].sample(20, random_state=1)
    g = build_hourly_grid(df[df["region"].isin(TARGETS)], TARGETS)
    for _, r in sample.iterrows():
        s, e = r["start_time"], r["end_time"]
        hrs = pd.date_range(s.floor("h"), e.ceil("h"), freq="1h", tz="UTC")
        slot = [h for h in hrs if h < e and h + pd.Timedelta(hours=1) > s]
        slot_idx = g.index.intersection(slot)
        assert len(slot_idx) > 0
        assert (g.loc[slot_idx, r["region"]] == 1).all()
