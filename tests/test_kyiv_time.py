import numpy as np
import pandas as pd

from src.data_loader import load_raw
from src.feature_engineering import make_features


def test_dst_summer_winter_offset(tmp_path):
    raw = pd.DataFrame([
        {"region": "Kyiv", "start_time": "2024-07-01 12:10:00", "end_time": "2024-07-01 12:20:00",
         "alert_type": "air_raid"},
        {"region": "Kyiv", "start_time": "2024-01-01 12:10:00", "end_time": "2024-01-01 12:20:00",
         "alert_type": "air_raid"},
    ])
    p = tmp_path / "r.csv"
    raw.to_csv(p, index=False)
    df = load_raw(p)
    assert str(df["start_time"].dt.tz) == "UTC"
    assert df["start_time"].iloc[0] == pd.Timestamp("2024-07-01 09:10:00", tz="UTC")
    assert df["start_time"].iloc[1] == pd.Timestamp("2024-01-01 10:10:00", tz="UTC")


def test_ambiguous_dst_hour_resolved_to_first_occurrence(tmp_path):
    raw = pd.DataFrame([{
        "region": "Kyiv", "start_time": "2024-10-27 03:30:00", "end_time": "2024-10-27 03:40:00",
        "alert_type": "air_raid",
    }])
    p = tmp_path / "r.csv"
    raw.to_csv(p, index=False)
    df = load_raw(p)
    assert len(df) == 1
    assert df["start_time"].iloc[0] == pd.Timestamp("2024-10-27 00:30:00", tz="UTC")


def test_dst_fold_end_resolved_to_second_occurrence(tmp_path):
    raw = pd.DataFrame([{
        "region": "Kyiv", "start_time": "2024-10-27 03:50:00", "end_time": "2024-10-27 03:10:00",
        "alert_type": "air_raid",
    }])
    p = tmp_path / "r.csv"
    raw.to_csv(p, index=False)
    df = load_raw(p)
    assert len(df) == 1
    assert df["start_time"].iloc[0] == pd.Timestamp("2024-10-27 00:50:00", tz="UTC")
    assert df["end_time"].iloc[0] == pd.Timestamp("2024-10-27 01:10:00", tz="UTC")


def test_mixed_offset_and_naive_strings(tmp_path):
    raw = pd.DataFrame([
        {"region": "Kyiv", "start_time": "2024-06-01 12:10:00+03:00",
         "end_time": "2024-06-01 12:20:00+03:00", "alert_type": "air_raid"},
        {"region": "Kyiv", "start_time": "2024-06-01 12:10:00",
         "end_time": "2024-06-01 12:20:00", "alert_type": "air_raid"},
    ])
    p = tmp_path / "r.csv"
    raw.to_csv(p, index=False)
    df = load_raw(p)
    assert len(df) == 2
    assert (df["start_time"] == pd.Timestamp("2024-06-01 09:10:00", tz="UTC")).all()


def test_same_wall_clock_alert_and_message(tmp_path):
    from src.grid_builder import build_hourly_grid
    from src.text_signals import build_hourly_flags, load_events

    wall = "2024-06-01 12:10:00"
    raw = pd.DataFrame([{"region": "Kyiv", "start_time": wall, "end_time": "2024-06-01 12:20:00",
                         "alert_type": "air_raid"}])
    rp = tmp_path / "r.csv"
    raw.to_csv(rp, index=False)
    mp = tmp_path / "m.csv"
    pd.DataFrame([{"timestamp": wall, "text": "Шахед"}]).to_csv(mp, index=False)
    g = build_hourly_grid(load_raw(rp), ["Kyiv"])
    flags = build_hourly_flags(load_events(mp), g.index)
    assert g.loc[pd.Timestamp("2024-06-01 09:00", tz="UTC"), "Kyiv"] == 1
    assert flags.loc[pd.Timestamp("2024-06-01 09:00", tz="UTC")] == 1


def test_calendar_uses_kyiv_time():
    idx = pd.date_range("2024-06-01", periods=240, freq="1h", tz="UTC")
    grid = pd.DataFrame({"Kyiv": [0] * 240, "Sumy": [0] * 240}, index=idx)
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=1, lags=[1], windows=[6])
    noon_kyiv = X.loc[pd.Timestamp("2024-06-05 09:00:00", tz="UTC")]
    assert abs(noon_kyiv["hour_sin"] - 0.0) < 1e-9
    assert noon_kyiv["is_night"] == 0
    midnight_kyiv = X.loc[pd.Timestamp("2024-06-05 21:00:00", tz="UTC")]
    assert abs(midnight_kyiv["hour_cos"] - np.cos(2 * np.pi * 0 / 24)) < 1e-9
    assert midnight_kyiv["is_night"] == 1
