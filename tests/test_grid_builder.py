import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import make_features
from src.grid_builder import build_hourly_grid


def test_grid_empty_raises():
    with pytest.raises(ValueError):
        build_hourly_grid(pd.DataFrame(columns=["region", "start_time", "end_time"]), ["Kyiv"])


def test_grid_marks_active_slot():
    raw = pd.DataFrame(
        [
            {
                "region": "Kyiv",
                "start_time": pd.Timestamp("2024-01-01 00:30", tz="UTC"),
                "end_time": pd.Timestamp("2024-01-01 01:30", tz="UTC"),
                "alert_type": "air_raid",
            }
        ]
    )
    g = build_hourly_grid(raw, ["Kyiv"])
    assert g.loc[pd.Timestamp("2024-01-01 00:00", tz="UTC"), "Kyiv"] == 1
    assert g.loc[pd.Timestamp("2024-01-01 01:00", tz="UTC"), "Kyiv"] == 1


def test_grid_empty_region_is_zero():
    raw = pd.DataFrame(
        [
            {
                "region": "Kyiv",
                "start_time": pd.Timestamp("2024-01-01 00:10", tz="UTC"),
                "end_time": pd.Timestamp("2024-01-01 00:20", tz="UTC"),
                "alert_type": "air_raid",
            }
        ]
    )
    g = build_hourly_grid(raw, ["Kyiv", "Sumy"])
    assert (g["Sumy"] == 0).all()
    assert g["Kyiv"].max() == 1


def test_grid_exact_hour_boundaries():
    raw = pd.DataFrame([
        {
            "region": "Kyiv",
            "start_time": pd.Timestamp("2024-01-01 02:00", tz="UTC"),
            "end_time": pd.Timestamp("2024-01-01 04:00", tz="UTC"),
            "alert_type": "air_raid",
        },
        {
            "region": "Kyiv",
            "start_time": pd.Timestamp("2024-01-01 06:10", tz="UTC"),
            "end_time": pd.Timestamp("2024-01-01 06:20", tz="UTC"),
            "alert_type": "air_raid",
        },
    ])
    g = build_hourly_grid(raw, ["Kyiv"])
    assert g.loc[pd.Timestamp("2024-01-01 02:00", tz="UTC"), "Kyiv"] == 1
    assert g.loc[pd.Timestamp("2024-01-01 03:00", tz="UTC"), "Kyiv"] == 1
    assert g.loc[pd.Timestamp("2024-01-01 04:00", tz="UTC"), "Kyiv"] == 0
    assert g.loc[pd.Timestamp("2024-01-01 05:00", tz="UTC"), "Kyiv"] == 0
    assert g.loc[pd.Timestamp("2024-01-01 06:00", tz="UTC"), "Kyiv"] == 1


def test_grid_alert_longer_than_24h():
    raw = pd.DataFrame([{
        "region": "Kyiv",
        "start_time": pd.Timestamp("2024-01-01 10:15", tz="UTC"),
        "end_time": pd.Timestamp("2024-01-03 11:45", tz="UTC"),
        "alert_type": "air_raid",
    }])
    g = build_hourly_grid(raw, ["Kyiv"])
    assert len(g) == 50
    assert (g["Kyiv"] == 1).all()


def test_grid_matches_naive_reference_on_random_intervals():
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2024-03-01", tz="UTC")
    rows = []
    for _ in range(30):
        s = base + pd.Timedelta(minutes=int(rng.integers(0, 3000)))
        e = s + pd.Timedelta(minutes=int(rng.integers(1, 1500)))
        rows.append({"region": "Kyiv", "start_time": s, "end_time": e, "alert_type": "air_raid"})
    raw = pd.DataFrame(rows)
    g = build_hourly_grid(raw, ["Kyiv"])
    for h in g.index:
        expected = int(((raw["start_time"] < h + pd.Timedelta(hours=1)) & (raw["end_time"] > h)).any())
        assert g.loc[h, "Kyiv"] == expected, h
    raw = pd.DataFrame([{
        "region": "Kyiv",
        "start_time": pd.Timestamp("2024-01-01 10:15", tz="UTC"),
        "end_time": pd.Timestamp("2024-01-03 11:45", tz="UTC"),
        "alert_type": "air_raid",
    }])
    g = build_hourly_grid(raw, ["Kyiv"])
    assert len(g) == 50
    assert (g["Kyiv"] == 1).all()


def test_grid_exact_full_hour_interval():
    raw = pd.DataFrame([{
        "region": "Kyiv",
        "start_time": pd.Timestamp("2024-01-01 10:00", tz="UTC"),
        "end_time": pd.Timestamp("2024-01-01 11:00", tz="UTC"),
        "alert_type": "air_raid",
    }])
    g = build_hourly_grid(raw, ["Kyiv", "Sumy"])
    assert g.loc[pd.Timestamp("2024-01-01 10:00", tz="UTC"), "Kyiv"] == 1
    assert g["Kyiv"].sum() == 1


def test_grid_cross_boundary_interval():
    raw = pd.DataFrame([{
        "region": "Kyiv",
        "start_time": pd.Timestamp("2024-01-01 10:30", tz="UTC"),
        "end_time": pd.Timestamp("2024-01-01 11:30", tz="UTC"),
        "alert_type": "air_raid",
    }])
    g = build_hourly_grid(raw, ["Kyiv"])
    assert g.loc[pd.Timestamp("2024-01-01 10:00", tz="UTC"), "Kyiv"] == 1
    assert g.loc[pd.Timestamp("2024-01-01 11:00", tz="UTC"), "Kyiv"] == 1
    assert g["Kyiv"].sum() == 2


def test_grid_rejects_naive_timestamps():
    raw = pd.DataFrame([{
        "region": "Kyiv",
        "start_time": pd.Timestamp("2024-01-01 10:00"),
        "end_time": pd.Timestamp("2024-01-01 11:00"),
        "alert_type": "air_raid",
    }])
    with pytest.raises(TypeError):
        build_hourly_grid(raw, ["Kyiv"])


def test_target_definition_h3():
    idx = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    grid = pd.DataFrame({"Kyiv": [0] * 30, "Sumy": [0] * 30}, index=idx)
    grid.index.name = "timestamp"
    grid.loc[idx[26], "Kyiv"] = 1
    _, y = make_features(grid, "Kyiv", ["Sumy"], horizon=3)
    assert y.loc[idx[24]] == 1
    assert y.loc[idx[25]] == 1
    assert y.loc[idx[26]] == 1
    assert y.loc[idx[27]] == 0
