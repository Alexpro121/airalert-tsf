
from src.data_loader import fetch_real_alerts

VOL = ("region,started_at,finished_at,naive\n"
       "Kyiv City,2024-06-01 10:10:00+00:00,2024-06-01 11:20:00+00:00,False\n"
       "Sumska oblast,2024-06-01 12:00:00+00:00,2024-06-01 12:30:00+00:00,False\n")


def test_fetch_success(monkeypatch, tmp_path):
    import src.data_loader as dl

    monkeypatch.setattr(dl, "_download", lambda url, timeout: VOL.encode())
    df = fetch_real_alerts(tmp_path / "a.csv", required=["Kyiv", "Sumy"])
    assert set(df["region"]) == {"Kyiv", "Sumy"}
    assert (tmp_path / "a.csv.meta.json").exists()


def test_fetch_official_fallback(monkeypatch, tmp_path):
    import urllib.error

    import src.data_loader as dl

    def fake(url, timeout):
        if "volunteer" in url:
            raise urllib.error.URLError("down")
        return (b"oblast,raion,hromada,level,started_at,finished_at,source\n"
                b"Kyivska oblast,,,oblast,2024-06-01 10:10:00+00:00,"
                b"2024-06-01 11:20:00+00:00,official\n")

    monkeypatch.setattr(dl, "_download", fake)
    df = fetch_real_alerts(tmp_path / "a.csv", required=["Kyiv_Oblast"])
    assert (df["region"] == "Kyiv_Oblast").all()


def test_fetch_missing_region_raises(monkeypatch, tmp_path):
    import src.data_loader as dl

    monkeypatch.setattr(dl, "_download", lambda url, timeout: VOL.encode())
    try:
        fetch_real_alerts(tmp_path / "a.csv", required=["Lviv"])
    except ValueError as e:
        assert "zero events" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_empty_raises(monkeypatch, tmp_path):
    import src.data_loader as dl

    monkeypatch.setattr(dl, "_download",
                        lambda url, timeout: b"region,started_at,finished_at\n")
    try:
        fetch_real_alerts(tmp_path / "a.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
