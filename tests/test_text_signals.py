import numpy as np
import pandas as pd

from src.data_loader import normalize_region
from src.evaluate import safe_roc_auc
from src.feature_engineering import make_features
from src.text_signals import ALL_CLEAR_KEYWORDS, build_hourly_flags, load_events


def test_ukrainian_inflections_match():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    words = ["ракети", "шахедів", "дронами", "балістичних", "балістика",
             "балістики", "МіГ-31К", "Ту-95МС"]
    for w in words:
        ev = pd.DataFrame([{"timestamp": idx[30], "text": f"Увага, {w} в повітрі"}])
        assert build_hourly_flags(ev, idx).loc[idx[30]] == 1, w


def test_false_positives_rejected():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    for txt in ["ескадрон пролетів", "ескадрони на параді", "кондрон",
                "цивільний літак сів", "дрохва в полі",
                "синдром втоми", "андронний колайдер"]:
        ev = pd.DataFrame([{"timestamp": idx[30], "text": txt}])
        assert build_hourly_flags(ev, idx).loc[idx[30]] == 0, txt


def test_latin_homoglyph_mig_matches():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    ev = pd.DataFrame([{"timestamp": idx[30], "text": "Зліт МiГ-31К"}])
    assert build_hourly_flags(ev, idx).loc[idx[30]] == 1


def test_keywords_hit_and_shift():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    ev = pd.DataFrame([{"timestamp": idx[30], "text": "Зліт МіГ-31К, ракета Кинджал"}])
    flags = build_hourly_flags(ev, idx)
    assert flags.loc[idx[30]] == 1
    assert flags.loc[idx[29]] == 0
    grid = pd.DataFrame(
        {"Kyiv": np.zeros(50, dtype=int), "Sumy": np.zeros(50, dtype=int)}, index=idx
    )
    X, _ = make_features(grid, "Kyiv", ["Sumy"], horizon=3, events=flags)
    assert X["has_threat_mention"].loc[idx[31]] == 1
    assert X["has_threat_mention"].loc[idx[30]] == 0


def test_nan_text_does_not_crash():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    ev = pd.DataFrame(
        [{"timestamp": idx[10], "text": None}, {"timestamp": idx[20], "text": "Шахед"}]
    )
    flags = build_hourly_flags(ev, idx)
    assert flags.loc[idx[10]] == 0
    assert flags.loc[idx[20]] == 1


def test_all_clear_separated_from_threat():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    ev = pd.DataFrame([{"timestamp": idx[30], "text": "Відбій тривоги"}])
    assert build_hourly_flags(ev, idx).loc[idx[30]] == 0
    assert build_hourly_flags(ev, idx, keywords=ALL_CLEAR_KEYWORDS).loc[idx[30]] == 1
    grid = pd.DataFrame({"Kyiv": np.zeros(50, dtype=int)}, index=idx)
    X, _ = make_features(
        grid,
        "Kyiv",
        [],
        horizon=3,
        events=build_hourly_flags(ev, idx),
        all_clear=build_hourly_flags(ev, idx, keywords=ALL_CLEAR_KEYWORDS),
    )
    assert X["has_threat_mention"].loc[idx[31]] == 0
    assert X["has_all_clear"].loc[idx[31]] == 1


def test_all_clear_with_threat_words_excluded():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    ev = pd.DataFrame([{"timestamp": idx[30], "text": "Відбій ракетної небезпеки"}])
    assert build_hourly_flags(ev, idx, exclude_keywords=ALL_CLEAR_KEYWORDS).loc[idx[30]] == 0
    assert build_hourly_flags(ev, idx, keywords=ALL_CLEAR_KEYWORDS).loc[idx[30]] == 1


def test_mixed_message_keeps_threat_clause():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    ev = pd.DataFrame([{"timestamp": idx[30],
                        "text": "Відбій на Київщині. Нова група БпЛА на Чернігівщині"}])
    assert build_hourly_flags(ev, idx, exclude_keywords=ALL_CLEAR_KEYWORDS).loc[idx[30]] == 1


def test_missing_events_file_gives_zeros():
    assert load_events("no/such/file.csv").empty
    idx = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    flags = build_hourly_flags(load_events("no/such/file.csv"), idx)
    assert (flags == 0).all()
    grid = pd.DataFrame({"Kyiv": np.zeros(50, dtype=int)}, index=idx)
    X, _ = make_features(grid, "Kyiv", [], horizon=3)
    assert "has_threat_mention" not in X.columns
    assert "has_all_clear" not in X.columns


def test_safe_roc_single_class():
    assert np.isnan(safe_roc_auc(np.zeros(10), np.linspace(0, 1, 10)))
    assert safe_roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) > 0.9


def test_normalize_region():
    assert normalize_region("Kyiv City") == "Kyiv"
    assert normalize_region("Kyivska oblast") == "Kyiv_Oblast"
    assert normalize_region("Sumska oblast") == "Sumy"
    assert normalize_region("Chernihivska oblast") == "Chernihiv"
    assert normalize_region("Kharkiv_Oblast") == "Kharkiv_Oblast"
    assert normalize_region("Zhytomyr_Oblast") == "Zhytomyr_Oblast"
    assert normalize_region("Cherkasy_Oblast") == "Cherkasy_Oblast"
    assert normalize_region("Vinnytsia_Oblast") == "Vinnytsia_Oblast"


def test_kyiv_transit_has_eight_corridors():
    import pandas as pd

    from src.config import TRANSIT_REGIONS
    from src.feature_engineering import make_features

    nbrs = TRANSIT_REGIONS["Kyiv"]
    assert len(nbrs) == 8
    idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
    grid = pd.DataFrame({r: [0] * 60 for r in ["Kyiv", *nbrs]}, index=idx)
    X, _ = make_features(grid, "Kyiv", nbrs, horizon=3, lags=[1, 2], windows=[6])
    for n in nbrs:
        assert f"nbr_{n}" in X.columns
        assert f"nbr_{n}_lag2" in X.columns
