import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from main import run
from src.config import Settings

REPO_RAW = Path(__file__).parent.parent / "data" / "raw" / "alerts.csv"


def _hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cfg(tmp_path: Path) -> Settings:
    return Settings(
        raw_path=str(tmp_path / "alerts.csv"),
        synthetic_path=str(tmp_path / "synth.csv"),
        grid_path=str(tmp_path / "grid.csv"),
        report_dir=str(tmp_path / "reports"),
        airforce_path=str(tmp_path / "msgs.csv"),
        test_days=10,
        n_splits=2,
    )


def _args(days: int = 60) -> SimpleNamespace:
    return SimpleNamespace(synthetic=True, real=False, days=days, text_signals=False)


def _write_msgs(tmp_path: Path, grid_end: str = "2026-10-09 22:10:00+00:00") -> None:
    import pandas as pd

    pd.DataFrame([
        {"timestamp": "2026-09-01 03:10:00+00:00", "text": "Зліт МіГ-31К, чекайте тривогу"},
        {"timestamp": grid_end, "text": "Шахеди курсом на Київ"},
    ]).to_csv(tmp_path / "msgs.csv", index=False)


@pytest.mark.parametrize("backend", ["lightgbm", "hgb"])
def test_e2e_run_synthetic(tmp_path, monkeypatch, backend):
    import main as main_mod

    monkeypatch.setattr(main_mod, "detect_backend", lambda: backend)
    before = _hash(REPO_RAW) if REPO_RAW.exists() else None
    results = run(_cfg(tmp_path), _args())
    assert {"heuristic", "persistence", "always", "climatology", "laghour",
            "dummy_logistic", "main"} <= set(results)
    meta = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert meta["data_source"] == "synthetic"
    assert meta["use_text_signals"] is False
    assert meta["prevalence"]["train"] > 0
    assert "calibration_gap" in meta
    assert abs(meta["calibration_gap"]["main"]) < 1.0
    main_m = meta["results"]["main"]
    assert "pr_auc_delta_vs_persistence" in main_m
    assert "pr_auc_delta_vs_laghour" in main_m
    assert main_m["pr_auc_delta_vs_laghour"]["n_boot"] > 0
    assert 0.02 <= main_m["threshold"] <= 0.9
    assert "recall>=0.85" in meta["threshold_rule"]
    for f in ("pr_curve.png", "calibration.png", "model.pkl"):
        assert (tmp_path / "reports" / f).exists(), f
    if (tmp_path / "reports" / "shap_summary_fallback.png").exists():
        pass
    else:
        assert (tmp_path / "reports" / "shap_summary.png").exists()
        assert len(list((tmp_path / "reports").glob("shap_waterfall_*.png"))) >= 1
    assert (tmp_path / "synth.csv").exists()
    if before is not None:
        assert _hash(REPO_RAW) == before


def test_e2e_run_with_text_signals(tmp_path, monkeypatch):
    import main as main_mod
    from src.config import Settings

    monkeypatch.setattr(main_mod, "detect_backend", lambda: "lightgbm")
    _write_msgs(tmp_path)
    cfg = Settings(
        raw_path=str(tmp_path / "alerts.csv"),
        synthetic_path=str(tmp_path / "synth.csv"),
        grid_path=str(tmp_path / "grid.csv"),
        report_dir=str(tmp_path / "reports"),
        airforce_path=str(tmp_path / "msgs.csv"),
        use_text_signals=True,
        test_days=10,
        n_splits=2,
    )
    args = SimpleNamespace(synthetic=True, real=False, days=60, text_signals=True)
    results = run(cfg, args)
    meta = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert meta["use_text_signals"] is True
    assert meta["text_active_hours"] > 0
    assert "main" in results


def test_e2e_no_calendar_keeps_baselines(tmp_path, monkeypatch):
    import main as main_mod

    monkeypatch.setattr(main_mod, "detect_backend", lambda: "hgb")
    cfg = _cfg(tmp_path)
    args = SimpleNamespace(synthetic=True, real=False, days=60, text_signals=False,
                           no_calendar=True)
    results = run(cfg, args)
    assert {"dummy_logistic", "laghour", "main"} <= set(results)
    meta = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert meta["results"]["dummy_logistic"]["pr_auc"] > 0
    assert meta["results"]["main"]["pr_auc"] > 0


def test_unknown_region_raises():
    from types import SimpleNamespace

    from main import build_config

    with pytest.raises(ValueError, match="valid"):
        build_config(SimpleNamespace(region="Lviv", text_signals=False))
