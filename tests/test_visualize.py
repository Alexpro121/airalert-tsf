from pathlib import Path

import pytest

from src.visualize import FIGURES, visualize


def test_visualize_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing inputs"):
        visualize(tmp_path)


def test_visualize_creates_figures(tmp_path):
    import shutil

    src = Path(__file__).parent.parent
    for rel in ("data/processed/grid.csv", "data/raw/alerts.csv",
                "reports/metrics.json", "reports/model.pkl"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / rel, dst)
    done = visualize(tmp_path)
    for f in FIGURES:
        assert (tmp_path / "reports" / "figures" / f).exists(), f
    assert set(done) == set(FIGURES)
