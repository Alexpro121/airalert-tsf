import numpy as np
import pytest

from src.evaluate import (
    block_bootstrap_delta_pr_auc,
    block_bootstrap_pr_auc,
    brier_skill_score,
)


def test_brier_skill_score_manual():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]
    ref = sum(y) / len(y)
    manual = 1 - (sum((a - b) ** 2 for a, b in zip(y, p)) / len(y)) / (
        sum((a - ref) ** 2 for a in y) / len(y))
    assert brier_skill_score(y, p, ref=ref) == pytest.approx(manual)


def test_bootstrap_point_estimate_known_case():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 60)
    p = np.where(y == 1, 0.9, 0.1) + rng.normal(0, 0.01, 120)
    out = block_bootstrap_pr_auc(y, p, block=24, n_boot=50, seed=1)
    assert out["point"] > 0.95
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["n_boot"] > 0


def test_delta_known_case():
    y = np.array([0, 1] * 60)
    pa = np.where(y == 1, 0.9, 0.1).astype(float)
    pb = np.full(120, 0.5)
    out = block_bootstrap_delta_pr_auc(y, pa, pb, block=24, n_boot=50, seed=1)
    assert out["point"] > 0.4
    assert out["lo"] > 0


def test_block_indices_length():
    from src.evaluate import _block_indices

    rng = np.random.default_rng(0)
    assert len(_block_indices(50, 24, rng)) == 50
    assert len(_block_indices(720, 24, rng)) == 720
