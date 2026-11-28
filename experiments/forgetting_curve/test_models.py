import numpy as np
import pytest

from experiments.forgetting_curve.data import split_of
from experiments.forgetting_curve.models import (HalfLifeRegression, Leitner, auc, bootstrap_mae_difference, features,
                                                 target_half_life)


def synthetic(n=20_000, theta=(1.5, -1.0, -1.0), seed=0):
    """Reviews generated from a known HLR model, with binomial noise like real sessions."""
    rng = np.random.default_rng(seed)
    right, wrong = rng.integers(0, 12, n), rng.integers(0, 6, n)
    delta = rng.exponential(3.0, n)
    h = 2.0 ** (features(right, wrong) @ np.array(theta))
    p = rng.binomial(4, 2.0 ** (-delta / h)) / 4
    return {"p": p, "delta_days": delta, "right": right, "wrong": wrong,
            "user": np.array([f"u{i % 200}" for i in range(n)]), "lexeme": np.array(["w"] * n)}


def test_gradient_matches_finite_differences():
    data = synthetic(500)
    model = HalfLifeRegression(alpha=0.01, lam=0.001)
    X = features(data["right"], data["wrong"])
    h = target_half_life(data["p"], data["delta_days"])
    theta = np.array([0.4, -0.3, 0.2])
    _, grad = model.loss_and_gradient(theta, X, data["p"], data["delta_days"], h)
    for i in range(3):
        step = np.zeros(3)
        step[i] = 1e-6
        up = model.loss_and_gradient(theta + step, X, data["p"], data["delta_days"], h)[0]
        down = model.loss_and_gradient(theta - step, X, data["p"], data["delta_days"], h)[0]
        assert grad[i] == pytest.approx((up - down) / 2e-6, rel=1e-3, abs=1e-6)


def test_hlr_learns_more_correct_answers_mean_longer_memory_and_beats_leitner():
    train, test = synthetic(seed=1), synthetic(seed=2)
    model = HalfLifeRegression(alpha=0.0, steps=600).fit(train)
    assert model.theta[0] > 0 > model.theta[1]  # right answers lengthen, wrong answers shorten the half-life
    hlr_mae = np.mean(np.abs(test["p"] - model.predict(test)[0]))
    leitner_mae = np.mean(np.abs(test["p"] - Leitner().predict(test)[0]))
    assert hlr_mae < leitner_mae


def test_auc_handles_ties_and_extremes():
    assert auc([0, 0, 1, 1], np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc([0, 1, 0, 1], np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(auc([1, 1], np.array([0.2, 0.3])))


def test_learner_split_is_deterministic_and_bootstrap_is_paired():
    assert split_of("u:abc") == split_of("u:abc")
    users = np.array(["a", "a", "b", "c"])
    result = bootstrap_mae_difference(users, np.array([0.1, 0.1, 0.2, 0.3]), np.array([0.1, 0.1, 0.2, 0.3]), 50)
    assert result["difference"] == 0 and result["ci95"] == [0, 0]
