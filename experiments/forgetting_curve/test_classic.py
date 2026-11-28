import numpy as np
import pytest

pytest.importorskip("sklearn", reason="Optional research extra is not installed")

from experiments.forgetting_curve import classic  # noqa: E402


def synthetic(n=6000, seed=1):
    """Sessions where recall truly falls with mistakes and time, as in the real data."""
    rng = np.random.default_rng(seed)
    right, wrong = rng.integers(0, 8, n), rng.integers(0, 6, n)
    days = rng.exponential(3, n)
    z = 0.3 * np.sqrt(1 + right) - 0.8 * np.sqrt(1 + wrong) + 2.5 - 0.3 * np.log1p(days)
    p = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return {"p": p, "right": right, "wrong": wrong, "delta_days": days,
            "user": np.array([f"u{i % 50}" for i in range(n)])}


def test_classification_metrics_match_a_hand_count():
    p = np.array([0.0, 0.2, 1.0, 1.0, 0.0])        # forgot, forgot, remembered, remembered, forgot
    q = np.array([0.3, 0.9, 0.2, 0.95, 0.4])       # flagged as forgotten when q < 0.5: rows 0, 2, 4
    m = classic.classification(p, q, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5) and m["f1"] == pytest.approx(2 / 3)


def test_log_loss_for_fractional_targets():
    assert classic.log_losses(np.array([1.0]), np.array([0.9]))[0] == pytest.approx(-np.log(0.9))
    assert classic.log_losses(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(np.log(2))


PROBE = {"p": np.zeros(2), "right": np.array([3, 3]), "wrong": np.array([0, 4]), "delta_days": np.array([1.0, 1.0])}


@pytest.mark.parametrize("name,params", [("knn", {"k": 50}), ("decision_tree", {"max_depth": 4}),
                                         ("random_forest", {"max_depth": 4})])
def test_course_models_learn_that_mistakes_hurt(name, params):
    clean, missed = classic.SklearnRecall(name, **params).fit(synthetic()).predict(PROBE)
    assert 0 < missed < clean < 1


def test_svm_regression_predicts_the_typical_outcome_not_a_probability():
    """SVR's epsilon-insensitive loss behaves like absolute error, so it estimates something close to the
    median outcome. With most sessions remembered, it predicts ~1 even after four mistakes: the same failure
    MAE caused for half-life regression, and the reason the SVM loses on log loss."""
    data = synthetic()
    clean, missed = classic.SklearnRecall("svm", C=1.0).fit(data).predict(PROBE)
    true_missed = 1 / (1 + np.exp(-(0.3 * 2 - 0.8 * np.sqrt(5) + 2.5 - 0.3 * np.log(2))))
    assert missed > 0.9 and missed - true_missed > 0.15 and abs(clean - missed) < 0.05  # truth: 0.75


def test_threshold_is_chosen_from_validation_predictions():
    data = synthetic()
    q = classic.SklearnRecall("decision_tree", max_depth=4).fit(data).predict(data)
    threshold = classic.forgot_threshold(data["p"], q)
    assert 0 < threshold < 1
    best = classic.classification(data["p"], q, threshold)["f1"]
    assert best >= classic.classification(data["p"], q, 0.5)["f1"]


def test_svm_and_knn_train_on_capped_subsamples(monkeypatch):
    monkeypatch.setattr(classic, "SVM_TRAIN", 500)
    model = classic.SklearnRecall("svm", C=1.0).fit(synthetic())
    assert model.trained_on == 500
