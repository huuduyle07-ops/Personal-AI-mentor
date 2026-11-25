"""Offline audit invariants; synthetic examples only."""
import copy

import pytest

pytest.importorskip("sklearn")

from experiments.router.audit import check_dataset, select
from experiments.router.baseline import Baseline
from experiments.router.dataset import generate
from experiments.router.mixed_baseline import MixedBaseline, columns


def test_family_template_and_label_integrity():
    for seed in (0, 1, 42, 2026):
        result = check_dataset(generate(seed))
        assert result["family_counts"] == {"train": 12, "validation": 4, "test": 8}
        assert result["row_counts"] == {"train": 240, "validation": 80, "test": 160}


@pytest.mark.parametrize("corruption", ["overlap", "label", "template_overlap"])
def test_integrity_check_rejects_leakage_or_label_corruption(corruption):
    splits = generate(42)
    if corruption == "overlap":
        splits["validation"].append(copy.deepcopy(splits["train"][0]))
    elif corruption == "label":
        splits["validation"][0]["label"] = "RULE"
        # This specific row is LLM in the seeded data.
        assert splits["validation"][0]["family"] == "work_conflict"
    else:
        splits["validation"][0]["input"]["text"] = splits["train"][0]["input"]["text"]
    with pytest.raises(ValueError):
        check_dataset(splits)


def test_seed42_validation_collapse_is_reproducible():
    splits = generate(42)
    _, result = select(Baseline, 42, splits["train"], splits["validation"])
    assert result["selected_C"] == 0.5
    assert result["validation"]["macro_f1"] == pytest.approx(0.1)
    assert result["validation"]["confusion_matrix"] == [[0, 20, 0, 0]] * 4
    assert all(candidate["validation"]["macro_f1"] == pytest.approx(0.1) for candidate in result["candidates"])


def test_mixed_features_fit_training_only_and_handle_unseen_commands():
    splits = generate(42)
    model, _ = select(MixedBaseline, 42, splits["train"], splits["validation"])
    transformer = model.pipeline.steps[0][1]
    assert "checkins" not in transformer.named_transformers_["categories"].categories_[0]
    assert "checkins" in columns([row["input"] for row in splits["validation"]])[:, 1]
    assert set(model.predict([row["input"] for row in splits["test"]])) <= {"RULE", "CLARIFY", "LLM", "DEFER"}
    assert model.pipeline.steps[1][1].n_features_in_ == transformer.transform(columns([splits["test"][0]["input"]])).shape[1]


def test_all_missing_training_minutes_remain_transformable():
    train = generate(42)["train"]
    for row in train:
        row["input"]["minutes"] = None
    model = MixedBaseline().fit(train)
    event = {**train[0]["input"], "minutes": 30, "command": "never_seen"}
    assert len(model.predict([event])) == 1
