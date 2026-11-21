import importlib.util

import pytest

from experiments.router.dataset import generate


def test_reproducible_family_split_without_text_overlap():
    splits = generate(42)
    assert splits == generate(42)
    assert splits != generate(43)
    families = {key: {row["family"] for row in value} for key, value in splits.items()}
    texts = {key: {row["input"]["text"] for row in value} for key, value in splits.items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not families[first] & families[second]
        assert not texts[first] & texts[second]
    assert all({row["label"] for row in records} == {"RULE", "CLARIFY", "LLM", "DEFER"} for records in splits.values())


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="Optional research extra is not installed")
def test_experiment_saves_loads_and_compares(tmp_path):
    from experiments.router.run import run

    metrics = run(tmp_path, samples_per_family=2)
    assert metrics["rule"]["accuracy"] == 1
    assert len(metrics["baseline"]["confusion_matrix"]) == 4
    assert metrics["artifact_bytes"] > 0
    assert (tmp_path / "metrics.json").exists()
