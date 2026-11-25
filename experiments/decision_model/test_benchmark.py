"""Synthetic validity and model invariants; no external services or personal data."""
import copy
import inspect
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

np = pytest.importorskip(
    "numpy",
    reason="Use the isolated decision_model uv environment"
)

torch = pytest.importorskip(
    "torch",
    reason="Use the isolated decision_model uv environment"
)
torch = pytest.importorskip("torch", reason="Use the isolated decision_model uv project for these tests")

from experiments.decision_model.data import family_split, generate_partition, load_config, load_partition, samples_from_episode
from experiments.decision_model.features import FeatureEncoder, ablate, drop_history_channels, raw_arrays, raw_window
from experiments.decision_model.metrics import evaluate_predictions
from experiments.decision_model.models import HistoricalProbability, LSTM, masked_bce
from experiments.decision_model.prepare import prepare
from experiments.decision_model.simulator import episode, persona
from experiments.decision_model.train import train


@pytest.fixture
def small_config():
    config = load_config()
    config.update(personas_per_family=2, episodes_per_persona=1, epochs=2, model_seeds=[7], hidden_sizes=[4])
    return config


def test_grouped_split_and_random_stream_independence(small_config):
    config = small_config
    partitions = {split: generate_partition(config, split) for split in ("train", "validation", "test")}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("family", "persona_id", "episode_id", "group_id", "id"):
            first = {row[key] for records in partitions[left] for row in records}
            second = {row[key] for records in partitions[right] for row in records}
            assert not first & second
    assert {key: len(value) for key, value in family_split(config).items()} == {"train": 6, "validation": 3, "test": 3}
    family = family_split(config)["train"][0]
    person = persona(config, family, 0)
    initial = episode(config, person)
    changed = {**config, "augmentation_seed": config["augmentation_seed"] + 1, "personas_per_family": 3}
    assert family_split(changed) == family_split(config)
    assert persona(changed, family, 0) == person
    augmented = episode(changed, person)
    assert initial["daily"] == augmented["daily"]  # augmentation cannot alter labels or assignment
    assert initial["events"] != augmented["events"]
    assert initial == episode(config, person)


def test_chronology_future_cutoff_and_label_noninterference(small_config):
    rows, _ = generate_partition(small_config, "train")
    sample = rows[0]
    cutoff = datetime.fromisoformat(sample["cutoff"])
    assert datetime.fromisoformat(sample["report_window_start"]) > cutoff
    assert all(datetime.fromisoformat(event["timestamp"]) <= cutoff for event in sample["history"])
    values, masks = raw_window(sample)
    assert values.shape == masks.shape == (14, 14)
    changed = copy.deepcopy(sample)
    changed["targets"], changed["target_mask"], changed["oracle"] = [1, 1], [1, 1], [0, 0]
    assert np.array_equal(values, raw_window(changed)[0])
    changed["history"].append({**sample["history"][-1], "timestamp": (cutoff + timedelta(seconds=2)).isoformat()})
    with pytest.raises(ValueError, match="cutoff"):
        raw_window(changed)


def test_missing_report_is_distinct_from_zero_and_rejections_do_not_adjust(small_config):
    rows, _ = generate_partition(small_config, "train")
    x, observed = raw_arrays(rows)
    assert ((x[:, :, 4] == 0) & (observed[:, :, 2] == 0)).any()
    assert ((x[:, :, 4] == 1) & (x[:, :, 2] == 0) & (observed[:, :, 2] == 1)).any()
    for row in rows:
        assert row["target_mask"][1] == row["targets"][0]
    rejected = x[:, :, 8] == 1
    assert rejected.any()
    assert np.array_equal(x[:, :, 0][rejected], x[:, :, 1][rejected])
    approved = x[:, :, 7] == 1
    assert approved.any() and (x[:, :, 1][approved] < x[:, :, 0][approved]).all()


def test_pair_last_events_identical_but_history_and_future_probability_differ(small_config):
    _, pairs = generate_partition(small_config, "test")
    assert len(pairs) % 2 == 0
    for high, low in zip(pairs[::2], pairs[1::2]):
        xh, mh = raw_window(high)
        xl, ml = raw_window(low)
        assert high["group_id"] == low["group_id"]
        assert np.array_equal(xh[-1], xl[-1]) and np.array_equal(mh[-1], ml[-1])
        assert not np.array_equal(xh[:-1], xl[:-1])
        assert all(a > b for a, b in zip(high["oracle"], low["oracle"]))


def test_preprocessing_training_only_masks_and_robustness(small_config):
    rows, _ = generate_partition(small_config, "train")
    values, masks = raw_arrays(rows)
    encoder = FeatureEncoder().fit(values, masks)
    saved_mean = encoder.mean.copy()
    altered = values * 1000
    encoder.transform(altered, masks)
    assert np.array_equal(saved_mean, encoder.mean)
    encoded = encoder.transform(values, masks)
    assert np.all(encoded[:, :, :14][masks == 0] == 0)
    damaged, damaged_masks = drop_history_channels(values, masks, [row["id"] for row in rows], 1, .5)
    assert np.array_equal(damaged[:, -1], values[:, -1])
    assert (damaged_masks[:, :-1, 4] == 0).any()
    shuffled = ablate(encoded, "shuffled", [row["id"] for row in rows], 1)
    assert np.array_equal(encoded[:, -1], shuffled[:, -1])
    assert not np.array_equal(encoded[:, :-1], shuffled[:, :-1])
    assert ablate(encoded, "no_history", [row["id"] for row in rows], 1).shape[1] == 1


def test_masked_loss_zero_gradient_and_conditional_scoring():
    logits = torch.zeros((3, 2), requires_grad=True)
    targets = torch.tensor([[1., float("nan")], [0., 1.], [1., 0.]])
    masks = torch.tensor([[1., 0.], [1., 1.], [1., 1.]])
    loss = masked_bce(logits, targets, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0, 1] == 0 and logits.grad[1, 1] != 0
    y = np.array([[0, 0], [1, 0], [1, 1]])
    mask = np.array([[1, 0], [1, 1], [1, 1]])
    p = np.array([[.2, .99], [.8, .1], [.8, .9]])
    first = evaluate_predictions(y, mask, p)
    p[0, 1] = .01
    assert first == evaluate_predictions(y, mask, p)
    assert first["B_completion_given_report"]["n"] == 2


def test_train_never_opens_test_and_checkpoints_round_trip(tmp_path, small_config, monkeypatch):
    data, models = tmp_path / "data", tmp_path / "models"
    prepare(data, small_config)
    import experiments.decision_model.train as training
    original = training.load_partition
    accessed = []

    def guard(folder, split, **kwargs):
        assert split in {"train", "validation"}, "Training touched held-out test"
        accessed.append(split)
        return original(folder, split, **kwargs)

    monkeypatch.setattr(training, "load_partition", guard)
    result = train(data, models)
    assert accessed == ["train", "validation"]
    assert (models / "selection.json").exists()
    from experiments.decision_model.evaluate import load_network
    entry = result["models"]["lstm_joint_full_s7"]
    loaded = load_network(models / entry["file"])
    assert loaded.predict_proba(np.zeros((2, 14, 28), dtype=np.float32)).shape == (2, 2)
    # Hash seal rejects edited data instead of silently consuming a new test set.
    path = data / "test.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="sealed"):
        load_partition(data, "test")


def test_window_boundary_provenance_and_invalid_inputs(small_config):
    rows, _ = generate_partition(small_config, "train")
    row = copy.deepcopy(rows[0])
    row["history"][0]["source"] = "model_interpretation"
    with pytest.raises(ValueError, match="provenance"):
        raw_window(row)
    cutoff = datetime.fromisoformat(rows[0]["cutoff"])
    goal = {"timestamp": cutoff.replace(hour=9, minute=0, second=0).isoformat(),
            "kind": "goal_declared", "source": "user_reported", "payload": {"minutes": 30}}
    report = {"timestamp": cutoff.replace(hour=23, minute=0, second=0, microsecond=0).isoformat(),
              "kind": "progress_reported", "source": "user_reported", "payload": {"minutes": 0}}
    row["history"] = [goal, report]
    assert raw_window(row)[0][-1, 4] == 1
    report["timestamp"] = cutoff.replace(hour=23, minute=0, second=0, microsecond=1).isoformat()
    values, masks = raw_window(row)
    assert values[-1, 4] == 0 and masks[-1, 2] == 0 and values[-1, 13] == 1
    assert not any(name in inspect.getsource(__import__("experiments.decision_model.models", fromlist=[""]))
                   for name in ("import discord", "import sqlite3", "from router import"))
