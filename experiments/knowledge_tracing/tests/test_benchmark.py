"""Synthetic fixtures only: leakage, ordering, preprocessing, masks and sealing."""
import copy
import inspect
import io
import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch", reason="Use the isolated knowledge_tracing uv project")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow", reason="Use the isolated knowledge_tracing uv project")

from experiments.knowledge_tracing import train as training
from experiments.knowledge_tracing.data_adapter import RangeFile, learner_split, reconstruct
from experiments.knowledge_tracing.dataset import Windows, make_windows
from experiments.knowledge_tracing.features import FeatureEncoder, context_of, events_array, perturb
from experiments.knowledge_tracing.metrics import learner_bootstrap_delta, scores
from experiments.knowledge_tracing.models import FittedPredictor
from experiments.knowledge_tracing.models.gru import RecurrentModel
from experiments.knowledge_tracing.models.statistical import StatisticalModel
from experiments.knowledge_tracing.utils import DEFAULT_CONFIG, digest, read_json, write_json


@pytest.fixture
def histories():
    return [{"user_id": f"synthetic-{j}", "events": [
        {"user_id": f"synthetic-{j}", "days": i / 10, "interaction_index": i + 1,
         "source_partition": "train", "source_row": 10*j+i,
         "correct": int((i+j) % 3 != 0), "response_time_seconds": None if i % 4 == 0 else i + 2,
         "session": "lesson" if i % 2 else "practice", "format": "listen" if i % 3 else "reverse_translate",
         "reference_tokens": ["unavailable-before-answer"], "token_labels": [i % 2],
         "token_accuracy": float(i % 2)} for i in range(10)]} for j in range(4)]


def test_learner_disjoint_deterministic_split():
    ids = [f"id-{i}" for i in range(100)]
    split = learner_split(ids, 42)
    assert split == learner_split(list(reversed(ids)), 42)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [70, 15, 15]
    assert len(set().union(*map(set, split.values()))) == 100
    assert not set(split["train"]) & set(split["test"])


def test_source_partition_order_and_ties(histories):
    rows = copy.deepcopy(histories[0]["events"][:3])
    rows[1]["days"] = rows[0]["days"]
    rows[2].update(source_partition="validation", interaction_index=1)
    grouped, audit = reconstruct(rows)
    assert [e["sequence_index"] for e in grouped["synthetic-0"]] == [0, 1, 2]
    assert audit["equal_day_transitions"] == 1
    assert audit["cross_partition_transitions"] == 1
    rows[2]["days"] = -1
    with pytest.raises(ValueError, match="ordering time|Nonmonotone"):
        reconstruct(rows)


def test_no_silent_sort_or_duplicate(histories):
    rows = histories[0]["events"]
    with pytest.raises(ValueError, match="Duplicate"):
        reconstruct([rows[0], rows[0]])
    with pytest.raises(ValueError):
        reconstruct([rows[2], rows[1]])


def test_target_outcome_and_reference_noninterference(histories):
    encoder = FeatureEncoder.fit(histories)
    before = Windows(histories[:1], encoder, 16)
    changed = copy.deepcopy(histories[:1])
    target = changed[0]["events"][-1]
    target.update(correct=1-target["correct"], token_labels=[99], token_accuracy=-999,
                  response_time_seconds=999999, reference_tokens=["secret"], exercise_id="secret",
                  token_ids=["secret"], pos_tags=["secret"], days=999)
    after = Windows(changed, encoder, 16)
    for name in ("x", "lengths", "summary", "context"):
        np.testing.assert_array_equal(getattr(before, name)[-1], getattr(after, name)[-1])
    assert before.y[-1] != after.y[-1]
    assert set(context_of(target)) == {"format", "session"}


def test_identifiers_not_features(histories):
    encoder = FeatureEncoder.fit(histories)
    a = Windows(histories, encoder, 16)
    changed = copy.deepcopy(histories)
    for learner in changed:
        learner["user_id"] = "other-" + learner["user_id"]
        for event in learner["events"]:
            event["user_id"], event["exercise_id"] = "other", "reference-derived-secret"
    b = Windows(changed, encoder, 16)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.summary, b.summary)


@pytest.mark.parametrize("length", [16, 32, 64])
def test_requested_windows_are_causal_and_masks_match(histories, length):
    long = copy.deepcopy(histories[:1])
    template = long[0]["events"]
    long[0]["events"] = [dict(template[i % len(template)], days=i/10,
                              interaction_index=i+1, correct=int(i % 2 == 0)) for i in range(80)]
    encoder = FeatureEncoder.fit(long)
    data = Windows(long, encoder, length)
    assert data.x.shape[1] == length
    np.testing.assert_array_equal(data.lengths, np.minimum(np.arange(1, 80), length))
    assert np.count_nonzero(data.x[0, 1:]) == 0
    assert data.y[-1] == long[0]["events"][-1]["correct"]
    expected = encoder.encode_events(events_array(long[0]["events"])[:-1][-length:])
    np.testing.assert_array_equal(data.x[-1], expected)


def test_train_only_fit_and_missing_not_zero(histories):
    encoder = FeatureEncoder.fit(histories)
    original = asdict(encoder)
    unfamiliar = copy.deepcopy(histories[:1])
    unfamiliar[0]["events"][-1].update(format="unseen-category", response_time_seconds=1e20)
    Windows(unfamiliar, encoder, 16)
    assert asdict(encoder) == original
    a, b = events_array(histories[0]["events"][:2])
    a["response"], b["response"] = np.nan, 0
    encoded = encoder.encode_events([a, b])
    assert encoded[0, 4] == 1 and encoded[1, 4] == 0
    assert encoder.categorical({"format": "new", "session": None})[1] == 1


@pytest.mark.parametrize("kind", ["gru", "lstm"])
def test_padding_does_not_influence_predictions_or_gradients(kind):
    torch.manual_seed(42)
    model = RecurrentModel(3, 2, 2, hidden_size=4, kind=kind)
    x = torch.randn(3, 8, 3, requires_grad=True)
    lengths = torch.tensor([3, 6, 0])
    summary, context = torch.randn(3, 2), torch.randn(3, 2)
    original = model(x, lengths, summary, context)
    changed = x.detach().clone()
    for i, length in enumerate(lengths):
        changed[i, length:] = 1e6
    torch.testing.assert_close(original, model(changed, lengths, summary, context))
    original.sum().backward()
    for i, length in enumerate(lengths):
        assert torch.count_nonzero(x.grad[i, length:]) == 0


def test_shuffled_complete_event_bundles_and_missingness(histories):
    bundles = events_array(histories[0]["events"])
    shuffled = perturb(bundles, "shuffled", "fixture", 42)
    assert shuffled == perturb(bundles, "shuffled", "fixture", 42)
    assert sorted(map(repr, shuffled)) == sorted(map(repr, bundles))
    dropped = perturb(bundles, "missing50", "fixture", 42)
    assert len(dropped) < len(bundles)
    assert all(event in bundles for event in dropped)
    assert all(np.isnan(e["correct"]) for e in perturb(bundles, "no_correctness", "fixture", 42))


@pytest.mark.parametrize("mode", ["no_correctness", "no_time", "no_metadata"])
def test_feature_ablations_cannot_bypass_removed_inputs_through_summaries(histories, mode):
    encoder = FeatureEncoder.fit(histories)
    altered = copy.deepcopy(histories)
    for learner in altered:
        for event in learner["events"]:
            if mode == "no_correctness":
                event["correct"] = 1 - event["correct"]
            elif mode == "no_time":
                event["days"] *= 100
                event["response_time_seconds"] = 1000000
            else:
                event["format"] = event["session"] = "altered"
    original = Windows(histories, encoder, 16, mode)
    changed = Windows(altered, encoder, 16, mode)
    for name in ("x", "summary", "context"):
        np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))


def test_statistical_prior_and_expanding_history(histories):
    encoder = FeatureEncoder.fit(histories)
    model = StatisticalModel(0.6, 2)
    assert model.predict_events([]) == pytest.approx(0.6)
    data = make_windows(histories[:1], encoder, 0)
    assert data.statistics[-1, 1] == 9
    assert data.statistics[-1, 0] == sum(e["correct"] for e in histories[0]["events"][:-1])
    assert model.predict_counts(data.statistics)[0] == pytest.approx((histories[0]["events"][0]["correct"]+1.2)/3)


@pytest.mark.parametrize("kind", ["gru", "lstm", "statistical"])
def test_roundtrip_and_single_prediction_contract(histories, tmp_path, kind):
    encoder = FeatureEncoder.fit(histories)
    data = Windows(histories[:1], encoder, 16)
    model = (StatisticalModel(.6, 2) if kind == "statistical" else
             RecurrentModel(data.x.shape[-1], data.summary.shape[-1], data.context.shape[-1], 4, kind))
    predictor = FittedPredictor(kind, model, encoder, 16)
    values = predictor.predict_windows(data)
    history = histories[0]["events"]
    assert predictor.predict(history[:-1], context_of(history[-1])) == pytest.approx(values[-1], abs=1e-6)
    predictor.save(tmp_path)
    restored = FittedPredictor.load(tmp_path)
    np.testing.assert_allclose(values, restored.predict_windows(data), atol=1e-6)


def test_metrics_calibration_and_group_weighting():
    y, p, groups = np.array([0, 1, 1, 0]), np.array([.1, .8, .6, .7]), np.array([0, 0, 0, 1])
    result = scores(y, p, groups)
    assert sum(b["n"] for b in result["calibration"]["bins"]) == 4
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["evaluated_learners"] == 2
    delta = learner_bootstrap_delta(y, p, p, groups)
    assert delta["paired_learner_bootstrap_95_ci"] == [0, 0]
    assert scores(np.ones(3), np.ones(3)*.8)["roc_auc"] is None


def test_range_cache_offline_and_no_whole_file_fallback(tmp_path, monkeypatch):
    from experiments.knowledge_tracing import data_adapter
    write_json(tmp_path / "size.json", {"size": 10})
    (tmp_path / "2-4.bin").write_bytes(b"abc")
    reader = RangeFile("https://example.test/file", tmp_path, offline=True)
    reader.seek(2)
    assert reader.read(3) == b"abc"
    with pytest.raises(FileNotFoundError):
        reader.read(1)
    class IgnoredRange(io.BytesIO):
        status = 200
        headers = {}
    monkeypatch.setattr(data_adapter, "fetch", lambda *a, **k: IgnoredRange(b"entire-file"))
    with pytest.raises(RuntimeError, match="refusing full"):
        RangeFile("https://example.test/file", tmp_path / "new")


def test_training_never_loads_test_and_evaluation_seals(histories, tmp_path, monkeypatch):
    from experiments.knowledge_tracing import evaluate as evaluation
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    files = {}
    for split in ("train", "validation", "test"):
        path = data / f"{split}.jsonl"
        split_histories = copy.deepcopy(histories)
        for learner in split_histories:
            learner["user_id"] = split + learner["user_id"]
        path.write_text("\n".join(json.dumps(h) for h in split_histories), encoding="utf-8")
        files[path.name] = digest(path)
    write_json(data / "manifest.json", {"files": files, "splits": {}, "provenance": {}})
    config = read_json(DEFAULT_CONFIG)
    config.update(epochs=1, sequence_lengths=[16], logistic_c=[1], statistical_strength=[2], ablations=[])
    actual = training.load_split
    touched = []
    def guarded(data, split):
        touched.append(split)
        assert split != "test"
        return actual(data, split)
    monkeypatch.setattr(training, "load_split", guarded)
    training.train(config, data, models)
    assert touched == ["train", "validation"]
    assert not (models / "test_opened.json").exists()
    with pytest.raises(FileExistsError):
        training.train(config, data, models)
    # Exercise the complete evaluator and plots using synthetic learners only.
    original_eval_load = evaluation.load_split
    def load_after_seal(*args):
        assert (models / "test_opened.json").exists()
        return original_eval_load(*args)
    monkeypatch.setattr(evaluation, "load_split", load_after_seal)
    result = evaluation.evaluate(data, models, tmp_path / "evaluation")
    assert set(result["main"]) == {"statistical", "logistic", "gru", "lstm", "mlp"}
    assert all("ordered_minus_control" in r for r in result["retrained_controls"])
    assert set(result["capacity_comparisons"]) == {"gru", "lstm"}
    assert len(list((tmp_path / "evaluation").glob("*.png"))) == 4
    with pytest.raises(ValueError, match="already opened"):
        evaluation.evaluate(data, models, tmp_path / "evaluation")
    prediction_file = next(models.rglob("weights.pt"))
    prediction_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Artifact changed"):
        evaluation.evaluate(data, models, tmp_path / "evaluation", replay=True)
