import csv
from collections import Counter
from copy import deepcopy
import json

import pytest

from experiments.pm_sft.contract import INTENTS, dumps

from .annotation import SCHEMA, validate
from .dataset import load_reference
from .evaluate import export, score, summarize
from .ollama import Ollama
from .resources import pressure_reason


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def row_for(reference, prefix):
    return next(row for row in reference if row["id"].startswith(prefix) and row["language"] == "en")


def test_reference_families_and_coverage(reference):
    assert Counter(r["language"] for r in reference) == {"en": 20, "vi": 20, "ko": 20}
    assert set(Counter(r["family"] for r in reference).values()) == {3}
    assert len({r["text"] for r in reference}) == 60
    for lang in ("en", "vi", "ko"):
        intents = Counter(i for r in reference if r["language"] == lang for i in r["gold"]["i"])
        assert intents.keys() == set(INTENTS)
        assert min(intents.values()) >= 1


@pytest.mark.parametrize("raw", ['{"i":[],"i":[]}', "NaN", "not json"])
def test_strict_parse(raw, reference):
    assert not validate(raw, reference[0])["parsed"]


@pytest.mark.parametrize("key,value", [("minutes", True), ("minutes", 26.0), ("minutes", -1),
                                      ("unknown", "today"), ("date", "last week")])
def test_schema_rejects_invalid_slot(key, value, reference):
    row = reference[0]
    annotation = deepcopy(row["gold"])
    annotation["s"][key] = value
    checks = validate(dumps(annotation), row)
    assert checks["parsed"] and not checks["schema"]


def test_grounding_requires_literal_spans_and_contiguous_quantity(reference):
    row = row_for(reference, "12-")
    annotation = deepcopy(row["gold"])
    annotation["s"]["minutes"] = 6
    annotation["e"]["s"]["minutes"] = "6 minutes"
    checks = validate(dumps(annotation), row)
    assert checks["evidence"] and not checks["numbers"]
    annotation["e"]["i"] = ["unattributed intent evidence"]
    assert not validate(dumps(annotation), row)["evidence"]


def test_hour_conversion_and_conflicting_temporal_phrase(reference):
    row = row_for(reference, "19-")
    assert validate(dumps(row["gold"]), row)["numbers"]
    annotation = deepcopy(row["gold"])
    annotation["s"]["minutes"] = 1
    assert not validate(dumps(annotation), row)["numbers"]
    row = row_for(reference, "17-")
    annotation = deepcopy(row["gold"])
    annotation["s"]["date"] = "yesterday"
    annotation["e"]["s"]["date"] = "yesterday"
    checks = validate(dumps(annotation), row)
    assert checks["evidence"] and not checks["dates"]


def test_correction_is_scored_without_repair(reference):
    row = row_for(reference, "12-")
    annotation = deepcopy(row["gold"])
    annotation["s"]["minutes"] = 64
    annotation["e"]["s"]["minutes"] = "64 minutes"
    raw = dumps(annotation)
    result = score(row, {"status": "ok", "response": {"response": raw}, "latency_s": 1})
    assert result["checks"]["numbers"]
    assert result["checks"]["value"]["s"]["minutes"] == 64
    assert result["metrics"]["exact_intent"] and not result["metrics"]["whole_example"]
    assert result["metrics"]["slot_counts"] == (0, 1, 1)


def test_approval_is_annotation_only(reference):
    row = row_for(reference, "06-")
    annotation = deepcopy(row["gold"])
    assert not validate(dumps(annotation), row)["policy_flags"]
    annotation["m"] = []
    annotation["i"] = ["APPROVE_ADJUSTMENT"]
    annotation["e"]["i"] = annotation["e"]["i"][:1]
    assert "decision_without_trusted_pending" in validate(dumps(annotation), row)["policy_flags"]
    annotation["authorized"] = True
    checks = validate(dumps(annotation), row)
    assert not checks["schema"]
    assert "action_or_authorization_claim" in checks["policy_flags"]


def test_unspoken_proposal_id_is_flagged_without_a_decision_intent(reference):
    row = row_for(reference, "14-")
    annotation = deepcopy(row["gold"])
    annotation["s"]["proposal"] = "P-2026"
    annotation["e"]["s"]["proposal"] = "P-2026"
    checks = validate(dumps(annotation), row)
    assert checks["schema"] and not checks["evidence"]
    assert checks["policy_flags"] == ["ungrounded_proposal_id"]


def test_partial_metrics_count_attempts_and_do_not_award_empty_matches(reference):
    row = row_for(reference, "08-")
    result = score(row, {"status": "timeout", "latency_s": 90})
    summary = summarize([result], expected=20)
    assert summary["attempted"] == 1 and summary["expected"] == 20
    assert summary["timeouts"]["rate"] == 1
    assert summary["missing_exact"]["count"] == 0
    assert summary["latency_s"]["n"] == 0
    assert summary["slot"]["f1"] is None
    assert summary["macro_intent_f1"] == 0


def test_review_retains_both_raw_outputs_and_flags_disagreement(tmp_path, reference):
    row = reference[0]
    first = {"status": "ok", "response": {"response": dumps(row["gold"])}, "latency_s": 1}
    second = {"status": "ok", "response": {"response": "{}"}, "latency_s": 1}
    models = {"4b": first, "9b": second}
    path = tmp_path / "review.csv"
    export(path, [row], {model: {row["id"]: raw} for model, raw in models.items()},
           {model: {row["id"]: score(row, raw)} for model, raw in models.items()})
    with path.open(encoding="utf-8-sig", newline="") as file:
        item = next(csv.DictReader(file))
    assert item["4b_raw"] == first["response"]["response"]
    assert item["9b_raw"] == "{}"
    assert item["model_disagreement"] == "True"


def test_generate_uses_structured_output_and_keeps_trusted_state_out_of_annotation(monkeypatch, reference):
    client = Ollama("http://127.0.0.1:11435")
    calls = []
    monkeypatch.setattr(client, "call", lambda *args, **kwargs: calls.append((args, kwargs)))
    row = row_for(reference, "07-")
    options = {"num_ctx": 2048, "num_predict": 256, "temperature": 0}
    client.generate("qwen3.5:4b", row, {"options": options, "think": False, "keep_alive": "2m", "timeout_s": 90})
    (api, payload), kwargs = calls[0]
    assert api == "generate" and kwargs["timeout"] == 90
    assert payload["format"] == SCHEMA and payload["options"] == options
    assert not payload["stream"] and not payload["think"]
    supplied = json.loads(payload["prompt"])
    assert supplied["pending_id"] == "P-84" and supplied["message"] == row["text"]
    assert set(supplied) == {"pending_id", "message", "anchor_day", "anchor_zone"}
    with pytest.raises(ValueError):
        Ollama("https://example.com")


def test_memory_pressure_requires_low_headroom_for_hard_reads():
    config = {"minimum_running_available_mib": 1536, "minimum_running_available_commit_mib": 2048,
              "paging_headroom_mib": 3072, "paging_read_mib_s": 32}
    sample = {"free_ram_mib": 6000, "free_commit_mib": 12000, "page_reads_mib_s": 300}
    assert pressure_reason(sample, config) is None
    assert pressure_reason({**sample, "free_ram_mib": 2048}, config) == "sustained_hard_page_reads_under_ram_pressure"
    assert pressure_reason({**sample, "free_ram_mib": 1000}, config) == "low_available_ram"
    assert pressure_reason({**sample, "free_commit_mib": 1000}, config) == "low_commit_headroom"
