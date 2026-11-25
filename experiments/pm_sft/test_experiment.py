"""Offline tests for scoring, policy, split validation and assistant masking."""

from copy import deepcopy
import json

import pytest

from .contract import disposition, parse, validate
from .dataset import LANGUAGES, PARTITIONS, audit, load_split, prepare
from .evaluate import evaluate, score_one
from .model import encode_completion


@pytest.mark.parametrize("raw", [
    '{"i":["REPORT_PROGRESS"],"s":{"minutes":true},"m":[]}',
    '{"i":["REPORT_PROGRESS"],"s":{"minutes":1441},"m":[]}',
    '{"i":["QUERY_PROGRESS"],"s":{"date":"2026-02-30"},"m":[]}',
    '{"i":["CLARIFY"],"s":{"target":20},"m":["target"]}',
    '{"i":["CLARIFY"],"s":{},"m":[]}',
    '{"i":["CLARIFY","CLARIFY"],"s":{},"m":["minutes"]}',
    '{"i":["CLARIFY"],"i":["QUERY_PROGRESS"],"s":{},"m":[]}',
    '{"i":["QUERY_PROGRESS"],"s":{},"m":[],"action":"SET_TARGET"}',
])
def test_reject_invalid_output(raw):
    with pytest.raises(ValueError):
        validate(parse(raw))


def test_approval_requires_trusted_matching_pending_and_human_review():
    value = {"i": ["APPROVE_ADJUSTMENT"], "s": {"proposal": "P-42"}, "m": []}
    assert disposition(value, pending=None) == "clarify"
    assert disposition(value, pending="P-41") == "clarify"
    assert disposition(value, pending="P-42", unresolved=["date"]) == "clarify"
    assert disposition(value, pending="P-42") == "human_review"
    contradictory = {**value, "i": value["i"] + ["REJECT_ADJUSTMENT"]}
    assert disposition(contradictory, pending="P-42") == "clarify"


def test_report_requires_duration_and_external_uncertainty_overrides_prediction():
    value = {"i": ["REPORT_PROGRESS"], "s": {}, "m": []}
    assert disposition(value, pending=None) == "clarify"
    value["s"]["minutes"] = 19
    assert disposition(value, pending=None, unresolved=["goal"]) == "clarify"
    assert disposition(value, pending=None) == "human_review"


def test_scoring_separates_parse_schema_intents_slots_and_missing():
    row = {"id": "probe", "language": "ko", "pending": None,
           "gold": {"i": ["REPORT_PROGRESS", "CLARIFY"], "s": {"minutes": 20}, "m": ["date"]}}
    raw = '{"m":[],"s":{"minutes":20,"date":"today"},"i":["REPORT_PROGRESS"]}'
    outcome = score_one(row, raw, 5)
    assert outcome["parse_valid"] and outcome["schema_valid"] and outcome["policy_violation"]
    assert not outcome["intent_exact"] and not outcome["whole_json"] and not outcome["missing_exact"]
    assert outcome["slots"] == {"minutes": True, "date": False}
    invalid = score_one(row, '{"action":"SET_TARGET"}', 5)
    assert invalid["parse_valid"] and not invalid["schema_valid"] and invalid["policy_violation"]
    assert invalid["slots"] == {"minutes": False}
    assert not score_one(row, "{}", 5)["schema_valid"]
    assert not score_one(row, "Here is JSON: " + raw, 5)["parse_valid"]
    reversed_intents = {**row["gold"], "i": list(reversed(row["gold"]["i"]))}
    aggregate = evaluate([row], lambda _: json.dumps(reversed_intents))
    assert aggregate["by_language"]["ko"]["whole_json"] == 1
    assert aggregate["overall"]["per_intent"]["REPORT_PROGRESS"]["f1"] == 1


def test_family_leakage_and_seal(tmp_path):
    source = tmp_path / "input.psv"
    source.write_text("\n".join(
        f"{split}|{split}||GENERAL_CONVERSATION|{{}}||{split} hello|{split} chào|{split} 안녕"
        for split in PARTITIONS
    ), encoding="utf-8")
    output = tmp_path / "corpus"
    manifest = prepare(source, None, output)
    assert manifest["counts"]["train"]["languages"] == dict.fromkeys(LANGUAGES, 1)
    train, validation, holdout = [load_split(output, split) for split in PARTITIONS]
    duplicate = deepcopy(validation[0])
    duplicate["family"] = train[0]["family"]
    with pytest.raises(ValueError, match="crosses partitions"):
        audit(train + [duplicate] + validation[1:] + holdout)
    (output / "holdout.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_split(output, "holdout")
    assert len(load_split(output, "train")) == 3


def test_native_tokenizer_mask_if_cached():
    pytest.importorskip("transformers")
    from .dataset import CONFIG, read_json
    from .model import generation_text, load_tokenizer

    try:
        tok = load_tokenizer(read_json(CONFIG))
    except OSError:
        pytest.skip("Pinned TinyLlama tokenizer is not cached")
    row = {"id": "mask", "text": "오늘 18분 하고 끝냈어.", "pending": None,
           "gold": {"i": ["REPORT_PROGRESS"], "s": {"minutes": 18}, "m": []}}
    encoded = encode_completion(tok, row, 256)
    first = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
    assert encoded["labels"][:first] == [-100] * first
    prompt_tokens = tok(generation_text(tok, row), add_special_tokens=False)["input_ids"]
    assert prompt_tokens[:first] == encoded["input_ids"][:first]
    assert encoded["labels"][first:] == encoded["input_ids"][first:]
    assert encoded["labels"][-1] == tok.eos_token_id
    with pytest.raises(ValueError, match="no truncation"):
        encode_completion(tok, row, 32)
