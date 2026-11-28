import json
import random

from dm_intents import normalized
from experiments.intent_sft import data
from llm_adapter import INTENT_PROMPT


def rows(name):
    path = data.OUT / f"{name}.jsonl"
    return [json.loads(line)["messages"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_training_messages_are_not_copies_of_evaluation_messages():
    evaluation = data.evaluation_messages()
    for name in ("train", "valid"):
        for messages in rows(name):
            assert not data.too_close(messages[1]["content"], evaluation), messages[1]["content"]


def test_train_and_validation_do_not_share_messages():
    train = {normalized(m[1]["content"]) for m in rows("train")}
    assert not train & {normalized(m[1]["content"]) for m in rows("valid")}


def test_every_label_passes_the_bots_own_checks():
    for name in ("train", "valid"):
        for system, user, answer in rows(name):
            assert system["content"] == INTENT_PROMPT  # trained with exactly the production prompt
            actions = json.loads(answer["content"])["actions"]
            assert data.consistent(user["content"], actions), user["content"]


def test_generation_is_reproducible():
    first = data.generate(random.Random(data.SEED), "ko")
    assert first == data.generate(random.Random(data.SEED), "ko")


def test_report_minutes_match_the_duration_in_the_text():
    for text, actions in data.generate(random.Random(1), "en"):
        if actions[0]["intent"] == "REPORT_PROGRESS" and len(actions) == 1:
            assert data.consistent(text, actions), text


def test_near_duplicates_are_caught():
    assert data.too_close("I studied 20 minutes today!", [normalized("i studied 20 minutes today")])
    assert not data.too_close("quiz me please", [normalized("what's my total yesterday")])
