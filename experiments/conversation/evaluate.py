"""Evaluate the DM interpreter on hand-labelled synthetic messages only."""
import argparse
import asyncio
import json
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

from config import Config
from dm_intents import APPROVE, CLARIFY, REJECT, control, interpret, normalized, validate
from llm_adapter import OllamaAdapter

DATA = Path(__file__).with_name("scenarios.jsonl")


def intents(actions):
    return {item["intent"] for item in actions}


def slots(actions):
    return {(item["intent"], key, value) for item in actions for key, value in item.items() if key != "intent"}


def ratio(count, total):
    return round(count / total, 4) if total else None


def latency(values):
    if not values:
        return None
    p95 = sorted(values)[ceil(len(values) * 0.95) - 1]
    return {"median": round(median(values), 3), "p95": round(p95, 3)}


async def evaluate(mode, model, url, timeout):
    cases = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]
    adapter = OllamaAdapter(Config(ollama_enabled=mode != "disabled", ollama_model=model,
                                   ollama_url=url, ollama_timeout=timeout))
    outcomes, langs = Counter(), {}
    correct = gold_slots = returned_slots = matched_slots = 0
    need_clarify = did_clarify = matched_clarify = invalid = fallback = safe = 0
    failures, times, model_times = [], [], []
    for case in cases:
        start = perf_counter()
        result = await interpret(case["text"], adapter)
        times.append((perf_counter() - start) * 1000)
        expected, actual = case["expected"], result.actions
        outcomes[result.source] += 1
        if result.source in {"ollama", "timeout", "unavailable_or_invalid"}:
            model_times.append(result.latency_ms)
        try:
            if intents(actual) & {APPROVE, REJECT}:
                if result.source != "rule" or actual != [control(normalized(case["text"]))]:
                    raise ValueError("Decision did not come from explicit user text")
            else:
                validate({"actions": actual})
        except ValueError:
            invalid += 1
        if result.source in {"disabled", "timeout", "unavailable_or_invalid"}:
            fallback += 1
            safe += intents(actual) == {CLARIFY}
        is_correct = intents(expected) == intents(actual)
        correct += is_correct
        lang = langs.setdefault(case["language"], [0, 0])
        lang[0] += is_correct
        lang[1] += 1
        gold, predicted = slots(expected), slots(actual)
        gold_slots += len(gold)
        returned_slots += len(predicted)
        matched_slots += len(gold & predicted)
        need_clarify += CLARIFY in intents(expected)
        did_clarify += CLARIFY in intents(actual)
        matched_clarify += CLARIFY in intents(expected) & intents(actual)
        if not is_correct or gold != predicted:
            failures.append({"id": case["id"], "returned": actual, "source": result.source})
    return {"mode": mode, "model": model if mode == "live" else None, "timeout_seconds": timeout, "count": len(cases),
            "intent_set_accuracy": ratio(correct, len(cases)),
            "intent_accuracy_by_language": {lang: ratio(*counts) for lang, counts in langs.items()},
            "slot_precision": ratio(matched_slots, returned_slots), "slot_recall": ratio(matched_slots, gold_slots),
            "invalid_action_rate_after_validation": ratio(invalid, len(cases)),
            "clarify_precision": ratio(matched_clarify, did_clarify),
            "clarify_recall": ratio(matched_clarify, need_clarify),
            "fallback_count": fallback, "safe_fallback_rate": ratio(safe, fallback),
            "latency_ms": latency(times), "backend_attempt_ms": latency(model_times),
            "outcomes": dict(outcomes), "errors": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["disabled", "unavailable", "live"], required=True)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--url")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    url = options.url or ("http://127.0.0.1:1" if options.mode == "unavailable" else "http://127.0.0.1:11434")
    report = asyncio.run(evaluate(options.mode, options.model, url, options.timeout))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output:
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    if options.mode == "live" and not report["outcomes"].get("ollama"):
        raise SystemExit("No validated live inference; inspect backend and model availability.")
