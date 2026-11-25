# Multilingual DM evaluation

Can a small whole-message grammar handle the demo safely when Ollama is offline, and does optional local interpretation cover more paraphrases? This evaluation exercises the interpreter used by the Discord DM handler. The existing production router and its separate experiments have their own inputs and evaluation.

## Results

One run per mode on Windows, Python 3.11.9, November 24, 2026. All runs used the same 39 hand-labelled synthetic DMs, 13 each in English, Vietnamese and Korean. Twenty-seven messages took the rule path, including all three acceptance paraphrases. The live run used the installed `qwen3.5:4b` model, temperature zero, a fresh loopback server and a 60-second timeout. The application's default model and timeout retain their existing values.

| Mode | Exact intent set | Slot precision / recall | Clarify precision / recall | Invalid actions returned | Interpreter median / p95 |
|---|---:|---:|---:|---:|---:|
| Ollama disabled | 33/39 | 100% / 81.8% | 53.8% / 100% | 0/39 | 0.016 / 1.428 ms |
| Ollama enabled, endpoint unavailable | 33/39 | 100% / 81.8% | 53.8% / 100% | 0/39 | 0.053 / 2,047 ms |
| Ollama available | 38/39 | 93.9% / 93.9% | 87.5% / 100% | 0/39 | 0.072 / 1,820 ms |

The unavailable endpoint produced a clarification on **12/12 attempted calls**. Its median attempt took 2.03 s. All 12 live calls returned validated proposals, with a median of 0.99 s and a p95 of 9.44 s including initial model loading. The live run asked an unnecessary clarification for a Vietnamese vocabulary quiz request. It identified both intents in a Korean request about a heavy workload, but proposed tiredness and tomorrow instead of busyness and today. A wrong slot can pass schema validation. The user sees its meaning before a model-interpreted write can proceed.

The set includes simple totals, mixed messages, decisions with IDs, numeric bounds, speculative or negated reports and phrasing outside the grammar. It serves as a development and regression set. It is small and includes phrases used to refine the grammar, so these scores do not estimate performance on unseen conversations. The run uses synthetic text exclusively. [Scenarios](scenarios.jsonl) · [Disabled results](disabled.json) · [Unavailable results](unavailable.json) · [Live results](live.json)

## Reproduce

From the repository root, use the project environment. For the last command, start Ollama locally with the named model already installed. The evaluator reads its own JSONL and does not load the bot's environment or its personal database.

```powershell
uv run --locked python -m experiments.conversation.evaluate --mode disabled
uv run --locked python -m experiments.conversation.evaluate --mode unavailable
uv run --locked python -m experiments.conversation.evaluate --mode live --model qwen3.5:4b --timeout 60
uv run --locked python -m pytest tests/test_conversation.py -q
```

Use `--output <path>` to save JSON and `--url <loopback URL>` to select another local endpoint. Unavailable mode defaults to port 1. Live mode exits unsuccessfully if no validated model response was received. The result files include backend outcomes, accuracy by language, and the case IDs and actual proposals for every mismatch.

## Metric definitions

- Intent accuracy requires the complete set of intents in each message. Slot precision and recall use exact (intent, parameter, value) matches, including proposal IDs.
- Clarification precision and recall score the explicit `CLARIFY` class against the seven cases labelled ambiguous or unsupported. Draft confirmations for interpretable compound requests are exercised separately in integration tests.
- Invalid-action rate counts structurally invalid proposals returned to the caller after validation. Malformed model responses, unknown actions and model-generated approvals are also injected in integration tests to check the resulting SQLite state. Safe fallback rate is the fraction of disabled, failed or timed-out inference attempts that return clarification.
- Latency uses wall-clock time around interpretation, including parser and adapter overhead. The p95 is the nearest-rank percentile. Discord delivery and SQLite execution are outside these timings.
