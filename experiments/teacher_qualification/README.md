# PM teacher qualification

Phase 2A tests whether the installed Qwen candidates can annotate multilingual accountability messages well enough to justify a later, independently reviewed distillation experiment. It runs entirely through a disposable Ollama server on Windows. See [measured results](BENCHMARK.md).

## Reproduce

Run from the repository root using the existing Python environment. Keep the sibling [PM SFT contract](../pm_sft/contract.py) in the checkout. The live run needs `ollama.exe`, the two installed model tags, and access to loopback and Windows performance counters. `nvidia-smi` provides device VRAM samples when available. The offline evaluator and tests need no running model. Choose a fresh local artifact directory for every run.

```powershell
$teacherPython = '.venv/Scripts/python.exe'
$teacherRun = 'experiments/teacher_qualification/artifacts/reproduction'

& $teacherPython -m pytest -q experiments/teacher_qualification
& $teacherPython -m experiments.teacher_qualification.run --output $teacherRun
& $teacherPython -m experiments.teacher_qualification.evaluate --run $teacherRun --review "$teacherRun/review.csv"
```

The run command exits nonzero on incomplete work and preserves the manifest. The evaluation command supports partial runs. Omit `--run` to export the references alone. Keep the Ollama desktop server idle and avoid starting other inference workloads while the benchmark runs. The benchmark binds a separate loopback port, enforces one server request slot and one loaded model, disables cloud and Vulkan, and checks the desktop server for loaded models before each inference call. It verifies both tags against the local inventory and uses the existing model store. See [configuration](config.json), [Ollama generation API](https://docs.ollama.com/api/generate), and [running model metadata](https://docs.ollama.com/api/ps).

The requested settings are 2,048 context tokens, 256 output tokens, temperature 0, seed 42, JSON Schema and `think: false`. A separate unscored request captures cold loading for each candidate. The runner confirms the context reported by Ollama, checks available RAM before the cold request, and unloads and verifies each runner before advancing. A hard RAM floor of 768 MiB stops immediately. Three consecutive samples showing available RAM below 1,536 MiB, available commit below 2,048 MiB, or hard page reads of at least 32 MiB/s with available RAM below 3,072 MiB stop the isolated process tree. A detected memory error or failed unload also stops the run. Raw hard page reads include mapped-file activity. The RAM condition helps distinguish pressure from ordinary weight loading.

## Reference and annotation contract

[reference.json](reference.json) contains 20 newly authored fictional scenario families with English, Vietnamese and Korean variants. It supplies 60 evaluation rows with correlated variants. Every language covers all nine labels with the same support. Progress appears seven times, clarification six times, recovery four times, query three times, obstacle, verification, approval and rejection twice each, and small talk once. The fixed date anchor is November 24, 2026 in Asia/Seoul. The labels are provisional and await independent human review. Neither evaluated model was used to generate or adjudicate them.

The vocabulary and slot constraints come from the Phase 1 PM SFT extraction contract. The production DM adapter uses a different actions envelope and requires explicit user input and database-backed checks for approval. In this experiment, `pending_id` is a synthetic fixture for matching spoken IDs. A matched ID can be annotated for review. Production authorization is determined by the current user, date, proposal status and plan revision in application state.

| Key | Meaning |
| --- | --- |
| `i` | All applicable PM intents |
| `s` | Extracted slots, including distinct completed minutes and requested target minutes |
| `m` | Fields requiring follow-up. `pending` represents a missing or mismatched pending fixture for a spoken proposal ID |
| `a` | Whether competing plausible meanings require clarification |
| `e` | Literal input quotations, one per intent in intent order and one per emitted slot |

The validator checks strict JSON, the closed schema, the PM contract, span existence and coverage, numeric units, explicit or anchored temporal values, and policy violations. It checks whether a whole recognized number or date expression supports a quote. Exact reference scoring catches wrong roles, missed corrections, and alternative selections that still quote real text. A literal intent quotation can pass the evidence check even if its meaning needs human adjudication. The validator preserves every original response.

## Scoring and review

- Report attempt coverage alongside rates. Attempted failures receive zero semantic credit. Unattempted rows stay outside measured rates. Intent metrics and slot counts require a valid schema and PM contract.
- Exact intent and missing-field accuracy compare unordered sets. Intent macro F1 averages the labels with observed gold or prediction support. Slot precision, recall and F1 use exact key/value pairs across responses, including extra and wrong values.
- Whole-example accuracy requires exact intents, slots, missing fields and ambiguity, plus valid evidence, numeric and temporal checks and no policy flags. Evidence validity means coverage and literal grounding. It allows alternatives to the provisional reference quotes.
- Policy flags include action or authorization claims, unknown intents, unsupported decisions, ungrounded proposal IDs, unrecognized unsupported requests and missed required clarification. Format errors and transport failures are separately visible.
- Latency is client wall time for completed responses, with nearest-rank p95. Throughput is generated tokens divided by Ollama's decode duration. Cold load uses the warmup envelope. Peak system RAM and device VRAM are sampled system-wide. The Ollama allocation split is separately reported from `size_vram / size`.
- The review CSV has all provisional gold labels, original outputs, validator results, disagreement markers and blank human-review columns. Disagreement compares the interpretations, ignoring evidence quote wording and set order. The two models share a family, so agreement also needs independent review. The manifest fingerprints the reference, prompt/schema and installed model digests. Raw inference, validation and scores have separate local artifacts.

For provisional primary-teacher suitability, require both 90% exact intents and 90% slot F1 overall, at least 85% exact intents in each language, 80% whole-example accuracy, 95% valid evidence, 90% missing-field recall, zero policy flags and a complete resource-feasible run. For secondary reviewer suitability, require 80% exact intents and 80% slot F1 overall, at least 70% exact intents in each language, 90% valid evidence, 80% missing-field recall, zero unsupported action/decision claims and a complete resource-feasible run. These are experiment decision rules. Independent human review is required before relying on the provisional labels, particularly exact boundaries of `goal` and `reason` spans.

The experiment directory is isolated from production and its artifacts directory is Git-ignored. The benchmark performs annotation qualification only.
