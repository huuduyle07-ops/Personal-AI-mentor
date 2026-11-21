# Offline router experiment

This package is research infrastructure. Production does not import it or load its model. All examples are synthetic and no Discord records are read.

## Hypothesis and baseline

Hypothesis: TF-IDF plus logistic regression may generalize across held-out accountability scenario families well enough to justify further research. The reference baseline is the production deterministic RuleRouter. Variables changed in this experiment are the routing policy and validation-selected regularization strength; both policies receive the same observable metadata and text.

Routing labels:

| Label | Meaning |
|---|---|
| RULE | A supported structured command can be handled deterministically |
| CLARIFY | Required fields are missing/invalid, an operation is unknown, or input is too terse |
| LLM | Sufficient free text needs interpretation; no application mutation is authorized |
| DEFER | Explicitly non-urgent free text may wait; not an automatic retry queue |

## Reproduce

```powershell
uv sync --locked --extra research
uv run --extra research python -m experiments.router.run --seed 42
uv run --extra research python -m pytest tests/test_experiment.py -q
```

Run from the repository root. `--output` selects a separate artifact directory. Default output is `experiments/router/artifacts/`, ignored by Git. The generated train/validation/test JSONL and metrics are readable; `baseline.joblib` is pickle-based, so load only artifacts you created and trust.

`dataset.py` defines six labeled scenario families per class. Seeded stratified family assignment gives three training, one validation, and two test families per class. With 20 unique numbered examples per family, sizes are 240/80/160. Whole templates, not individual paraphrases of the same template, are held out. Numeric substitutions diversify synthetic identifiers; they do not create independent semantic scenarios. Labels are specified independently in the dataset rather than produced by running the reference router.

`baseline.py` exposes `ExperimentalPolicy.predict(events)` and a TF-IDF character-ngram/logistic-regression implementation with save/load. Features contain text plus command, minutes, priority and non_urgent fields; neither label nor family ID is encoded. Candidate C values are 0.5, 2 and 8; the best validation macro-F1 selects the model, with ties preferring smaller C. The test split is evaluated only after selection. No test-guided tuning was performed.

`run.py` checks the saved/reloaded policy's prediction equality and records accuracy, per-class precision/recall/F1, confusion matrix, artifact bytes, training time, environment versions and median/p95 single-input inference latency after a warm-up. The Python allocation peak comes from tracemalloc and excludes native allocations; it is not total process memory. No GPU is used.

## Measured result (seed 42, 2026-11-22 KST)

| Policy | Test accuracy | Macro F1 | Median latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|---:|
| RuleRouter | 1.0000 | 1.0000 | 0.0028 | 0.0041 |
| TF-IDF + Logistic Regression | 0.6250 | 0.5595 | 2.2109 | 5.5391 |

Selected C: 0.5; validation macro-F1: 0.1000. Training plus model selection: 1.282s. Artifact: 138,300 bytes. Peak Python allocations: 5,855,934 bytes.

Confusion matrices use true rows and predicted columns in the order RULE, CLARIFY, LLM, DEFER.

| Policy / true label | RULE | CLARIFY | LLM | DEFER |
|---|---:|---:|---:|---:|
| Rule / RULE | 40 | 0 | 0 | 0 |
| Rule / CLARIFY | 0 | 40 | 0 | 0 |
| Rule / LLM | 0 | 0 | 40 | 0 |
| Rule / DEFER | 0 | 0 | 0 | 40 |
| Baseline / RULE | 20 | 20 | 0 | 0 |
| Baseline / CLARIFY | 0 | 40 | 0 | 0 |
| Baseline / LLM | 0 | 40 | 0 | 0 |
| Baseline / DEFER | 0 | 0 | 0 | 40 |

Full per-class metrics and configuration are preserved in [results_seed42.json](results_seed42.json). Reruns reproduce data and predictions on the locked environment; timing measurements vary by machine and load.

## Interpretation and limitations

The learned baseline underperformed the rule router: test LLM recall was zero, and half the held-out RULE examples were classified CLARIFY. This result does not support replacing the current deterministic policy. Short vocabulary templates, few independent families, class-specific text, and strong explicit metadata make this an engineering benchmark, not a realistic language-understanding evaluation. Rule-perfect labels also describe the rule contract by design. Synthetic test accuracy is not evidence of behavioral benefit or safety with real users.

The next experiment should add more independently reviewed scenario families and adversarial missing-field cases, report performance across seeds, and compare explicit categorical metadata features with the current character-ngram representation. Keep validation and test families separate before any tuning. Only after a stable benchmark should a temporal simulator define sequence features, missingness and action contracts for future LSTM or Mamba policies through the same predict interface. No LSTM, Mamba, distillation, or critic model has been trained or evaluated.
