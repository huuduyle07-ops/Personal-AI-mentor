# Real-World Knowledge Tracing Benchmark v0.1

An isolated, noncommercial SLAM pilot predicting **strict correctness of the next exercise on unseen learners**. It imports no bot, scheduler, router, SQLite store, or earlier experiment. It has no action-policy or production integration. See [dataset audit](results/DATASET_AUDIT.md), [measured report](results/REPORT.md), and [aggregate results](results/results.json).

## Reproduce from the repository root

Windows, Python 3.11, uv; PyTorch uses the explicit CPU wheel index. This package has its own dependency lock and environment; the application dependencies are untouched.

```powershell
uv --cache-dir .uv-cache sync --project experiments/knowledge_tracing --locked --python 3.11
uv run --project experiments/knowledge_tracing --locked python -m pytest experiments/knowledge_tracing/tests -q
uv run --project experiments/knowledge_tracing --locked python -m experiments.knowledge_tracing.data_adapter
uv run --project experiments/knowledge_tracing --locked python -m experiments.knowledge_tracing.train
uv run --project experiments/knowledge_tracing --locked python -m experiments.knowledge_tracing.evaluate
```

Or run all three stages with `python -m experiments.knowledge_tracing.benchmark`. Preparation, selection and test opening are sealed against accidental overwrite. For a new experiment, copy `configs/pilot.json`, freeze its changes, and use a fresh output directory:

```powershell
uv run --project experiments/knowledge_tracing --locked python -m experiments.knowledge_tracing.benchmark --config experiments/knowledge_tracing/configs/pilot.json --output experiments/knowledge_tracing/artifacts/repeat --offline
```

This last command needs both dependencies and range caches already populated. Add uv's `--offline` before `python` as well to prohibit dependency resolution over the network. If rebuilding an environment from the workspace dependency cache, use `uv --cache-dir .uv-cache sync --project experiments/knowledge_tracing --locked --offline --python 3.11`. `data_adapter --offline` never opens a network connection. The first fetch requires access to Hugging Face. A server that ignores HTTP Range causes an explicit error, never an automatic full-file download.

`data_adapter --learners 100` configures the learner sample. Training accepts `--batch-size 128`, `--epochs 8`, and `--sequence-length 32` for smaller resource budgets. The JSON config controls all three sequence lengths, sample/split seed, early stopping and CPU threads. Keep preparation and training configs consistent. The supplied run uses all 100 learners and all their available interactions, without capping attempts per learner.

After a completed evaluation, reproduce metrics without changing choices:

```powershell
uv run --project experiments/knowledge_tracing --locked --offline python -m experiments.knowledge_tracing.evaluate --replay --output experiments/knowledge_tracing/artifacts/pilot/evaluation_replay
```

Replay validates data manifest, source and every model artifact hash. A changed model/source requires a fresh run; the original test set is then exploratory and cannot be presented as newly unseen. Sealing is a local procedural safeguard, not cryptographic access control. Aggregate split statistics are produced during data audit; training opens only train/validation records.

## Objective and observation boundary

For each learner transition, history ends at completed attempt `t`; the target is the stored `correct` at `t+1`. Each learner's first attempt supplies history and is not a prediction target. Later outcomes of held-out learners enter their own subsequent histories, which is normal online knowledge tracing, with no gradient update or learner embedding.

The original source partitions reconstruct chronology. The new learner assignments define this benchmark's train/validation/test roles: a training learner contributes all of their original source partitions, including original source-test interactions. No held-out benchmark learner contributes to fitting.

Target context is allowlisted to **exercise `format` and `session` type**. The source paper's exercise UI and session description establish these categories before answering. Target `days` is excluded because the release does not independently establish whether its recorded instant precedes submission. Thus the time-since-previous feature is the incoming gap of the most recently **completed** exercise, not the interval to the target's possibly post-answer timestamp. A future dataset with verified exercise-start timestamps can support that additional feature.

Target correctness, token labels, token accuracy, response time, reference tokens, token/exercise identifiers and all aligned linguistic annotations are excluded. Prompts, country and client are also unused. The model never sees an original or encoded learner ID. IDs are used to group, preserve order, select/split learners and key reproducible perturbation randomness only.

## Shared features and preprocessing

- Each historical event supplies correctness, log response seconds, log incoming gap seconds, exercise format and session type. Zero response time is distinct from missing response time.
- Common summary features supply mean accuracy over the available window, recent 16/4-event accuracies, last correctness, recent-half minus older-half accuracy, linear trend, mean/last log response, mean/last log gap, and log number of preceding interactions.
- Numeric medians, means and standard deviations are fit on **training learners only**. Missing numbers use training medians plus explicit missing indicators. Categories have separate missing and unseen slots. Training summary statistics use causal windows up to 64 events; no validation/test preprocessing is fitted.
- Recurrent input has no absolute timestamp, local interaction index or per-event position counter. Both recurrent heads also receive the same summaries/context as logistic regression. This asks whether ordered events add useful information beyond engineered summaries.
- Windows have maximum lengths 16/32/64. Valid events are right-padded; `lengths` defines a padding mask and `pack_padded_sequence` prevents padding from affecting the recurrent state. Empty histories after simulated loss use a zero recurrent state and missing summaries.

## Models and selection

The statistical model uses `(historical successes + strength * training prior) / (observed labels + strength)`. The candidate histories are expanding/full, 16, 32 and 64; strengths are 2/10/30. No-history estimates revert to the training global prior. Logistic regression shares the summary/context representation, with C=0.1/1/10.

GRU and LSTM have one recurrent layer, hidden size 32 and a linear binary head. Training uses BCEWithLogitsLoss, AdamW (lr 0.003, weight decay 0.0001), gradient clipping at 1, batch size 256, at most 12 epochs and patience 3. The best checkpoint and window are selected by validation Brier score; probability outputs use sigmoid. Initialization, minibatch shuffle and perturbation seeds are fixed at 42; deterministic PyTorch algorithms and two CPU threads are enabled.

A summary-only ReLU MLP approximately matches the selected GRU parameter count. Its hidden size is computed from that count, its window follows the validation-selected GRU, and its epoch is selected on validation. It controls for added nonlinear capacity without an event sequence. This does not exactly match optimization or inductive bias.

## Evaluation protocol

The primary models are selected separately by minimum validation Brier. The statistical/logistic reference for paired comparisons is also chosen on validation. Test is opened once after main models **and all retrained controls** are frozen and hashed. Accuracy uses a fixed 0.5 threshold; no test threshold or calibration fitting occurs.

Report Brier, binary log loss, ROC-AUC, PR-AUC summarized by **average precision**, accuracy, precision, recall, F1, confusion matrices and ten equal-width calibration bins/ECE. Undefined one-class ROC/AP slices are null. Learner-macro scores weight learners equally; paired Brier intervals for the primary baseline, retrained ablations and summary-MLP comparisons resample entire learners (2,000 draws). Negative differences favor the ordered recurrent model. Only 15 test learners means substantial uncertainty; many overlapping windows are not independent replicates. Intervals are descriptive and do not correct for multiple comparisons.

Both diagnostics are reported:

1. Retrain at each 16/32/64 window; report the expanding statistical history separately. “Full” neural history means the selected bounded window, not an unbounded recurrent state.
2. Retrain last-only, shuffled, no-correctness, no-time and no-metadata controls at each architecture's already selected window and main hyperparameters. Early stopping still uses validation.

Additional inference-only tests perturb the selected chronological model with last-only, full event-bundle shuffling, and random loss of 20%/50% of historical events. Gaps travel with their historical event; they are never recomputed as negative intervals after shuffling. Rolling/trend summaries are recomputed from the visible window, so original chronological summaries cannot bypass the shuffle. Historical incoming gaps still carry temporal context: shuffling is not removal of every temporal signal. Missing-event tests remove bundles within the selected window without refilling it and preserve the source-derived prior-attempt count; this assumes the count is known from separate telemetry. These are synthetic stress tests, not measured logging-failure distributions. Last-only retains the attempt count and latest event's incoming gap; it means one observed event plus those contextual statistics.

Short/medium/long history slices use 1–15, 16–63 and 64+ prior attempts. Failures are aggregated by history bucket, exercise format and confident errors; reports contain no learner IDs, literal answers or individual predictions.

## Outputs, resources and future interface

`artifacts/pilot/data/` contains local JSONL and the sealed manifest; `artifacts/cache/` stores pinned evidence and requested Parquet ranges; `artifacts/pilot/models/` contains encoders, candidate checkpoints, learning curves and selection hashes. `artifacts/pilot/evaluation/` contains full metrics and plots. All caches, raw/prepared records, checkpoints and weights are Git-ignored. Only aggregate statistics, report text and diagnostic charts are copied to `results/`.

Whole-process RSS is sampled every 20 ms. It includes interpreter, libraries, data and allocator caching, can miss brief peaks, and is not an isolated model footprint. Serialized predictor bytes and parameter counts are separate. Inference median/p95 use warmed, single-query CPU prediction including history encoding, excluding startup and I/O. Latency varies with complete history length because the adapter constructs historical gaps before truncation. Offline preprocessing replay is checked against original split hashes; exact cross-version/cross-machine training bitwise identity is not promised.

```python
from experiments.knowledge_tracing.models import FittedPredictor
# Load only trusted local artifacts after checking selection hashes.
model = FittedPredictor.load("experiments/knowledge_tracing/artifacts/pilot/models/<selected-id>")
probability = model.predict(history, {"format": "listen", "session": "practice"})
```

Every architecture implements `predict(history, next_exercise_context) -> probability`; batched `predict_windows` shares evaluation. A later Mamba implementation can use the same contract, masks, split and metrics. No Mamba, OPD, GRPO, RLHF, LoRA/QLoRA, distillation or personal-data fine-tuning is introduced here. Improved correctness prediction would not establish better productivity, accountability, learning gains or causal intervention benefit.

## Tests

```powershell
uv run --locked --extra research python -m pytest tests experiments/router/test_audit.py -q
uv run --project experiments/decision_model --locked python -m pytest experiments/decision_model/test_benchmark.py -q
uv run --project experiments/knowledge_tracing --locked python -m pytest experiments/knowledge_tracing/tests -q
```

The research tests use synthetic fixtures only. Neither real learner records nor model weights belong in Git.

## Attribution

Data copyright (c) 2016 Duolingo, Inc.; [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), without warranties. Source: [Data for the 2018 Duolingo Shared Task on SLAM](https://doi.org/10.7910/DVN/8SWHNO). Conversion: [bihungba1101/slam-en-es-knowledge-tracing](https://huggingface.co/datasets/bihungba1101/slam-en-es-knowledge-tracing/tree/44c6826ebbfb7ac7f0af5d181b5f1a65cf4f5304). This experiment changes grouping, splits and feature representation and derives research aggregates; it does not redistribute raw data or imply Duolingo endorsement. The source's noncommercial restrictions remain applicable; this is not authorization for commercial deployment.

Burr Settles, Chris Brust, Erin Gustafson, Masato Hagiwara and Nitin Madnani. 2018. [Second Language Acquisition Modeling](https://aclanthology.org/W18-0506/). Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications, pp. 56–65. ACL. Paper DOI: 10.18653/v1/W18-0506; data DOI: 10.7910/DVN/8SWHNO.
