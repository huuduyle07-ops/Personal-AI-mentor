# Temporal Decision Model benchmark v0.1

An isolated, synthetic **shadow-mode probability benchmark**, not an action policy. It asks whether a one-layer LSTM improves future outcome prediction over historical rates and logistic regression with strong historical summaries. It does not import production modules, read SQLite or Discord records, send messages, change goals/schedules, trigger recovery, or modify approvals. No Mamba dependency, Mamba training, OPD, GRPO or RLHF is included.

## Reproduce in the isolated CPU environment

From the repository root, with Python 3.11.9 and uv:

```powershell
uv sync --project experiments/decision_model --locked
uv run --project experiments/decision_model --locked python -m experiments.decision_model.prepare
uv run --project experiments/decision_model --locked python -m experiments.decision_model.train
uv run --project experiments/decision_model --locked python -m experiments.decision_model.evaluate
uv run --project experiments/decision_model --locked python -m pytest experiments/decision_model/test_benchmark.py -q
```

The experiment has its own `pyproject.toml`, `uv.lock` and `.venv`. PyTorch comes from the explicit CPU wheel index; root dependencies and lockfile are unchanged. Add `--cache-dir .uv-cache` immediately after `uv` for a workspace-local cache. Once dependencies are installed, add `--offline` after `--locked` for offline runs. All default data/models/evaluation outputs go under ignored `experiments/decision_model/artifacts/`.

Preparation refuses to overwrite a sealed manifest. Training refuses to overwrite a completed selection. Evaluation opens test once and writes `test_opened.json`; `--replay` permits an unchanged-model reproduction with matching hashes. For a clean repeat, pass new directories to `prepare --output`, `train --data ... --output ...`, and `evaluate --data ... --models ... --output ...`. Do not retune from test results and present the same set as unseen afterward.

The original application and router tests use the root environment:

```powershell
uv run --locked --extra research python -m pytest tests experiments/router/test_audit.py -q
```

## Targets, observation time and provenance

Every input is exactly 14 complete chronological KST days. The cutoff is 23:59:59 on the last historical day; forecasts cover the **next calendar day's 18:00–23:00 KST reporting window**, inclusive. Reports outside that window are recorded separately and are not valid in-window reports.

- **A: report availability.** One if a valid report arrives in that future window; zero otherwise.
- **B: reported goal completion, conditional on A=1.** One if reported minutes reach that day's effective goal after any user-approved adjustment. Its loss and metrics are masked completely when A=0. A zero-minute valid report has A=1, B=0. A missing report has B unobserved, never failed. Stored B=0 under a zero mask is only a serialization placeholder.

Every event has an explicit timestamp, kind, source and payload. Sources distinguish declared facts, approval/rejection and quiz results; quiz performance is not proof of duration. Features include declared and effective goals, reported minutes/ratio, report availability, conflicts/obstacles, decisions, verification, routine changes, weekday and outside-window reports. Each channel has a separate observation mask, and the encoder's feature schema documents provenance. No persona/family ID, next-day event, target or simulator oracle probability is encoded. Unknown context/quiz/minutes are masked, not inferred as negative observations.

Numerical means/scales are fit using observed training cells only; missing normalized values are zero with mask zero. Logistic summary scaling is also fit on training rows only (B on its observed rows). Encoder artifacts are saved and hash-checked during evaluation.

## Simulator, split and independent units

Twelve newly authored temporal mechanisms describe steady cadence, alternating duties, weekly peaks, deadline ramps, interruption blocks, relocation, weekend obligations, rotating shifts, obligation pulses, late reporting, persistent workload and study cycles. They share a common stochastic outcome equation; this is not twelve independently validated behavioral theories.

Hidden persona reporting/completion propensities, recent reporting/completion history and today's exogenous circumstances generate outcomes causally. Future conflicts and adjustments are not available to the forecaster. A separate numeric stream changes reported amounts without changing labels. Routines change at specified points; conflicts/obstacles may be reported or unknown; adjustments can be approved or rejected; reports can be absent, outside-window or explicitly zero; quiz events are distinct observations.

`config.json` freezes independent seeds for family splitting, persona traits, behavioral outcomes, numeric augmentation, history shuffling and telemetry loss. All streams are keyed by stable IDs, so adding personas or numeric draws does not move family assignments or change existing personas.

| Partition | Scenario families | Personas | Natural episodes | Overlapping 14-day samples |
|---|---:|---:|---:|---:|
| Train | 6 | 144 | 288 | 4,032 |
| Validation | 3 | 72 | 144 | 2,016 |
| Sealed test | 3 | 72 | 144 | 2,016 |

The twelve whole families and all their personas/episodes/variants are split 6/3/3 before outcomes are drawn. Episodes are 28 days, producing 14 windows each. Windows overlap only within a partition; multiple episodes share their persona and are not independent people. The 8,064 natural samples are not 8,064 independent trials. Person-level grouping alone is insufficient, so family-level grouping is also enforced.

Each persona also has a matched high-/low-history diagnostic pair: 288/144/144 derived episodes across train/validation/test, one window per branch. Both branches have exactly identical latest-day observable events but different preceding histories. They use common future random draws; simulator probabilities are retained only for evaluation. These are controlled synthetic history interventions, not independent natural episodes. No paired data is used in training or hyperparameter selection; evaluation uses only the held-out pairs (72 matched pairs).

The newly generated test files and hashes are sealed during preparation. Training opens only train/validation. All model selections and model artifact hashes are written before evaluation loads test. This is a process-level embargo, not encryption: authors can inspect source and the generation rules. Future confirmatory claims require fresh independently reviewed families.

## Models and frozen controls

- **Statistical probabilities:** training prevalence, or Beta-smoothed empirical report/completion rates over the last 1, 7 or 14 days. B rates count only observed valid reports. Smoothing uses two pseudo-observations at the training prior.
- **Aggregated logistic regression:** separate A/B classifiers with 14-/7-/3-day observed means, last-day values, observed fractions, linear trends and latest-day masks. This is deliberately a strong summary comparator. C candidates are 0.1, 1 and 10; selection uses mean validation head log loss.
- **One-layer LSTM:** hidden size 16 or 32, Adam lr 0.01, batch size 256, maximum 30 epochs, gradient clipping 1. CPU threads=1 and deterministic algorithms enabled. Best hidden size and epoch are chosen by validation log loss only. Initialization seeds are 7, 19 and 43 on the same dataset split.
- **Multi-task vs separate:** a shared LSTM with two output heads is compared with two independent one-head LSTMs, each selected on its own validation head loss. Separate models have more combined parameters; artifact sizes are reported. `BCEWithLogitsLoss(reduction="none")` is masked before per-head normalization, with equal active-head weighting. Missing B contributes zero loss and zero gradient.

All learned model types are trained in chronological, no-history (latest day only), and shuffled-history modes. Shuffling permutes the first 13 days and preserves the latest day, masks and event-feature bundles; it uses no labels. Retrained shuffled models test performance with order removed; additional test-only shuffling of chronological models is a perturbation diagnostic, not a retrained baseline. Weekday remains an observable feature, so shuffling is not a proof that all temporal information has been erased.

`SequencePredictor.predict_proba(histories)` is the future model interface, returning only two probabilities. A future Mamba implementation can satisfy this interface without being installed or trained now. There is no executor interface or production hook.

## Evaluation and artifacts

Evaluation records Brier score, binary log loss and average precision (AP, the chosen PR-AUC summary; undefined for a one-class slice), plus ten fixed-bin reliability tables, ECE, mean confidence and observed rates. Calibration is descriptive; no post-test calibration is fitted. B always uses the original valid-report mask.

Robustness removes historical report-related telemetry on 20% or 50% of past days, preserving the latest day and labels. Missing channels become unknown with explicit mask zero. This is a defined synthetic MCAR stress test, not a model of every real missingness process. Native missing context/report patterns remain present in all conditions.

Paired analysis reports probability gaps and directional agreement against simulator probability gaps when latest-day inputs are identical. Per-family scores accompany global scores. Model sizes are serialized artifact sizes; single-window inference timing includes array normalization, summary computation/network inference, but excludes event JSON parsing, I/O and cold startup. CPU RSS is sampled every 20 ms and includes interpreter, libraries, datasets and allocator caching; it is not isolated model memory.

Files:

| File | Purpose |
|---|---|
| `simulator.py`, `data.py`, `prepare.py` | Chronological generation, independent RNGs, grouped splitting, sealing |
| `features.py` | Mask/provenance-aware encoder, strong summaries, ablations and missingness tests |
| `models.py` | Statistical estimates, logistic heads, shared/separate LSTM and future protocol |
| `train.py` | Train/validation-only model/checkpoint selection and source/artifact hashes |
| `metrics.py`, `evaluate.py` | Conditional metrics, calibration, robustness, paired analysis, timing/RSS |
| `test_benchmark.py` | Temporal cutoff, grouping, masking, controls, sealing and round-trip tests |
| `config.json`, `pyproject.toml`, `uv.lock` | Frozen protocol and isolated dependencies |

Generated artifacts include `manifest.json` and synthetic JSONL, encoder state, trusted-local joblib and PyTorch checkpoints, protocol/selection hashes, test opening marker, full evaluation JSON and report. Never load arbitrary external joblib files. PyTorch checkpoints load with `weights_only=True` and matching artifact hashes.

## Results and limits

The [measured report](results/REPORT.md) and [full result snapshot](results/results.json) record the frozen evaluation, including failures. Full-history separate LSTMs score A/B Brier 0.1967/0.2190 versus 0.1977/0.2105 for 7-day historical rates. History helps, but there is no clear overall LSTM advantage. Multi-task sharing worsens both full-history head means. Missing telemetry weakens LSTM calibration, and logistic paired/test-only-shuffle controls expose a low-variance scaling/support-shift failure; these are documented, not tuned away on the exposed test set. See the report before interpreting any apparent model advantage.

All outcomes are synthetic and reflect shared simulator assumptions. Predictions are reporting/completion probabilities, not judgments of intent, honesty, wellbeing or personal worth. Missing reporting is not goal failure. This benchmark cannot establish real-user behavioral benefit and authorizes no production behavior changes.
