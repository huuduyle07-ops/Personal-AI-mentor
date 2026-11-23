# Offline router experiment audit

2026-11-22 KST. Production code, RuleRouter, the original generator/baseline/runner, dependencies and lockfile are unchanged. Only synthetic inputs were used; no Discord records, credentials or personal database were accessed. No sequence model, OPD or GRPO was introduced.

## Reproduce

From the repository root, using the existing lockfile (Python 3.11.9, scikit-learn 1.9.1):

```powershell
uv sync --locked --extra research
uv run --locked --extra research python -m experiments.router.run --seed 42 --output experiments/router/artifacts/audit/reproduction
uv run --locked --extra research python -m experiments.router.audit
uv run --locked --extra research python -m pytest experiments/router/test_audit.py tests/test_experiment.py -q
```

Use `uv --cache-dir .uv-cache ...` if the default cache is restricted. After dependencies are installed, add `--offline` after `--locked` to run without package-network access. The audit writes `protocol.json`, `selection.json` and full `metrics.json` under `experiments/router/artifacts/audit/comparison/`. The protocol is written before training; all seeds' hyperparameters are selected and saved before the learned-model test phase. [audit_results.json](audit_results.json) preserves the measured snapshot, source hashes, seed-42 diagnoses and per-seed results. Original outputs were retained in separate directories.

Validation: **86 tests passed**, including 9 focused audit/experiment tests; one upstream discord.py `audioop` deprecation warning. The full suite used mocked services and ran outside the Windows sandbox because it blocks asyncio initialization. No production files changed during the comparison, verified by source hashes and working-tree scope.

## Observations: reproduction and validation collapse

The original run reproduced: RuleRouter test accuracy **1.0000**; serialized character-TF-IDF/logistic-regression test accuracy **0.6250**, macro-F1 **0.5595**, LLM recall **0.0000**. Validation accuracy is **0.2500**, macro-F1 **0.1000**. All C candidates (0.5, 2, 8) give the same validation predictions; the documented tie-break selects C=0.5.

Every confusion matrix below has true rows and predicted columns in the order **RULE, CLARIFY, LLM, DEFER**.

Original seed-42 validation:

```text
             RULE  CLARIFY  LLM  DEFER
RULE            0      20    0      0
CLARIFY         0      20    0      0
LLM             0      20    0      0
DEFER           0      20    0      0
```

All 80 rows are predicted CLARIFY. Its precision is 20/80=0.25, recall 1.0 and F1 0.40; other classes have F1 zero. Macro-F1 is therefore (0+0.40+0+0)/4=**0.10**, not a metric calculation bug. The four validation families are `schedule_control`, `excess_minutes`, `work_conflict` and `later_planning` (20 numerical variants each). The mixed model's validation matrix is diagonal with 20 in each class: accuracy/F1 **1.0/1.0**. This is only four family-level validation cases and does not predict similarly strong test performance.

Seed-42 test matrices:

```text
Serialized                 Mixed ColumnTransformer
[20, 20,  0,  0]           [20, 20,  0,  0]
[ 0, 40,  0,  0]           [20,  0, 20,  0]
[ 0, 40,  0,  0]           [ 0,  1, 39,  0]
[ 0,  0,  0, 40]           [ 0,  0,  0, 40]
```

The mixed model has accuracy **0.61875**, macro-F1 **0.57197**, LLM recall **0.975**, and CLARIFY recall **0.0** on this seed. Its improved LLM recall comes with complete loss of correct CLARIFY predictions; its seed-42 accuracy is slightly worse than the original.

## Observations: splitting, labels and effective sample size

- There are **24 authored scenario-template families**, six per class; 20 numerical substitutions each create 480 rows. Each split has **12/4/8 families**, corresponding to 240/80/160 rows. These are 24 distinct prototypes, not 480 independent semantic scenarios or a statistically independent sample of users. All seeds reuse the same pool; ten seeds do not create 240 new families.
- All three split pairs have zero overlap in family IDs, exact text, encoded inputs and text after replacing digit runs with a placeholder. These checks pass for all ten seeds. Labels match the declared family labels and RuleRouter on every generated example. Vectorizer vocabulary, IDF, one-hot categories, imputation statistics and scaling are fitted on training data only.
- No within-run split leakage was found by these checks. Repeated metadata across families is legitimate input, not future information. However, labels encode the existing deterministic contract, so perfect RuleRouter accuracy is expected by construction; it is not independent evidence of correct real-world escalation. There is no independently adjudicated natural-language gold standard.
- Only **13 distinct routing signatures** occur when counting command, minutes, non_urgent, priority and short/non-short text. Every priority is GENERAL; URGENT/SYSTEM and many interaction/invalid-field combinations are untested. Related prototypes and their causal signals overlap, so 24 is an upper bound on distinct semantic cases, not a guarantee of statistical independence.
- Label caveat: all 20 `report_total` texts interpolate a random number as the reported duration while metadata stays at 30 minutes. All 20 disagree in seed 42. Labels are consistent with structured-input precedence but the text/metadata combination is semantically contradictory. Do not interpret the generated text as clean independently labeled duration evidence.
- Split reproducibility requires **both seed and samples_per_family**. The same RNG draws numeric variants and then shuffles later classes, so changing the count from 20 to 2 changes family assignments even at seed 42. Future versions should separate split and augmentation RNG streams. The audited comparison fixes the count at 20 throughout.

## Observations and hypotheses: representations and failures

The original encoder concatenates `command=... minutes=... non_urgent=... priority=... | text`, converts None/booleans/numbers to strings, then applies character-within-word TF-IDF (3–5-grams, sublinear term frequency, default L2 normalization). Numbers have no ordinal/range meaning; metadata and prose share ngram features and one normalization. `family` and `label` are excluded.

The alternative in [mixed_baseline.py](mixed_baseline.py) keeps the **same raw fields**: identical character TF-IDF on text alone; OneHotEncoder on command/non_urgent/priority; median imputation with a missing indicator and retained empty columns, then StandardScaler on minutes. Missing commands use an explicit category. Unseen categories are ignored (all-zero encoding). No policy allowlist, validity flag, word-count feature or numerical threshold was added. Logistic regression and the C grid are unchanged. Fitting these transformers inside the training pipeline prevents preprocessing leakage. See the official [ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) and [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) documentation.

**Measured diagnostics:**

- **LLM → CLARIFY:** all 40 seed-42 test examples in `competing_tasks` and `reflection` fail. Training has `terse_help` labeled CLARIFY. The `help`-related character ngrams contribute against LLM in those longer prompts. An exact linear score decomposition for one `competing_tasks` example gives true-minus-CLARIFY contributions: intercept −0.1696, metadata −0.4256, text +0.4893, total **−0.1059**. Text favors LLM overall but does not overcome the other terms. The snapshot includes probabilities and exact per-example attributions; probabilities are not calibrated.
- **RULE → CLARIFY:** all 20 `report_total` examples fail in both representations. Seed-42 training only sees `report` with missing or −1 minutes, never a valid report. Of 240 training rows, 220 have missing minutes and 20 have −1; no other numeric value is observed. Imputed numeric training values are consequently constant, so scaling cannot create missing range evidence. The mixed model cannot learn that 30 is valid from these examples. In the serialized example, metadata contributes **−1.2512** to the RULE-minus-CLARIFY margin. The other held-out RULE family, `recovery_approval`, succeeds.
- **DEFER → CLARIFY in validation:** `later_planning` fails despite non_urgent=true. In one example, metadata supports DEFER (+0.3696) but text (−0.3166) and intercept (−0.2475) reverse the result. The mixed representation recognizes this explicit flag on every test seed.
- **New mixed-model failures:** seed-42 `terse_done` becomes LLM (20/20) and `unknown_command` becomes RULE (20/20). OneHotEncoder has never seen `invented`; it also has not seen the supported command `adjust` (or validation `checkins`). All are unknown categories rather than semantic support/unsupported-command knowledge. This representation does not learn the missing command-validity contract or a reliable short-text boundary.

**Hypotheses, not proven causal conclusions:** short repetitive CLARIFY templates strengthen shared metadata/help associations under joint normalization; separating metadata reduces this interference. The linear attributions support an association, but do not isolate which design choice causes the aggregate improvement. The mixed approach changes categorical handling, numerical meaning and normalization together; individual effects require a future predeclared ablation.

## Measured comparison across fixed seeds

Seeds: **0, 1, 2, 3, 4, 42, 73, 101, 2026, 2027**. Same family splits for both models within each seed; C chosen only by validation macro-F1, ties favor smaller C. No refitting on validation or test occurs. All models are frozen before learned-model test scoring. RuleRouter accuracy and macro-F1 remain 1.0 on both splits for every seed.

| Representation | Validation macro-F1, mean ± sample SD | Test accuracy, mean ± SD | Test macro-F1, mean ± SD |
|---|---:|---:|---:|
| Serialized TF-IDF | 0.4572 ± 0.2101 | 0.5831 ± 0.1441 | 0.5364 ± 0.1516 |
| Mixed ColumnTransformer | 0.6378 ± 0.2284 | 0.7244 ± 0.1650 | 0.7122 ± 0.1720 |

| Class | Serialized mean test recall | Mixed mean test recall |
|---|---:|---:|
| RULE | 0.2675 | 0.6000 |
| CLARIFY | 0.9000 | 0.4500 |
| LLM | 0.3275 | 0.8475 |
| DEFER | 0.8375 | 1.0000 |

Mean test macro-F1 rises by **0.1758**, and accuracy by **0.14125**. Macro-F1 improves on all ten seeds; accuracy improves on nine. CLARIFY recall falls substantially. Pooled matrices count **1,600 repeated evaluation rows**, not independent new scenarios:

```text
Serialized                 Mixed ColumnTransformer
[107, 280,  13,   0]       [240, 160,   0,   0]
[ 40, 360,   0,   0]       [140, 180,  80,   0]
[ 60, 203, 131,   6]       [  0,  61, 339,   0]
[ 40,   5,  20, 335]       [  0,   0,   0, 400]
```

These are measured diagnostic improvements on this synthetic pool, **not a production recommendation or a confirmatory unseen-benchmark claim**. The original test set was already public, and families recur across seeds; standard deviations combine split and numerical-augmentation sensitivity, not confidence intervals from independent samples. Representations and seeds were fixed before the alternative's outcomes were observed and no test result was used to select C or revise either model. All 24 families appeared in at least one test split. A future claim needs newly authored, sealed test families.

## Recommendations: separate temporal Decision Model benchmark

1. **Define a different task.** This router uses current-event fields only; no example requires memory. Define a timestamped history/state contract and a finite set of proposed decisions, with authorization and action execution kept separate. Include evidence provenance, missingness, outstanding questions and prior approvals. Never use missed replies as labels for dishonesty or failure.
2. **Make history necessary.** Create paired synthetic episodes with the same current input but different past verified/reported states and different justified next proposals. Label with explicit simulator rules plus independent review, not the current router's output. Keep future outcomes and target actions out of input features. Cover recovery rejection, stale approvals, uncertainty and bounded follow-ups.
3. **Split whole episodes and generator families.** Keep personas, shared prefixes, scenario templates and derived variants in one partition; preserve chronological order and compute summaries from past events only. Separate split and augmentation seeds. Freeze a fresh sealed test set; use grouped training/validation selection before opening it. Report unique families/episodes as well as row counts, and separate ordinary held-out phrasing from deliberately unseen commands or numeric ranges.
4. **Establish simple baselines first.** Compare fixed rules, current-event linear/tree models, and explicit history-summary models. Use no-history and shuffled-history ablations to test whether order/history adds value. No LSTM, Mamba, OPD or GRPO is needed yet.
5. **Measure the right errors.** Report per-action/per-family F1 and recall, missed clarification/escalation, permission violations (required zero), stale-proposal handling, latency/memory and interaction burden. Assess confidence calibration only on separate validation data. Synthetic outcome improvements are simulator results, not evidence of benefit with real users.

For the current routing benchmark, first correct conflicting text/metadata and cross valid/invalid minutes with multiple command and phrasing families. Include 0/1440 boundaries, unsupported commands, short/long inputs and priority/urgency combinations. More numbered copies cannot fix absent semantic coverage. Keep RuleRouter in production while a new benchmark is independently reviewed.
