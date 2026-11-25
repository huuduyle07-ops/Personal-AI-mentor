# Temporal Decision Model v0.1: measured report

2026-11-22 KST. **History helps; this run does not establish an overall LSTM advantage over simple historical estimates.** Separate LSTMs improve report prediction slightly but lose on conditional reported-goal completion. No production integration is justified by these synthetic results.

## Frozen protocol and independent units

A = valid report in next day's 18:00–23:00 KST window. B = reported minutes reach that day's approved effective target, conditional on A=1. B loss/metrics are masked when no valid report arrives; a valid zero-minute report remains an observed non-completion. Verification is separate from duration. The cutoff includes exactly the preceding 14 complete days.

The dataset has **12 authored temporal mechanisms, 288 personas, 576 natural episodes and 8,064 overlapping natural windows**. The train/validation/test allocation is 6/3/3 mechanisms, 144/72/72 personas, 288/144/144 natural episodes and 4,032/2,016/2,016 windows. All persona, episode and derived-group identities are partition-disjoint. Additional pairs are derived diagnostics, not independent natural episodes: the held-out set contains **72 matched pairs / 144 branches**. Paired train/validation data is not used for fitting or selection.

All 25 model variants were frozen using training/validation only before the first test opening. Logistic C=0.1/1/10; LSTM hidden=16/32 and best of 30 epochs; three initialization seeds 7/19/43. The statistical 7-day model has the best mean validation head log loss among the four statistical candidates. Its role as reference does not depend on the test ranking. The 14-day candidate is also shown as predeclared descriptive evidence.

The test has 2,016 A labels, **1,148 observed B labels**; positive rates are 0.5694 and 0.4233. No test outcome revised the simulator, features, models or hyperparameters. A hash-checked replay evaluates identical frozen artifacts; it is reproduction, not a second independent test.

## Main measured results

Lower Brier/log loss is better. AP is average precision, not trapezoidal PR area. LSTM values are means over three initialization seeds on one grouped dataset split.

| Model | A Brier | B Brier | A log loss | B log loss | A AP | B AP | A ECE | B ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| statistics_0 | 0.2453 | 0.2469 | 0.6838 | 0.6869 | 0.5694 | 0.4233 | 0.0129 | 0.0526 |
| statistics_1 | 0.2166 | 0.2371 | 0.6245 | 0.6670 | 0.6755 | 0.4961 | 0.0112 | 0.0343 |
| statistics_7 | 0.1977 | 0.2105 | 0.5818 | 0.6111 | 0.7765 | 0.6302 | 0.0348 | 0.0350 |
| statistics_14 | 0.2031 | 0.2091 | 0.5923 | 0.6095 | 0.7758 | 0.6565 | 0.0454 | 0.0359 |
| logistic_full | 0.1976 | 0.2220 | 0.5810 | 0.6352 | 0.7919 | 0.5961 | 0.0177 | 0.0405 |
| lstm_joint_full | 0.1983 | 0.2207 | 0.5825 | 0.6325 | 0.7957 | 0.6075 | 0.0415 | 0.0575 |
| lstm_separate_full | 0.1967 | 0.2190 | 0.5792 | 0.6291 | 0.7904 | 0.5998 | 0.0359 | 0.0494 |

Compared with the validation-preferred 7-day statistics baseline, separate LSTMs reduce A Brier by only **0.00103** but increase B Brier by **0.00849**. Their mean two-head Brier is worse (0.20784 vs 0.20411). Both learned approaches improve over the no-history global prior, but that does not establish a sequence-model advantage.

**Multi-task sharing worsens full-history means:** joint vs separate Brier is 0.19831 vs 0.19669 for A and 0.22067 vs 0.21899 for B. Log loss also worsens. This is an average result, not universal: seed 43's joint B head beats its separate counterpart. Full-history Brier sample SD across initialization seeds is:

- lstm_joint_full: A 0.00171; B 0.00599.
- lstm_separate_full: A 0.00079; B 0.00141.

Per-family Brier (A / B), averaging LSTM initialization seeds:

| Held-out family | 7-day statistics | Logistic summaries | Joint LSTM | Separate LSTM |
|---|---:|---:|---:|---:|
| interruption_blocks | 0.1863 / 0.2083 | 0.1828 / 0.2179 | 0.1808 / 0.2226 | 0.1817 / 0.2212 |
| persistent_workload | 0.2020 / 0.2127 | 0.2044 / 0.2222 | 0.2041 / 0.2215 | 0.2019 / 0.2199 |
| weekend_obligations | 0.2049 / 0.2106 | 0.2054 / 0.2264 | 0.2100 / 0.2176 | 0.2065 / 0.2154 |

The small report-head advantage is not uniform across mechanisms. The statistical reference beats both LSTM variants on B in all three held-out families. Three families and three optimization seeds do not support a statistical-significance or general population claim.

## History controls and paired diagnostics

| Training/input condition | Joint A/B Brier | Separate A/B Brier | Logistic A/B Brier |
|---|---:|---:|---:|
| full | 0.1983 / 0.2207 | 0.1967 / 0.2190 | 0.1976 / 0.2220 |
| no_history | 0.2177 / 0.2412 | 0.2174 / 0.2412 | 0.2169 / 0.2432 |
| shuffled | 0.2018 / 0.2193 | 0.2021 / 0.2233 | 0.2023 / 0.2212 |

Historical inputs improve both heads over latest-day-only inputs. Retrained shuffled-history models worsen A, but B is mixed: joint shuffled slightly beats joint chronological B. Thus ordering has a modest, inconsistent effect beyond historical contents. Test-only shuffling of chronological separate LSTMs yields A/B Brier 0.2011/0.2218, compared with native 0.1967/0.2190. Weekday features survive shuffling and can retain some timing information.

On 72 matched pairs with exactly identical latest-day observations, full-history LSTMs and historical rates assign the correct high/low probability ordering in all pairs for both heads. No-history models give exactly zero gap, as expected. Simulator mean high-minus-low probability gaps are A=0.2955, B=0.2160. Joint LSTM gaps are 0.2785/0.3512; separate LSTM gaps are 0.3707/0.3428; 7-day statistics gaps are 0.5556/0.5345. Direction is learned, but estimated effect magnitudes are not uniformly accurate; history rates also solve this check.

**Recorded failure, not evidence of neural superiority:** logistic paired direction is only 20.8% and test-only shuffling yields extreme log loss (~6.40 A / 6.70 B). A subsequent read-only training-data diagnostic found near-constant summary features scaled by roughly 1e-9 to 1e-8. Shuffling breaks the constant chronological seven-day weekday mean; paired current-day goal resets to 30 can break a constant training goal trajectory. These out-of-support features become enormous standardized inputs and saturate logistic probabilities. Paired comparisons therefore also contain an unintended support shift for some personas. This weakens that control as a pure model-capability comparison. No corrective preprocessing or paired-generation change was made after test exposure. A fresh benchmark revision should guard low-variance summaries using training-only criteria and keep each pair's base goal unchanged, with new sealed evaluation cases.

## Missingness and calibration failures

Past report telemetry is hidden on 20%/50% of historical days; latest-day input and future target masks stay unchanged. Unknown channels carry mask zero, not observed no-report/failed-goal labels.

| Model | Native A/B Brier | 20% loss A/B | 50% loss A/B |
|---|---:|---:|---:|
| statistics_7 | 0.1977 / 0.2105 | 0.2019 / 0.2114 | 0.2043 / 0.2198 |
| logistic_full | 0.1976 / 0.2220 | 0.2021 / 0.2225 | 0.2076 / 0.2311 |
| lstm_joint_full | 0.1983 / 0.2207 | 0.2067 / 0.2216 | 0.2359 / 0.2246 |
| lstm_separate_full | 0.1967 / 0.2190 | 0.2036 / 0.2221 | 0.2295 / 0.2266 |

The LSTM report head is less robust than historical rates under this stress. At 50% loss, separate-LSTM A ECE rises from 0.0359 to 0.1454 (joint: 0.0415 to 0.1589). Explicit masks prevent conflating unknown with false but do not guarantee robustness or calibration. Full fixed-bin reliability tables are preserved in results.json. ECE alone is insufficient: the weak global-prior model has relatively low A ECE but much worse Brier. No post-test calibrator was fit.

## Cost, validity checks and boundaries

| Model | Mean serialized bytes | Mean warm single-window median latency (ms) |
|---|---:|---:|
| statistics_7 | 331 | 0.0461 |
| logistic_full | 7545 | 1.4135 |
| lstm_joint_full | 28534 | 0.6126 |
| lstm_separate_full | 37095 | 1.0787 |

CPU training/model selection took 218.95s. Sampled training process peak RSS was 574.3 MiB; evaluation peak was 464.5 MiB. These include datasets, interpreter and library memory, with allocator caching and 20 ms sampling; they are not isolated model allocations. Timings include normalization and model/summary execution but exclude event parsing, file I/O and startup. Full p95 measurements and environment versions are in the JSON.

Validation evidence: **86 existing application/router tests + 8 new benchmark tests passed** in their respective locked environments; one existing audioop deprecation warning. Benchmark sources compiled. An independent check reconstructed 378 train/validation labels from actual future event timestamps and approved targets. All 8,640 natural-plus-derived windows were checked for provenance and chronological cutoffs; all cross-partition persona/family/episode/group IDs are disjoint. Frozen source hashes stayed unchanged from training through test evaluation. Root production code, dependencies, routing and authorization are untouched.

Artifacts: [results.json](results.json) contains every model/condition, per-family score, reliability bin, pair diagnostic, timing and memory measurement; [selection.json](selection.json) records validation choices and hashes; [manifest.json](manifest.json) records data hashes, seeds and counts. Checkpoints and full synthetic histories remain under the ignored local artifacts directory and are regenerated by the README commands.

## Interpretation and next experiment

**Observation:** historical information is useful on this simulator; LSTMs learn it. **Measured limit:** they do not beat simpler historical estimates across both targets, and sharing the encoder worsens full-history means. **Hypothesis:** much of the simulated predictive signal is already represented by recent rates and rolling summaries, while unseen workload mechanisms and conditional-completion distribution shifts limit learned generalization. The present ablations do not prove this causal explanation.

Do not select a production model from these results. Retain shadow-only probability output. A next predeclared version should fix the recorded low-variance/support-shift controls using development data, add independently authored mechanisms and separately sealed families, repeat grouped splits, and compare against the same historical baselines. Keep calibration and hyperparameter decisions within training/validation. No OPD, GRPO, RLHF or Mamba training is warranted yet.

The shared synthetic outcome equations, only three held-out mechanisms, overlapping windows, persona dependence and initialization-only repetitions limit inference. There was no independent human review of behavioral realism. Synthetic score improvements do not establish real-user accountability benefits or justify messaging, notification, goal, recovery or approval changes.
