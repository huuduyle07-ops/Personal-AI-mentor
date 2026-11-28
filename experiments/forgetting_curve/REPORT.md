# Forgetting-curve model for the vocabulary quiz: measured report

Run date: 2026-11-25. Code: `data.py` (download and split), `models.py` (models and metrics), `run.py` (selection, single test opening, export). Raw numbers: [results/results.json](results/results.json), [results/selection.json](results/selection.json).

**Finding:** the pre-declared model, half-life regression (HLR), wins on the pre-declared metric (MAE) but loses to a simpler logistic model on every proper scoring metric. It also learns that wrong answers *lengthen* memory. The bot therefore uses the logistic recall model. That switch was decided after the test set was opened, so its test numbers are not a clean confirmatory estimate (see Limitations).

## Question

Can a model trained on real review logs predict how likely a learner is to recall a word, well enough to choose which Korean word the quiz asks next?

## Data

- **Source:** Duolingo half-life regression learning traces (Settles & Meeder 2016), [doi:10.7910/DVN/N8XJME](https://doi.org/10.7910/DVN/N8XJME), CC BY-NC 4.0. Cite: B. Settles and B. Meeder, *A Trainable Spaced Repetition Model for Language Learning*, ACL 2016.
- **Sample:** the first 1,000,000 rows of the 13-million-row file, streamed and stopped early (28.97 MB compressed downloaded). This covers about 23.5 hours of logs, 20,204 learners and six course languages (English, Spanish, French, German, Italian, Portuguese). The sample's SHA-256 is recorded in `artifacts/provenance.json`; the data itself is Git-ignored.
- **Each row** is one word in one practice session:
  - **`p_recall`:** the fraction of times the word was answered correctly in that session (the target).
  - **`delta`:** time since the word was last seen.
  - **History:** the learner's earlier right and wrong counts for that word.
- **Split:** by learner, using a SHA-256 hash with seed 42, 70/15/15. No learner appears in two partitions.

| Partition | Rows | Learners | Mean p_recall |
|---|---:|---:|---:|
| Train | 698,761 | 14,173 | 0.894 |
| Validation | 153,890 | 3,006 | 0.894 |
| Test | 147,349 | 3,025 | 0.899 |

84% of sessions have perfect recall (p = 1). The median time since last review is 0.8 days, and the 90th percentile is 16 days.

## Models

All models use features that transfer to new words: √(1+right), √(1+wrong), a bias, and the time since last review. The one exception is the `hlr_lexeme` ablation, which adds a learned weight for each word.

| Model | Prediction |
|---|---|
| constant_mean | Training-set mean recall |
| leitner | `2^(−Δ/h)` with h = 2^(right − wrong), a fixed rule |
| pimsleur | `2^(−Δ/h)` with h = 2^(2.35·seen − 16.46), a fixed rule |
| hlr | `2^(−Δ/h)` with h = 2^(θ·x), squared loss plus α·(h − h_target)² plus L2 (Settles & Meeder) |
| hlr_lexeme | HLR plus one weight per word (ablation) |
| logistic_regression | sigmoid(w·[x, log(1+Δ)]), fit to p_recall by cross-entropy |

- **Training:** full-batch Adam in numpy, 400 steps.
- **Hyperparameters:** α ∈ {0, 0.001, 0.01}, L2 penalty ∈ {0, 0.001} (logistic: L2 ∈ {0, 0.001}).
- **Selection:** by **validation MAE**, chosen in advance because it is the paper's main metric. The test learners were then scored once.
- **Unit tests:** `test_models.py` covers a finite-difference gradient check, recovery of known parameters on synthetic data, AUC tie handling, and the learner bootstrap.

## Test results (3,025 held-out learners, opened once)

Lower MAE and MSE are better; higher AUC is better. AUC ranks sessions with p_recall ≥ 0.5 above those below. The last column is the Spearman correlation between the predicted half-life and the half-life implied by the observed recall.

| Model | MAE | MSE | AUC | Half-life Spearman |
|---|---:|---:|---:|---:|
| constant_mean | 0.1735 | 0.0716 | 0.500 | — |
| leitner | 0.2056 | 0.1440 | 0.557 | −0.036 |
| pimsleur | 0.4163 | 0.3665 | 0.528 | −0.035 |
| **hlr** | **0.1107** | 0.0801 | 0.547 | 0.160 |
| hlr_lexeme | 0.1108 | 0.0801 | 0.547 | 0.158 |
| **logistic_regression** | 0.1699 | **0.0705** | **0.600** | — |

The table below shows the MAE difference between HLR and each other model, with 95% intervals from 1,000 paired resamples of whole test learners. A negative difference favors HLR.

| Contrast | ΔMAE | 95% interval |
|---|---:|---|
| HLR − constant | −0.0629 | [−0.0643, −0.0613] |
| HLR − Leitner | −0.0949 | [−0.1006, −0.0900] |
| HLR − logistic | −0.0592 | [−0.0604, −0.0578] |
| HLR − HLR + per-word weights | −0.0001 | [−0.0001, −0.0001] |

The ordering of HLR against the constant, Leitner and Pimsleur baselines matches the pattern in the original paper, which reports an HLR MAE of about 0.13 and an AUC of about 0.54. As in the paper, per-word weights add almost nothing.

## What went wrong with HLR

1. **MAE rewards always predicting "remembered".** With 84% of targets at exactly 1.0, the error-minimizing prediction is close to 1. HLR predicts above 0.9 for 96% of test sessions, while the observed recall in that bin is 0.90. Its MSE (0.0801) is worse than the constant's (0.0716). MAE is not a proper scoring rule for probabilities, and that is why it chose this model.
2. **The weights point the wrong way for scheduling.** The MAE-selected HLR learned θ = [right +2.00, wrong +2.19, bias +2.41], so each wrong answer lengthens the predicted memory. The data says the opposite. With right answers held at 3–6, mean recall falls from 0.91 with 0 wrong answers to 0.85 with 2, 0.78 with 4 and 0.62 with 8. A quiz using these weights would show the words you miss *less* often.
3. **The functional form fits poorly.** `2^(−Δ/h)` forces recall toward 1 just after a review, but recall here is about 0.9 at almost any interval, because Duolingo schedules reviews before people forget. Retraining HLR with log loss on validation did not help: its log loss was 0.63–0.66, against 0.338 for the constant.

Validation-only diagnostics, run after the test opening and used only to choose the production model:

| Validation | MAE | MSE | AUC | Log loss |
|---|---:|---:|---:|---:|
| constant | 0.1781 | 0.0763 | 0.500 | 0.3377 |
| HLR (squared loss, α = 0.01) | 0.1248 | 0.0872 | 0.544 | 0.6563 |
| HLR (log loss, best start) | 0.1546 | 0.0989 | 0.552 | 0.6307 |
| logistic (1,500 steps) | 0.1750 | 0.0755 | 0.588 | 0.3338 |

## Decision: what the bot uses

The bot uses `logistic_regression`, the exact model scored on the test learners above. Its weights are exported to [`models/recall_model.json`](../../models/recall_model.json) and loaded by `recall_model.py`:

```
p = sigmoid(0.143·√(1+right) − 0.423·√(1+wrong) + 2.428 − 0.110·log(1+days))
```

The signs make sense: correct answers help, mistakes hurt more, and recall fades with time. The quiz asks the known word with the lowest predicted recall once it falls below 90%; otherwise it introduces a new word. This gives growing review intervals without any hand-written spacing rule: after 1, 3 and 6 correct answers, a word becomes due after about 0.25, 1.5 and 4.5 days. Two scheduling rules sit on top of the model. A missed word is retried once, straight away. And a word you have just answered rests for an hour, because the model rewards each correct answer only slightly (+0.14 per √right against −0.42 per √wrong): after one mistake, a word stays just below 90% even right after a correct answer, and without the rest the quiz repeated it indefinitely. If the weights file is missing or invalid, a Leitner rule takes over.

## Course algorithms: KNN, SVM, decision tree and random forest

Added 2026-11-27 to compare the production model with the algorithms taught in the course (Lectures 5–7). Code: [`classic.py`](classic.py), tests: [`test_classic.py`](test_classic.py). Raw numbers: [results/classic_selection.json](results/classic_selection.json), [results/classic_test.json](results/classic_test.json).

**Finding:** the random forest predicts recall slightly better than logistic regression: log loss 0.3192 against 0.3212, a difference of −0.0020 [−0.0029, −0.0012]. A decision tree and KNN are also slightly better, and the SVM is clearly worse. The gains are small (about 0.6% of log loss) and come with a model the bot can't personalize or explain, so the bot keeps logistic regression.

### Protocol

- **Same everything:** the same sample, learner split, features (√(1+right), √(1+wrong), log(1+days)) and test learners as above. Each model predicts p_recall directly:
  - **KNN:** the mean recall of the k nearest training sessions.
  - **SVM:** support vector regression with an RBF kernel, clipped to [0, 1].
  - **Decision tree:** a regression tree whose leaves predict the mean recall of their sessions.
  - **Random forest:** 100 such trees on bootstrap samples, with 2 of the 3 features tried at each split, averaged.
- **Scaling:** KNN and the SVM use standardized features; trees don't need them.
- **Training size:** kernel SVMs scale roughly quadratically with data, so the SVM trains on a random 5,000 sessions and KNN on 200,000. The trees and logistic regression use all 698,761.
- **Selection by validation log loss, not MAE.** The earlier run showed that MAE rewards predicting "remembered" for everything. Grids:
  - KNN: k ∈ {25, 100, 400, 1600};
  - SVM: C ∈ {0.01, 0.1, 1, 10};
  - Decision tree: max depth ∈ {3, 6, 10, 14};
  - Random forest: max depth ∈ {6, 10, 14}.

  The KNN and SVM grids were widened once, on validation only, because their first best values sat at the edge of the grid. The widened grids changed validation log loss by at most 0.0004.
- **Chosen:** KNN k = 1600, SVM C = 0.1, tree depth 10, forest depth 10. Logistic regression keeps its earlier setting (no L2 penalty).
- **Test:** the test learners were scored once for these models. They had already been opened for the models above, and the logistic regression's test numbers were known, but nothing here was chosen with test data.
- **Lecture 4 classification view:** the task is detecting a forgotten word, a session with p_recall < 0.5 (7.0% of test sessions). Each model's threshold on predicted recall was chosen on validation to maximize F1 for that class.

### Test results (3,025 learners, 147,349 sessions)

| Model | Log loss | MSE | MAE | AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant (training mean) | 0.3265 | 0.0716 | 0.1735 | 0.500 | — | 0.000 | 0.000 | **0.930** |
| Logistic regression (in the bot) | 0.3212 | 0.0705 | 0.1699 | 0.600 | 0.104 | 0.362 | 0.161 | 0.736 |
| KNN (k = 1600) | 0.3204 | 0.0703 | 0.1673 | 0.600 | 0.105 | 0.340 | 0.160 | 0.749 |
| SVM (RBF, C = 0.1) | 0.3472 | 0.0741 | **0.1350** | 0.516 | 0.073 | **0.524** | 0.128 | 0.500 |
| Decision tree (depth 10) | 0.3201 | 0.0702 | 0.1683 | 0.606 | 0.104 | 0.382 | 0.164 | 0.725 |
| **Random forest (depth 10)** | **0.3192** | **0.0700** | 0.1686 | **0.612** | **0.111** | 0.330 | **0.167** | 0.768 |

AUC here ranks forgotten sessions below remembered ones. The last four columns detect forgotten words, using each model's validation threshold.

Each model minus logistic regression, in log loss (negative is better), with 95% intervals from 1,000 resamples of whole learners:

| Model | Difference | 95% interval |
|---|---:|---|
| Random forest | −0.0020 | [−0.0029, −0.0012] |
| Decision tree | −0.0011 | [−0.0020, −0.0003] |
| KNN | −0.0008 | [−0.0012, −0.0004] |
| Constant | +0.0053 | [+0.0039, +0.0074] |
| SVM | +0.0260 | [+0.0213, +0.0317] |

Confusion matrices for detecting forgotten words (test):

| Model | Flagged and forgot (TP) | Flagged but remembered (FP) | Missed (FN) | Correctly not flagged (TN) |
|---|---:|---:|---:|---:|
| Constant | 0 | 0 | 10,357 | 136,992 |
| Logistic regression | 3,749 | 32,365 | 6,608 | 104,627 |
| Random forest | 3,420 | 27,289 | 6,937 | 109,703 |

### What this shows

- **Trees capture a little more structure.** The forest's gain over logistic regression is statistically clear but small. A tree can model interactions between features (for example, the effect of a mistake depending on how many times the word was answered right), which a linear model can't; which interactions the forest uses wasn't analysed. KNN gets most of the same gain by averaging similar sessions, but only with a very large k (1,600) and 51 seconds to predict the test set, against 0.05 s for logistic regression.
- **The SVM fails the same way HLR did.** Support vector regression's epsilon-insensitive loss behaves like absolute error, so it predicts something close to the *typical* outcome instead of the probability. With 84% of sessions perfect, it predicts about 0.95 for almost every word, even after several mistakes (`test_classic.py` shows this on synthetic data whose true recall is 0.75). It has the **best MAE (0.1350) and the worst log loss and AUC**. MAE misleads a second time.
- **Accuracy is the wrong metric here.** The constant model has the highest accuracy (0.930) and detects no forgotten words at all, because 93% of sessions were remembered. This is the accuracy paradox from Lecture 4. F1 for the minority class and AUC are what separate the models.
- **The signal is weak for every algorithm.** The best F1 for forgotten words is 0.167: even the forest flags 8 remembered sessions for every forgotten one it finds. Counts and elapsed time carry little information about a single session, whatever the algorithm.

### Decision

The bot keeps logistic regression:
1. **Personalization:** the forest's gain (−0.0020 log loss) is smaller than the gain from personalizing logistic regression on simulated learners (−0.057, see the personalization report). Personalization needs a model with a few weights that a prior can sit on; a forest of 100 trees has no such form.
2. **Explanation:** logistic regression gives each word a smooth recall curve and a half-life that `/recall` explains. A forest's predictions are step-shaped and aren't guaranteed to fall with time or rise with right answers.
3. **Size and dependencies:** the logistic model is 4 numbers in a JSON file with no numpy or scikit-learn dependency. The depth-10 forest is 100 trees and would add scikit-learn to the bot.

If personalization didn't matter, the random forest would be the best choice on accuracy alone.

## Limitations

- **The model choice was post hoc.** The production model was chosen after the test set was opened, using validation evidence. The logistic model's test numbers were already visible, so they are not an unbiased estimate for that decision. A clean confirmation needs a fresh test set, for example learners from later rows of the file who are not in this sample.
- **Domain shift.** The model was trained on Duolingo courses for six European languages and tested on held-out learners from the same courses. It has not been evaluated on Korean, on this bot's quiz format, or on you. Your own quiz history will be the real test.
- **Weak signal.** All AUCs are near 0.55–0.60. Recall is hard to predict from counts and time alone; Duolingo's scheduling makes intervals depend on performance, which weakens the time effect.
- **A narrow sample.** The 23.5-hour slice means each learner's history counts were built up earlier, but timing patterns over longer periods are underrepresented.
- **Limited uncertainty estimates.** Single seed and single split; the intervals cover learner resampling only.

## Reproduce

```bash
uv sync --locked --extra research
uv run --locked --extra research python -m experiments.forgetting_curve.data
uv run --locked --extra research python -m experiments.forgetting_curve.run   # refuses to reopen the test set
uv run --locked --extra research python -m experiments.forgetting_curve.run --export-only
uv run --locked --extra research python -m experiments.forgetting_curve.classic select   # KNN, SVM, tree, forest: validation only
uv run --locked --extra research python -m experiments.forgetting_curve.classic test     # refuses to reopen the test set
uv run --locked --extra research python -m pytest experiments/forgetting_curve -q
```
