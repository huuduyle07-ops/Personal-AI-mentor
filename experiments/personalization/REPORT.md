# Personalized recall models: population vs personalized

Run date: 2026-11-25. Code: [`personalize.py`](../../personalize.py) (production personalization), [`models.py`](models.py) (logistic and HLR families), [`synthetic.py`](synthetic.py) (simulated learners), [`run.py`](run.py) (selection, test and real-data evaluation). Raw numbers: [results/selection.json](results/selection.json), [results/test.json](results/test.json), [results/real.json](results/real.json).

**Finding:** on simulated learners whose forgetting differs from the population, personalizing with a ridge prior at the population weights clearly improves next-review predictions for both model families. The gain grows with the learner's history. With fewer than 5 reviews, personalizing slightly hurts, so the bot waits for 5. The author's real history is too small (2 review events) to evaluate yet.

## Question

The deployed recall model was trained on Duolingo learners (see [forgetting_curve/REPORT.md](../forgetting_curve/REPORT.md)). Does adapting it to one learner's own quiz answers predict that learner's next review better, and how many reviews does that take?

## Models

Two families, each in a population and a personalized version:

| Family | Prediction | Population weights for real data |
|---|---|---|
| Logistic recall (deployed) | p = σ(w·[√(1+right), √(1+wrong), 1, log(1+days)]) | Duolingo logistic model (`models/recall_model.json`) |
| Half-life regression | p = 2^(−days/h), h = 2^(θ·[√(1+right), √(1+wrong), 1]) | Duolingo HLR (`forgetting_curve/results/results.json`) |

**Personalization:** for both families, MAP estimation with a Gaussian prior centred on the population weights, which is ridge shrinkage toward the prior:

```
minimise  Σ_i BCE(y_i, p_i(w))  +  (λ / 2) · ||w − w_population||²
```

- **λ is the prior strength.** It acts roughly like λ pseudo-reviews of evidence for the population weights. With few reviews the prior dominates; as reviews accumulate, the data takes over.
- **Logistic** is solved exactly by Newton's method in plain Python. The benchmark calls the same production function the bot runs, `personalize.fit`.
- **HLR** uses Adam, because its loss is not convex.

**Cold start:** with fewer than `MIN_EVENTS` reviews, the bot uses the population model unchanged. The quiz always uses the population model's ranking until then.

## What counts as a review event

`personalize.recall_events` turns quiz attempts into events with these features:
- right and wrong answers *before* this one;
- days since this word was last answered;
- whether the answer was correct.

Two kinds of attempts are excluded because they don't test memory:
- **A word's first answer:** there was nothing to remember yet.
- **An answer within an hour of a miss on that word:** the correct meaning had just been shown.

## Protocol: prequential (next-review) validation

For each learner, reviews are processed in time order. The prediction for review *i* uses a model fitted only on that learner's reviews 0…i−1. Weights are refitted after every review for the first 30 (where the cold-start threshold is decided), then every 10. This is temporal validation: no future review ever informs a past prediction.

## Synthetic multi-learner benchmark

Real personal data is far too small, so learners are simulated.

**The simulated learners:**
- **Ground truth:** exponential forgetting (the HLR form) with per-learner parameters θ = [0.9, −0.8, 1.5] + N(0, [0.25, 0.25, 0.9]). Learners differ in baseline memory, how much a correct answer helps, and how much a mistake hurts.
- **Study pattern:** each learner studies a 40-word deck. New words are introduced over time, and a random earlier word is reviewed about every 6 hours, so gaps range from hours to weeks. About 51% of reviews are correct.
- **Design choice:** the baseline memory term was raised from 0.3 to 1.5 before any evaluation, because 0.3 gave an unrealistic 38% correct rate.

**Splits:**

| Split | Seed | Learners | Reviews each | Use |
|---|---:|---:|---:|---|
| Population pool | 1 | 200 | 300 | Fit the population model of each family |
| Validation | 2 | 60 | 150 | Choose λ and the cold-start threshold |
| Test | 3 | 100 | 200 | Scored once with the chosen settings |

### Selection (validation learners only)

The grid was λ ∈ {1, 3, 10, 30, 100}, chosen by validation log loss with personalization always on:

| λ | 1 | 3 | **10** | 30 | 100 |
|---|---:|---:|---:|---:|---:|
| Logistic log loss | 0.4369 | 0.4316 | **0.4313** | 0.4358 | 0.4474 |
| HLR log loss | 0.4299 | 0.4249 | **0.4248** | 0.4312 | 0.4471 |

Change in log loss from personalizing (λ = 10, negative is better), by how many reviews the learner already had:

| Reviews so far | 0–4 | 5–9 | 10–19 | 20–29 | 30+ |
|---|---:|---:|---:|---:|---:|
| Logistic | **+0.0018** | −0.0134 | −0.0245 | −0.0441 | −0.0567 |
| HLR | +0.0002 | −0.0129 | −0.0325 | −0.0551 | −0.0649 |

Personalizing on fewer than 5 reviews slightly hurts. From 5 on, it helps every window. So `PRIOR_STRENGTH = 10` and `MIN_EVENTS = 5` were committed to `personalize.py` before the test learners were scored. The test phase refuses to run if the constants differ from the selection.

### Test results (100 held-out learners, 20,000 next-review predictions, opened once)

| Model | MAE | Log loss (BCE) | AUROC |
|---|---:|---:|---:|
| Logistic, population | 0.3011 | 0.4630 | 0.8625 |
| **Logistic, personalized (deployed)** | **0.2637** | **0.4061** | **0.8951** |
| HLR, population | 0.2926 | 0.4666 | 0.8630 |
| HLR, personalized | 0.2543 | 0.4015 | 0.8970 |

Personalized minus population log loss, with 95% intervals from 1,000 learner resamples:
- **Logistic:** −0.0569 [−0.0717, −0.0433]; personalized was better for 82 of 100 learners.
- **HLR:** −0.0651 [−0.0839, −0.0482]; better for 78 of 100 learners.

Log loss by history length (population → personalized):

| Reviews so far | 0–9 | 10–29 | 30–99 | 100+ |
|---|---|---|---|---|
| Logistic | 0.311 → 0.306 | 0.434 → 0.396 | 0.509 → 0.446 | 0.452 → 0.390 |
| HLR | 0.308 → 0.300 | 0.439 → 0.390 | 0.516 → 0.443 | 0.454 → 0.385 |

AUROC follows the same pattern. For the logistic model it goes 0.756 → 0.761 for 0–9 reviews and 0.854 → 0.895 for 100+.

**Reading these results:**
- **Personalization helps both families, and more with more data.** That matches the purpose of the ridge prior: small adjustments early, larger ones once there is evidence.
- **HLR personalizes slightly better here, but this setup favors it.** The simulated learners forget in exactly the HLR form, and the logistic model's log(1+days) term is a misspecification. This comparison is biased toward HLR, so it should not be read as evidence that HLR is the better family. On the real Duolingo data the opposite held (see the forgetting-curve report), which is why the bot keeps the logistic family.
- **MAE is reported because it is the standard HLR metric,** but log loss and AUROC are the ones to trust (see the forgetting-curve report on MAE).

## Real personal history (data/accountability.db, read-only)

| | Value |
|---|---|
| Review events | 2 (both correct) |
| Time since last review | median 0.03 h, max 0.05 h (2–3 minutes) |
| Days spanned | 0 |
| Population vs personalized | identical (personalization needs 5 events); log loss 0.121 (logistic) |
| AUROC | undefined (only one outcome) |

**No conclusion is possible from this.** Most of the logged answers were first sightings or immediate retries, so they are correctly excluded. A meaningful real-data comparison needs at least dozens of reviews, spread over days, and including some mistakes. Rerun `python -m experiments.personalization.run real` after a few weeks of quizzing.

## Cold-start caveats

- **Below 5 reviews the bot uses the population model unchanged.** The validation data showed personalizing that early adds noise.
- **Early personal weights stay close to the Duolingo prior.** At λ = 10, the first few dozen reviews move the weights only modestly, so early scheduling still inherits the population model's weaknesses. For example, it predicts very slow forgetting, which is why `/recall` shows half-lives of over a year.
- **The prior comes from another population.** Duolingo learners studying six European languages, not Korean learners using this bot. Personalization is the mechanism meant to correct for that, but only with enough real reviews.
- **Exposure is not random.** The quiz shows words when the model predicts they're due, so the review data is biased by the model itself. This is a standard problem in spaced repetition. A small amount of random review scheduling would be needed for unbiased evaluation.
- **λ and the threshold were tuned on simulated learners.** Real learners may need different values. Once enough real reviews exist, λ should be re-selected by prequential log loss on them.

## Explainable scheduling in the bot

`/recall` (Discord and the terminal chat) shows:
- which model is in use (population, or personalized with the review count);
- the top 5 seen words ranked as the scheduler sees them: due first (lowest predicted recall), then resting, then not due;
- for each word: predicted recall now, half-life (days until predicted recall falls to 50%), right/wrong counts, and the reason for its status;
- which word the next quiz will ask, and why.

Each quiz also records the model used (`population`, `personalized` or `fallback_rule`) and the review count in its `quiz_offered` event.

## Reproduce

```bash
uv sync --locked --extra research
uv run --locked --extra research python -m experiments.personalization.run select   # validation learners only
uv run --locked --extra research python -m experiments.personalization.run test     # refuses to reopen the test learners
uv run --locked --extra research python -m experiments.personalization.run real     # your history, read-only
uv run python personalize.py                                                        # what the model learns from you
uv run python -m pytest tests/test_personalize.py -q
```
