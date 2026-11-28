# Learned check-in timing (contextual bandit): simulation report

Run date: 2026-11-26. Production module: [`checkin_policy.py`](../../checkin_policy.py) (disjoint LinUCB in plain Python, state in the `bandit_state` table, `/policy` preview). Simulation: [`simulator.py`](simulator.py), [`policies.py`](policies.py), [`run.py`](run.py). Raw numbers: [results/selection.json](results/selection.json), [results/test.json](results/test.json).

**Finding:** both learned policies are far better than the bot's current fixed check-in schedule. LinUCB gains +25.5 cumulative reward per learner (95% interval 22.4 to 28.7) and is better for 94 of 100 learners. It also sends less than half as many nudges. But in the main setting, neither learned policy beats simply **never nudging** within 120 days, although an oracle shows that well-timed nudges are worth about +12. Learned timing beats silence only when an unnecessary nudge is cheap (cost 0.05). So the bandit stays preview-only: it is not used to send messages.

## Question

Can a contextual bandit learn, from one learner's own reactions, when a check-in is worth sending (a study nudge, a quiz nudge, or silence), better than fixed times?

## Decision problem

- **Decision points:** 12:00, 18:00 and 21:00 KST each day.
- **Actions:** `stay_silent`, `nudge_study`, `nudge_quiz`.
- **Context:** the 10 features the bot computes (`checkin_policy.context`), all read-only:

| Feature | Meaning |
|---|---|
| bias | Constant 1 |
| afternoon, evening, night | Time-of-day bucket (morning is the reference) |
| weekend | Saturday or Sunday |
| goal_progress | Minutes logged today ÷ daily target, capped at 1.5 |
| logged_today | Anything logged today |
| streak | Days in a row meeting the goal, scaled by 1/14 (capped at 14) |
| words_due | Vocabulary words due for review, scaled by 1/20 (capped at 20) |
| tired_or_busy | A tired or busy obstacle was reported in the last 24 hours |

- **Reward:** +1 if the learner engages within the window (studies or quizzes), minus the nudge cost. A nudge costs 0.15, or 0.45 if the learner is actually tired or busy. Silence costs nothing. The cost stands for annoyance and lost trust, which can't be measured, so it is a modelling assumption. That's why the results include a sensitivity analysis on it.

## Policies

| Policy | Description |
|---|---|
| **LinUCB** (production code) | Per action, A = I + Σxxᵀ and b = Σrx. Score = θᵀx + α√(xᵀA⁻¹x), with A⁻¹ kept up to date by Sherman–Morrison. Ties go to `stay_silent`, so cold start doesn't nudge blindly. |
| Linear Thompson sampling | Per action, sample θ ~ N(B⁻¹f, v²B⁻¹) and pick the best sampled score |
| Fixed schedule | The bot today: study nudge at 12:00 and 21:00, silent at 18:00 |
| Random | Uniform over the three actions |
| Never nudge | Always silent |
| Oracle | Knows each action's true expected reward; the reference for regret |

## Simulated learners

Each learner has:
- **Routine:** a favourite time of day, a weekend effect and a baseline motivation.
- **Nudge response:** how much a study nudge and a quiz nudge raise the chance of studying. Study nudges don't help once the goal is met; quiz nudges help mainly when words are due.
- **Tired spells:** a Markov process; 12% chance per day, rising to 60% if tired yesterday.
- **Busy work weeks:** 25% chance each Monday, lasting 5 weekdays.
- **Vocabulary:** due words pile up (Poisson(1.5) per day) and shrink when quizzed.
- **Engagement probability:** a logistic function of the hidden state, shown in the `simulator.py` docstring.

**Partial observability:** obstacles reach the bot only 70% of the time, so the `tired_or_busy` feature is noisy. The simulator knows every action's expected reward, so regret is exact (pseudo-regret).

## Protocol

- Each learner is simulated for 120 days (360 decisions), starting from a cold-start policy.
- Every policy meets the same learner: the learner's randomness depends only on (seed, learner).
- **Selection:** the grids α ∈ {0.1, 0.3, 1.0} and v ∈ {0.1, 0.3, 1.0} were scored on 40 validation learners (seed 2), by mean cumulative reward. Chosen: **α = 0.3** (validation 74.7 vs 74.3 and 73.1) and **v = 0.1** (74.2 vs 69.4 and 59.8). `checkin_policy.ALPHA` must equal the selection, or the test refuses to run.
- **Test:** 100 test learners (seed 3), scored once.
- **Sensitivity:** costs of 0.05, 0.15 (the main setting) and 0.30 were declared before validation. One earlier smoke run on 10 learners (seed 9) showed that never nudging was strong. The sensitivity analysis was added after that run, but before validation and test, and the simulator was not changed.

## Results: test learners, nudge cost 0.15

| Policy | Cumulative reward (95% CI) | Avg regret / decision | Unnecessary-nudge rate | Ignored-nudge rate | Nudges / day | Nudges while tired or busy | Engagement rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 85.6 [79.1, 91.8] | 0.000 | 0.0% | 41.9% | 1.19 | 0.0% | 29.8% |
| Never nudge | 73.5 [68.0, 78.9] | 0.044 | — | — | 0.00 | — | 20.4% |
| Thompson (v = 0.1) | 69.5 [63.9, 74.8] | 0.052 | 60.9% | 67.1% | 0.82 | 30.1% | 25.2% |
| **LinUCB (α = 0.3)** | **67.0 [61.6, 72.3]** | **0.055** | **57.8%** | **65.5%** | **0.88** | **27.1%** | **25.0%** |
| Fixed schedule | 41.6 [34.8, 47.8] | 0.132 | 65.5% | 66.8% | 2.00 | 36.8% | 28.9% |
| Random | 39.2 [32.4, 45.9] | 0.139 | 68.6% | 67.4% | 2.01 | 37.9% | 28.5% |

- **Unnecessary nudge:** a nudge whose true expected reward was no better than staying silent.
- **Ignored nudge:** a nudge that wasn't followed by engagement.

LinUCB minus each other policy, paired on the same learners (95% learner-bootstrap interval):

| Comparison | Reward difference | LinUCB better for |
|---|---:|---:|
| vs fixed schedule | **+25.5** [22.4, 28.7] | 94 / 100 learners |
| vs random | +27.9 [24.8, 31.0] | 97 / 100 |
| vs Thompson | −2.5 [−4.4, −0.6] | 41 / 100 |
| vs never nudge | **−6.4** [−8.5, −4.4] | 22 / 100 |

Average regret per decision over time:

| Days | 1–30 | 31–60 | 61–120 |
|---|---:|---:|---:|
| LinUCB | 0.074 | 0.054 | 0.047 |
| Thompson | 0.070 | 0.053 | 0.043 |
| Never nudge | 0.046 | 0.044 | 0.044 |
| Fixed schedule | 0.126 | 0.138 | 0.131 |

Sensitivity to the nudge cost (same learners and parameters; cumulative reward, with nudges per day in brackets):

| Nudge cost | Oracle | LinUCB | Thompson | Never nudge | Fixed schedule | Random |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 103.6 (1.77) | **80.8** (1.47) | 78.8 (1.29) | 73.5 (0) | 65.6 (2.0) | 63.3 (2.0) |
| 0.15 (main) | 85.6 (1.19) | 67.0 (0.88) | 69.5 (0.82) | **73.5** (0) | 41.6 (2.0) | 39.2 (2.0) |
| 0.30 | 75.0 (0.26) | 65.4 (0.35) | 66.2 (0.31) | **73.5** (0) | 5.6 (2.0) | 3.1 (2.0) |

## Interpretation

- **The current fixed schedule is the weakest realistic policy.** It nudges twice a day regardless of context: 37% of its nudges reach a tired or busy learner, and two thirds are ignored. Under every cost, the learned policies beat it by a wide margin.
- **Learning is visible but slow.** Bandit regret falls by about 37% from the first month to the last two. But exploring at cold start (it has to try nudges to learn about them) costs more than it gains within 120 days, so silence wins at the main cost. Thompson sampling explores a little more efficiently than LinUCB, but the interval for the difference is close to zero.
- **Value depends on the cost of an unnecessary nudge.** Learned timing beats silence when a needless nudge is cheap (cost 0.05), and the oracle shows real value at every cost. The open question for a real deployment is how much a needless nudge actually annoys a learner. That's a product judgement the simulation can't settle.
- **Why the bandits fall short of the oracle:**
  - **Model mismatch:** a linear reward model approximates a logistic truth with interactions (study nudges stop helping once the goal is met).
  - **Hidden state:** 30% of obstacles are never reported.
  - **Little data:** 360 decisions spread over 3 actions × 10 features.

## Decision

The bandit is **preview-only**. `/policy` shows the context, each action's predicted reward and exploration bonus, and the action it would choose. It sends nothing, and the fixed check-ins are unchanged, as `AGENT.md` requires (no automatic replacement without evidence). These results do support **reducing or contextualizing the fixed check-ins**, for example skipping the evening nudge once the goal is met or after a tired/busy report. That rule-based change would need its own approval.

### Shadow mode (added 2026-11-26)

`checkin_policy.ShadowLearner` lets the bandit learn from real outcomes without changing what is sent:

1. **Study nudges:** after each delivered fixed check-in (12:00 and 21:00), the context at that moment is logged in `bandit_shadow` as a `nudge_study` decision, together with the action the bandit would have chosen.
2. **Silence:** between 18:00 and 20:00, if check-ins are enabled and not paused, one `stay_silent` observation is logged per day. Nothing is sent.
3. **Scoring:** three hours after each observation, the outcome is scored with the same reward as the simulation. It counts as engaged if the learner logged study or answered a quiz in that window. The bandit is then updated (A and b in `bandit_state`).

Shadow mode only runs while check-ins are on, stores feature vectors but never message text, and swallows its own errors. Tests check that sends are identical with and without it.

**Limits:**
- **No quiz nudges:** the bot never sends one, so the `nudge_quiz` arm stays at cold start.
- **Off-policy learning:** actions follow the fixed schedule rather than the bandit, so learning is off-policy. The time-of-day features absorb most of the schedule's influence, but at 12:00 and 21:00 the bandit only ever sees `nudge_study`, and at 18:00 only `stay_silent`. Its estimates for the actions *not* taken in each slot therefore come from other slots.
- **One user's data:** the evidence will accumulate slowly.

## Limitations

- **Simulated learners only.** The routines, effect sizes and reward costs are assumptions, not data.
- **Little real outcome data yet.** Shadow mode (above) now learns from real check-ins, but only while check-ins are enabled, and only for the study-nudge and silence arms.
- **The engagement window is simplified.** The simulator counts engagement in the same decision slot; the production reward definition uses a 3-hour window.
- **Narrow uncertainty estimates.** Single simulated population; the intervals cover learner resampling only.

## Reproduce

```bash
uv sync --locked --extra research
uv run --locked --extra research python -m experiments.checkin_bandit.run select   # validation learners
uv run --locked --extra research python -m experiments.checkin_bandit.run test     # refuses to reopen test learners
uv run python -m pytest tests/test_checkin_policy.py -q
```
