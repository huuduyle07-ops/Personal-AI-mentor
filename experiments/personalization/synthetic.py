"""Synthetic learners whose memories decay at different personal rates.

Ground truth is exponential forgetting (HLR form) with per-learner parameters:
    p(recall) = 2^(-days / h),   h = 2^(theta_learner . [sqrt(1+right), sqrt(1+wrong), 1])
theta_learner = BASE + noise, so learners differ in baseline memory (bias), how much a correct
answer helps, and how much a mistake hurts. Each learner studies a 40-word deck: new words are
introduced over time, and a random earlier word is reviewed at each step, so gaps range from
hours to weeks. Only reviews of previously seen words are events, matching personalize.py.
"""
import numpy as np

BASE = np.array([0.9, -0.8, 1.5])     # correct answers lengthen memory, mistakes shorten it
SPREAD = np.array([0.25, 0.25, 0.9])  # learners differ most in overall forgetting speed
WORDS = 40


def learner_theta(rng):
    return BASE + rng.normal(0, SPREAD)


def simulate(rng, theta, n_events):
    seen, events, t = {}, [], 0.0
    while len(events) < n_events:
        t += rng.exponential(0.25)  # days between quiz questions
        new_ok = len(seen) < WORDS and (not seen or rng.random() < 0.25)
        if new_ok:
            word = len(seen)
            seen[word] = {"right": int(rng.random() < 0.6), "wrong": 0, "last": t}
            seen[word]["wrong"] = 1 - seen[word]["right"]
            continue
        word = int(rng.integers(len(seen)))
        state = seen[word]
        days = t - state["last"]
        h = 2.0 ** (theta @ np.array([np.sqrt(1 + state["right"]), np.sqrt(1 + state["wrong"]), 1.0]))
        correct = int(rng.random() < 2.0 ** (-days / h))
        events.append((state["right"], state["wrong"], days, correct))
        state["right"] += correct
        state["wrong"] += 1 - correct
        state["last"] = t
    return events


def cohort(seed, learners, events_each):
    rng = np.random.default_rng(seed)
    return [simulate(rng, learner_theta(rng), events_each) for _ in range(learners)]
