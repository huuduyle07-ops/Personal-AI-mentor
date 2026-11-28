"""Simulated learners with routines, tired spells and busy weeks.

Each day has three decision points (12:00, 18:00, 21:00 KST). At each, the policy sees the same
features the bot computes (checkin_policy.FEATURES) and picks stay_silent, nudge_study or nudge_quiz.
The learner then engages (studies or quizzes) with a probability that depends on hidden state:

    self = base + time preference + weekend effect + 0.08*streak - 2.0*goal met - 1.2*tired - 1.0*busy
    p(engage | stay_silent) = sigmoid(self)
    p(engage | nudge_study) = sigmoid(self + study lift * (goal not met) - 0.8*busy)
    p(engage | nudge_quiz)  = sigmoid(self + quiz lift * (1 if words due else 0.3) - 0.6*tired + 0.3*goal met)

Reward = engaged - nudge cost, where a nudge costs 0.15, or 0.45 if the learner is tired or busy
(the same definition as checkin_policy.py). Tiredness and busyness are only reported to the bot 70%
of the time, so the policy's obstacle feature is a noisy view of the truth. Because the simulator
knows every action's expected reward, regret is computed exactly (pseudo-regret).
"""
from dataclasses import dataclass, field

import numpy as np

from checkin_policy import ACTIONS, FEATURES

SLOTS = (12, 18, 21)
BUCKETS = {12: "afternoon", 18: "evening", 21: "night"}
TARGET = 30
NUDGE_COST, ANNOYANCE = 0.15, 0.30
REPORT_PROBABILITY = 0.7


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@dataclass
class Learner:
    base: float
    preference: dict
    weekend: float
    study_lift: float
    quiz_lift: float
    rng: np.random.Generator
    day: int = 0
    minutes: int = 0
    streak: int = 0
    due: int = 3
    tired: bool = False
    busy_left: int = 0
    reported: bool = False
    history: list = field(default_factory=list)

    @classmethod
    def sample(cls, rng):
        favourite = rng.choice(list(BUCKETS.values()))
        preference = {b: (1.0 if b == favourite else rng.normal(-0.4, 0.3)) for b in BUCKETS.values()}
        return cls(base=rng.normal(-1.2, 0.5), preference=preference, weekend=rng.normal(0, 0.6),
                   study_lift=rng.normal(1.0, 0.4), quiz_lift=rng.normal(0.8, 0.4), rng=rng)

    @property
    def busy(self):
        return self.busy_left > 0

    def start_day(self):
        weekday = self.day % 7
        if weekday == 0 and self.busy_left == 0 and self.rng.random() < 0.25:
            self.busy_left = 5  # a busy work week
        self.tired = self.rng.random() < (0.6 if self.tired else 0.12)
        obstacle = self.tired or self.busy
        self.reported = obstacle and self.rng.random() < REPORT_PROBABILITY
        self.due = min(40, self.due + int(self.rng.poisson(1.5)))
        self.minutes = 0

    def end_day(self):
        self.streak = self.streak + 1 if self.minutes >= TARGET else 0
        if self.busy_left and self.day % 7 < 5:
            self.busy_left -= 1
        self.day += 1

    def features(self, hour):
        bucket = BUCKETS[hour]
        values = {"bias": 1.0, "afternoon": float(bucket == "afternoon"), "evening": float(bucket == "evening"),
                  "night": float(bucket == "night"), "weekend": float(self.day % 7 >= 5),
                  "goal_progress": min(self.minutes / TARGET, 1.5), "logged_today": float(self.minutes > 0),
                  "streak": min(self.streak, 14) / 14, "words_due": min(self.due, 20) / 20,
                  "tired_or_busy": float(self.reported)}
        return [values[name] for name in FEATURES]

    def engage_probabilities(self, hour):
        met = float(self.minutes >= TARGET)
        z = (self.base + self.preference[BUCKETS[hour]] + self.weekend * (self.day % 7 >= 5)
             + 0.08 * min(self.streak, 10) - 2.0 * met - 1.2 * self.tired - 1.0 * self.busy)
        return {"stay_silent": sigmoid(z),
                "nudge_study": sigmoid(z + self.study_lift * (1 - met) - 0.8 * self.busy),
                "nudge_quiz": sigmoid(z + self.quiz_lift * (1.0 if self.due else 0.3) - 0.6 * self.tired + 0.3 * met)}

    def nudge_cost(self):
        from . import simulator  # read at call time so the sensitivity analysis can vary it
        return simulator.NUDGE_COST + ANNOYANCE * (self.tired or self.busy)

    def expected_rewards(self, hour):
        cost = self.nudge_cost()
        p = self.engage_probabilities(hour)
        return {a: p[a] - (0.0 if a == "stay_silent" else cost) for a in ACTIONS}

    def act(self, hour, action):
        """Apply an action; returns (realised reward, engaged)."""
        engaged = self.rng.random() < self.engage_probabilities(hour)[action]
        if engaged:
            if action == "nudge_quiz":
                self.due = max(0, self.due - 3)
                self.minutes += int(self.rng.integers(5, 15))
            else:
                self.minutes += int(self.rng.integers(15, 35))
        cost = 0.0 if action == "stay_silent" else self.nudge_cost()
        return float(engaged) - cost, engaged
