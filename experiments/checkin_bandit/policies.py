"""Policies compared in the simulation. LinUCB is the production class from checkin_policy.py."""
import numpy as np

from checkin_policy import ACTIONS, FEATURES, LinUCB


class LinUCBPolicy:
    name = "linucb"

    def __init__(self, alpha, rng=None):
        self.model = LinUCB(alpha=alpha)

    def choose(self, x, hour):
        return self.model.choose(x)[0]

    def update(self, action, x, reward):
        self.model.update(action, x, reward)


class ThompsonPolicy:
    """Linear Thompson sampling: per action, theta ~ N(B^-1 f, v^2 B^-1) with B = I + sum x x^T."""
    name = "thompson"

    def __init__(self, v, rng):
        d = len(FEATURES)
        self.v, self.rng = v, rng
        self.B = {a: np.eye(d) for a in ACTIONS}
        self.f = {a: np.zeros(d) for a in ACTIONS}

    def choose(self, x, hour):
        x = np.asarray(x)
        samples = {}
        for a in ACTIONS:
            cov = np.linalg.inv(self.B[a])
            theta = self.rng.multivariate_normal(cov @ self.f[a], self.v ** 2 * cov)
            samples[a] = theta @ x
        best = max(samples.values())
        return next(a for a in ACTIONS if samples[a] >= best - 1e-12)

    def update(self, action, x, reward):
        x = np.asarray(x)
        self.B[action] += np.outer(x, x)
        self.f[action] += reward * x


class FixedSchedule:
    """The bot's current behaviour: a study check-in at midday (12:00) and in the evening (21:00)."""
    name = "fixed_schedule"

    def __init__(self, *_):
        pass

    def choose(self, x, hour):
        return "nudge_study" if hour in (12, 21) else "stay_silent"

    def update(self, *_):
        pass


class RandomPolicy:
    name = "random"

    def __init__(self, _=None, rng=None):
        self.rng = rng

    def choose(self, x, hour):
        return ACTIONS[int(self.rng.integers(len(ACTIONS)))]

    def update(self, *_):
        pass


class Oracle:
    """Knows every action's true expected reward (unavailable in reality); regret is measured against it."""
    name = "oracle"

    def __init__(self, *_):
        self.expected = None

    def choose(self, x, hour):
        best = max(self.expected.values())
        return next(a for a in ACTIONS if self.expected[a] >= best - 1e-12)

    def update(self, *_):
        pass


class NeverNudge:
    name = "never_nudge"

    def __init__(self, *_):
        pass

    def choose(self, x, hour):
        return "stay_silent"

    def update(self, *_):
        pass
