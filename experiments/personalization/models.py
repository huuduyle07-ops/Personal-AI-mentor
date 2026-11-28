"""The two model families compared: logistic recall (deployed) and half-life regression (HLR).

Both are personalized the same way: MAP estimation with a Gaussian prior at the population weights,
minimising  sum_i BCE(y_i, p_i)  +  (strength / 2) * ||w - w_population||^2.
The logistic version calls the production code in personalize.py, so the benchmark evaluates
exactly what the bot runs.
"""
import numpy as np

import personalize
from experiments.forgetting_curve.models import MAX_HALF_LIFE, MIN_HALF_LIFE

EPS = 1e-6


def logistic_design(events):
    e = np.asarray([row[:3] for row in events], float)
    return np.column_stack([np.sqrt(1 + e[:, 0]), np.sqrt(1 + e[:, 1]), np.ones(len(e)), np.log1p(e[:, 2])])


def hlr_design(events):
    e = np.asarray([row[:3] for row in events], float)
    return np.column_stack([np.sqrt(1 + e[:, 0]), np.sqrt(1 + e[:, 1]), np.ones(len(e))]), e[:, 2]


class Logistic:
    name = "logistic"

    @staticmethod
    def fit_population(events, iterations=30):
        """Plain maximum likelihood on a large pool (Newton, tiny ridge for stability)."""
        X, y = logistic_design(events), np.asarray([e[3] for e in events], float)
        w = np.zeros(X.shape[1])
        for _ in range(iterations):
            p = 1 / (1 + np.exp(-(X @ w)))
            step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w)), X.T @ (p - y))
            w -= step
            if np.abs(step).max() < 1e-10:
                break
        return w

    @staticmethod
    def personalize(events, prior, strength):
        return np.asarray(personalize.fit(events, list(prior), strength))

    @staticmethod
    def predict(events, w):
        p = 1 / (1 + np.exp(-(logistic_design(events) @ w)))
        return np.clip(p, EPS, 1 - EPS)


class HalfLife:
    """p = 2^(-days / h), h = 2^(theta . [sqrt(1+right), sqrt(1+wrong), 1]), fit by log loss."""
    name = "hlr"
    LO, HI = np.log2(MIN_HALF_LIFE), np.log2(MAX_HALF_LIFE)

    @classmethod
    def _grad(cls, theta, X, days, y, prior, strength):
        z = X @ theta
        zc = np.clip(z, cls.LO, cls.HI)
        h = 2.0 ** zc
        p = np.clip(2.0 ** (-days / h), EPS, 1 - EPS)
        dp_dz = p * np.log(2) ** 2 * days / h
        dl_dp = -(y / p) + (1 - y) / (1 - p)
        grad = X.T @ (dl_dp * dp_dz * (z == zc))
        if prior is not None:
            grad = grad + strength * (theta - prior)
        return grad

    @classmethod
    def _adam(cls, theta, grad, steps, lr):
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        for t in range(1, steps + 1):
            g = grad(theta)
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            theta = theta - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        return theta

    @classmethod
    def fit_population(cls, events):
        X, days = hlr_design(events)
        y = np.asarray([e[3] for e in events], float)
        n = len(y)
        return cls._adam(np.array([0.0, 0.0, 1.0]), lambda t: cls._grad(t, X, days, y, None, 0) / n, 1500, 0.02)

    @classmethod
    def personalize(cls, events, prior, strength):
        X, days = hlr_design(events)
        y = np.asarray([e[3] for e in events], float)
        scale = len(y) + strength  # normalise the step size; the optimum is unchanged
        return cls._adam(np.array(prior, float), lambda t: cls._grad(t, X, days, y, prior, strength) / scale, 300, 0.02)

    @classmethod
    def predict(cls, events, theta):
        X, days = hlr_design(events)
        h = 2.0 ** np.clip(X @ theta, cls.LO, cls.HI)
        return np.clip(2.0 ** (-days / h), EPS, 1 - EPS)

    @classmethod
    def half_life(cls, theta, right, wrong):
        return float(2.0 ** np.clip(np.array([np.sqrt(1 + right), np.sqrt(1 + wrong), 1.0]) @ theta, cls.LO, cls.HI))
