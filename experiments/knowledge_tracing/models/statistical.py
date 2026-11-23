from __future__ import annotations

import numpy as np


class StatisticalModel:
    """Beta-smoothed historical rate centered on training-only global prevalence."""

    def __init__(self, prior, strength=10.0):
        self.prior, self.strength = float(prior), float(strength)

    def predict_counts(self, counts):
        counts = np.asarray(counts)
        return (counts[:, 0] + self.strength * self.prior) / (counts[:, 1] + self.strength)

    def predict_events(self, events):
        c = np.asarray([e["correct"] for e in events])
        return float(self.predict_counts([[np.nansum(c), np.isfinite(c).sum()]])[0])
