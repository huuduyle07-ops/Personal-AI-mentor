"""Learned forgetting curve: predicts how likely you are to recall a word right now.

The population weights come from experiments/forgetting_curve (trained on Duolingo review logs,
evaluated on held-out learners); personalize.py adapts them to one learner's quiz history.
This module only does inference, so it has no numpy or research dependency. If the weights file
is missing or invalid, a fixed Leitner-style rule is used instead.
"""
import json
import math
from pathlib import Path

WEIGHTS = Path(__file__).with_name("models") / "recall_model.json"
FEATURES = ["sqrt_right", "sqrt_wrong", "bias", "log1p_days"]
FOREVER = float("inf")


def features(right, wrong, days):
    return [math.sqrt(1 + right), math.sqrt(1 + wrong), 1.0, math.log1p(max(0.0, float(days)))]


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


class RecallModel:
    def __init__(self, path=WEIGHTS, *, weights=None, source=None):
        self.weights, self.source = None, "fallback_rule"
        if weights is not None:
            self.weights, self.source = [float(w) for w in weights], source or "custom"
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            weights = payload["weights"]
            if (payload.get("model") != "logistic_recall" or payload.get("features") != FEATURES
                    or len(weights) != len(FEATURES)
                    or not all(isinstance(w, (int, float)) and math.isfinite(w) for w in weights)):
                raise ValueError("Unexpected recall model format")
            self.weights, self.source = [float(w) for w in weights], "population"
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def recall(self, right, wrong, days):
        """Probability of recalling a word, given past right/wrong answers and days since last review."""
        days = max(0.0, float(days))
        if self.weights is None:
            return 2.0 ** (-days / self._leitner_half_life(right, wrong))
        return sigmoid(sum(w * x for w, x in zip(self.weights, features(right, wrong, days))))

    def days_until(self, right, wrong, target):
        """Days after a review until predicted recall falls to `target` (0 if already below; inf if never)."""
        if self.weights is None:
            return self._leitner_half_life(right, wrong) * math.log2(1 / target)
        w_right, w_wrong, bias, w_time = self.weights
        static = w_right * math.sqrt(1 + right) + w_wrong * math.sqrt(1 + wrong) + bias
        needed = math.log(target / (1 - target))  # logit of the target
        if static <= needed:
            return 0.0
        if w_time >= 0:
            return FOREVER  # this model never predicts forgetting over time
        exponent = (needed - static) / w_time
        return FOREVER if exponent > 700 else math.expm1(exponent)

    def half_life(self, right, wrong):
        """Days until predicted recall falls to 50%."""
        return self.days_until(right, wrong, 0.5)

    @staticmethod
    def _leitner_half_life(right, wrong):
        return min(max(2.0 ** (right - wrong), 1 / 96), 274)  # doubles per net correct answer
