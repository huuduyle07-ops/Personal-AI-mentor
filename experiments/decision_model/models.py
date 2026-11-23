"""Shadow-only probability models; no action/executor interface."""
from typing import Protocol

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from experiments.decision_model.features import N, aggregate


class SequencePredictor(Protocol):
    """Future Mamba implementations can implement this without runtime coupling."""
    def predict_proba(self, histories: np.ndarray) -> np.ndarray:
        """Return [P(report in window), P(completion | valid report)]."""
        ...


class HistoricalProbability:
    def __init__(self, width=7):
        self.width = width

    def fit(self, values, masks, targets, target_mask):
        self.prior = np.array([(targets[:, head] * target_mask[:, head]).sum() / target_mask[:, head].sum().clip(1)
                               for head in range(2)])
        return self

    def predict_raw(self, values, masks):
        if self.width == 0:
            return np.tile(self.prior, (len(values), 1))
        valid = masks[:, -self.width:, 4]
        reports = values[:, -self.width:, 4]
        ratios = values[:, -self.width:, 3]
        report_known = masks[:, -self.width:, 3] * reports
        p_a = ((reports * valid).sum(axis=1) + 2 * self.prior[0]) / (valid.sum(axis=1) + 2)
        p_b = (((ratios >= 1) * report_known).sum(axis=1) + 2 * self.prior[1]) / (report_known.sum(axis=1) + 2)
        return np.stack((p_a, p_b), axis=1)


class LogisticHeads:
    def __init__(self, regularization=1.0):
        self.regularization = regularization
        self.models = []

    def fit(self, histories, targets, masks):
        features = aggregate(histories)
        for head in range(2):
            known = masks[:, head].astype(bool)
            if len(np.unique(targets[known, head])) < 2:
                raise ValueError("Each training head requires both classes")
            model = make_pipeline(StandardScaler(), LogisticRegression(C=self.regularization, max_iter=1000))
            model.fit(features[known], targets[known, head])
            self.models.append(model)
        return self

    def predict_proba(self, histories):
        features = aggregate(histories)
        return np.stack([model.predict_proba(features)[:, 1] for model in self.models], axis=1)


class LSTM(nn.Module):
    def __init__(self, hidden_size=16, heads=2):
        super().__init__()
        self.hidden_size, self.heads = hidden_size, heads
        self.encoder = nn.LSTM(2 * N, hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, heads)

    def forward(self, sequence):
        _, (state, _) = self.encoder(sequence)
        return self.output(state[-1])

    def predict_proba(self, histories):
        self.eval()
        with torch.inference_mode():
            return self(torch.as_tensor(histories, dtype=torch.float32)).sigmoid().numpy()


class SeparateLSTM:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, histories):
        return np.concatenate([model.predict_proba(histories) for model in self.models], axis=1)


def masked_bce(logits, targets, masks):
    """Equal head weighting over observed labels; missing B has zero gradient."""
    observed = masks.bool()
    safe_targets = torch.where(observed, targets, torch.zeros_like(targets))
    losses = nn.BCEWithLogitsLoss(reduction="none")(logits, safe_targets)
    counts = masks.sum(dim=0)
    per_head = (losses * masks).sum(dim=0) / counts.clamp_min(1)
    return per_head.sum() / (counts > 0).sum().clamp_min(1)
