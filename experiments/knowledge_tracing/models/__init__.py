"""Architecture-independent probability interface and local artifact persistence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import torch

from ..features import FeatureEncoder, events_array, perturb
from ..utils import read_json, write_json


class Predictor(Protocol):
    def predict(self, history: list[dict], next_exercise_context: dict) -> float: ...


class FittedPredictor:
    def __init__(self, kind, model, encoder, length, mode="full", seed=42):
        self.kind, self.model, self.encoder = kind, model, encoder
        self.length, self.mode, self.seed = length, mode, seed

    def predict(self, history, next_exercise_context):
        events = events_array(history)
        if self.length:
            events = events[-self.length:]
        events = perturb(events, self.mode, "inference", self.seed)
        if self.kind == "statistical":
            return self.model.predict_events(events)
        summary = self.encoder.encode_summary(events, len(history))[None]
        context = self.encoder.encode_context(next_exercise_context, self.mode)[None]
        if self.kind == "logistic":
            return float(self.model.predict_proba(np.concatenate([summary, context], axis=1))[:, 1][0])
        x = self.encoder.encode_events(events)
        padded = np.zeros((1, max(1, self.length), len(self.encoder.event_names)), np.float32)
        padded[0, :len(x)] = x
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(padded), torch.tensor([len(x)]),
                                torch.from_numpy(summary), torch.from_numpy(context))
            return float(torch.sigmoid(logits)[0])

    def predict_windows(self, data, batch_size=512):
        if self.kind == "statistical":
            return self.model.predict_counts(data.statistics)
        if self.kind == "logistic":
            return self.model.predict_proba(data.tabular)[:, 1]
        self.model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(data), batch_size):
                sl = slice(start, start + batch_size)
                logits = self.model(torch.from_numpy(data.x[sl]), torch.from_numpy(data.lengths[sl]),
                                    torch.from_numpy(data.summary[sl]), torch.from_numpy(data.context[sl]))
                values.append(torch.sigmoid(logits).numpy())
        return np.concatenate(values)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {"kind": self.kind, "encoder": asdict(self.encoder), "length": self.length,
                    "mode": self.mode, "seed": self.seed}
        if self.kind in ("gru", "lstm", "mlp"):
            metadata["network"] = self.model.config
            torch.save(self.model.state_dict(), path / "weights.pt")
        else:
            joblib.dump(self.model, path / "model.joblib")
        write_json(path / "predictor.json", metadata)

    @classmethod
    def load(cls, path):
        # Only call after validating the trusted-local selection's artifact hashes.
        path = Path(path)
        metadata = read_json(path / "predictor.json")
        encoder = FeatureEncoder(**metadata["encoder"])
        if metadata["kind"] in ("gru", "lstm", "mlp"):
            from .gru import RecurrentModel
            model = RecurrentModel(**metadata["network"])
            model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
        else:
            model = joblib.load(path / "model.joblib")
        return cls(metadata["kind"], model, encoder, metadata["length"], metadata["mode"], metadata["seed"])
