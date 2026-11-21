"""Offline baseline implementing a common interface for later research models."""
from typing import Protocol

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from experiments.router.dataset import encode


class ExperimentalPolicy(Protocol):
    def predict(self, events: list[dict]) -> list[str]: ...


class Baseline:
    def __init__(self, *, seed=42, regularization=2.0):
        self.pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
            LogisticRegression(C=regularization, max_iter=1000, random_state=seed),
        )

    def fit(self, records):
        self.pipeline.fit([encode(row["input"]) for row in records], [row["label"] for row in records])
        return self

    def predict(self, events):
        return self.pipeline.predict([encode(event) for event in events]).tolist()

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        # Pickle-based artifacts must come from your own trusted experiment run.
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance
