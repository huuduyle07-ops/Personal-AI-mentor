"""Audit-only alternative: identical raw fields, separate feature transformers."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def columns(events):
    """No family, label, rule-derived flags, text-length or threshold features."""
    return np.asarray([
        [event["text"], event["command"] if event["command"] is not None else "<missing>",
         str(event["non_urgent"]), event["priority"],
         np.nan if event["minutes"] is None else event["minutes"]]
        for event in events
    ], dtype=object)


class MixedBaseline:
    def __init__(self, *, seed=42, regularization=2.0):
        self.pipeline = make_pipeline(
            ColumnTransformer([
                ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), 0),
                ("categories", OneHotEncoder(handle_unknown="ignore"), [1, 2, 3]),
                ("minutes", make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                    StandardScaler(),
                ), [4]),
            ]),
            LogisticRegression(C=regularization, max_iter=1000, random_state=seed),
        )

    def fit(self, records):
        self.pipeline.fit(columns([row["input"] for row in records]), [row["label"] for row in records])
        return self

    def predict(self, events):
        return self.pipeline.predict(columns(events)).tolist()
