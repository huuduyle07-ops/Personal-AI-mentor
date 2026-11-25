"""Strict pre-outcome allowlist; learner IDs never enter numerical features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import stable_hash

HISTORY_FIELDS = ("correct", "response_time_seconds", "format", "session", "days")
CONTEXT_FIELDS = ("format", "session")
MODES = ("full", "last", "shuffled", "missing20", "missing50",
         "no_correctness", "no_time", "no_metadata")


def context_of(event):
    # Neither target days nor anything derived from a submission is admitted.
    return {field: event.get(field) for field in CONTEXT_FIELDS}


def events_array(history):
    """Bundles retain their *historical* incoming gap when reordered/deleted."""
    rows = []
    for i, event in enumerate(history):
        prev = history[i - 1] if i else None
        gap = (event["days"] - prev["days"]) * 86400 if prev else np.nan
        response = event.get("response_time_seconds")
        response = np.nan if response is None or response < 0 else np.log1p(response)
        rows.append({"correct": float(event["correct"]), "response": response,
                     "gap": np.log1p(gap) if np.isfinite(gap) and gap >= 0 else np.nan,
                     "format": event.get("format"), "session": event.get("session")})
    return rows


def perturb(events, mode, key, seed):
    events = [dict(event) for event in events]
    rng = np.random.default_rng(int(stable_hash("perturb", seed, key)[:16], 16))
    if mode == "last":
        events = events[-1:]
    elif mode == "shuffled":
        # Permute complete bundles, never columns independently. No position feature survives.
        events = [events[i] for i in rng.permutation(len(events))]
    elif mode.startswith("missing"):
        rate = int(mode.removeprefix("missing")) / 100
        events = [event for event in events if rng.random() >= rate]
    elif mode == "no_correctness":
        for event in events:
            event["correct"] = np.nan
    elif mode == "no_time":
        for event in events:
            event["response"] = event["gap"] = np.nan
    elif mode == "no_metadata":
        for event in events:
            event["format"] = event["session"] = None
    elif mode != "full":
        raise ValueError(mode)
    return events


def observed_mean(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else np.nan


@dataclass
class FeatureEncoder:
    categories: dict
    event_medians: list
    event_means: list
    event_scales: list
    summary_medians: list
    summary_means: list
    summary_scales: list

    @classmethod
    def fit(cls, histories):
        # Fit on training events only, before seeing validation/test.
        events = [e for h in histories for e in events_array(h["events"])]
        categories = {field: sorted({e[field] for e in events if e[field] is not None})
                      for field in CONTEXT_FIELDS}
        encoder = cls(categories, [], [], [], [], [], [])
        raw = np.asarray([[e["correct"], e["response"], e["gap"]] for e in events])
        encoder.event_medians, encoder.event_means, encoder.event_scales = fit_numeric(raw)
        summaries = []
        for learner in histories:
            encoded_events = events_array(learner["events"])
            for i in range(1, len(encoded_events)):
                summaries.append(encoder.raw_summary(encoded_events[max(0, i-64):i], i))
        encoder.summary_medians, encoder.summary_means, encoder.summary_scales = fit_numeric(summaries)
        return encoder

    @property
    def event_names(self):
        return ["history_correct", "history_log_response_seconds", "history_log_incoming_gap_seconds",
                "correct_missing", "response_missing", "gap_missing"] + self.category_names("history")

    def category_names(self, prefix):
        return [f"{prefix}_{f}={value}" for f in CONTEXT_FIELDS
                for value in ["<missing>", "<unknown>"] + self.categories[f]]

    @property
    def summary_names(self):
        return ["window_accuracy", "recent16_accuracy", "recent4_accuracy", "last_correct",
                "recent_half_minus_older_half", "linear_correctness_trend",
                "mean_log_response", "last_log_response", "last_historical_log_gap",
                "mean_log_gap", "log_observed_history_count"]

    def categorical(self, record):
        result = []
        for field in CONTEXT_FIELDS:
            values = self.categories[field]
            value = record.get(field)
            slot = 0 if value is None else (values.index(value) + 2 if value in values else 1)
            row = [0.0] * (len(values) + 2)
            row[slot] = 1.0
            result.extend(row)
        return result

    def encode_events(self, events):
        if not events:
            return np.zeros((0, len(self.event_names)), np.float32)
        raw = [[e["correct"], e["response"], e["gap"]] for e in events]
        numeric = transform_numeric(raw, self.event_medians, self.event_means, self.event_scales)
        return np.concatenate([numeric, np.asarray([self.categorical(e) for e in events])], axis=1).astype(np.float32)

    def raw_summary(self, events, prior_count=None):
        c = np.asarray([e["correct"] for e in events])
        half = max(1, len(c) // 2)
        trend = np.nan
        valid = np.flatnonzero(np.isfinite(c))
        if len(valid) > 1:
            x = valid / max(1, len(c) - 1)
            trend = float(np.dot(x - x.mean(), c[valid] - c[valid].mean()) / np.sum((x - x.mean())**2))
        return [observed_mean(c), observed_mean(c[-16:]), observed_mean(c[-4:]),
                c[-1] if len(c) else np.nan,
                observed_mean(c[half:]) - observed_mean(c[:half]), trend,
                observed_mean([e["response"] for e in events]),
                events[-1]["response"] if events else np.nan,
                events[-1]["gap"] if events else np.nan,
                observed_mean([e["gap"] for e in events]), np.log1p(len(events) if prior_count is None else prior_count)]

    def encode_summary(self, events, prior_count=None):
        return transform_numeric([self.raw_summary(events, prior_count)], self.summary_medians,
                                 self.summary_means, self.summary_scales)[0].astype(np.float32)

    def encode_context(self, context, mode="full"):
        clean = {field: context.get(field) for field in CONTEXT_FIELDS}
        if mode == "no_metadata":
            clean = dict.fromkeys(CONTEXT_FIELDS)
        return np.asarray(self.categorical(clean), np.float32)

    def contract(self):
        return {"history": self.event_names, "historical_summaries": self.summary_names,
                "summary_missing_indicators": True, "next_exercise": self.category_names("target"),
                "target_allowlist": list(CONTEXT_FIELDS), "target_time_excluded": True,
                "identifiers_as_features": False, "count": "prior interactions, excluding target",
                "gap": "incoming gap of a completed historical event, never target submission time"}


def fit_numeric(raw):
    raw = np.asarray(raw, dtype=float)
    medians = [float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.0 for col in raw.T]
    filled = np.where(np.isfinite(raw), raw, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales < 1e-6] = 1.0
    return list(medians), means.tolist(), scales.tolist()


def transform_numeric(raw, medians, means, scales):
    raw = np.asarray(raw, dtype=float)
    missing = ~np.isfinite(raw)
    filled = np.where(missing, medians, raw)
    return np.concatenate([(filled - means) / scales, missing.astype(float)], axis=1)
