"""Observable event features with explicit masks and provenance, train-only scaling."""
import json
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np

from experiments.decision_model.simulator import rng

FEATURES = (
    ("goal_minutes", "user_reported"), ("effective_target", "user_reported/user_approved"),
    ("reported_minutes", "user_reported"), ("reported_ratio", "derived_from_user_reported"),
    ("report_available", "window_closed_system_observation"), ("conflict", "user_reported"),
    ("obstacle", "user_reported"), ("approved", "user_approved"), ("rejected", "user_rejected"),
    ("quiz_passed", "quiz_result_not_duration_proof"), ("routine_changed", "user_reported"),
    ("weekday_sin", "system_timestamp"), ("weekday_cos", "system_timestamp"),
    ("outside_window_report", "system_timestamp"),
)
N = len(FEATURES)
CONTINUOUS = (0, 1, 2, 3, 11, 12)
SOURCES = {"goal_declared": {"user_reported"}, "context_reported": {"user_reported"},
           "routine_changed": {"user_reported"}, "adjustment_decided": {"user_approved", "user_rejected"},
           "progress_reported": {"user_reported"}, "verification_result": {"quiz_result"}}


def raw_window(sample):
    cutoff = datetime.fromisoformat(sample["cutoff"])
    if cutoff.utcoffset() is None:
        raise ValueError("History cutoff must be timezone-aware")
    first = cutoff.date() - timedelta(days=13)
    values = np.zeros((14, N), dtype=np.float32)
    masks = np.zeros_like(values)
    # Absence of a report/approval in a completely observed, closed window is
    # known. Minutes/quiz/context are unknown unless explicitly observed.
    known = (4, 7, 8, 10, 11, 12, 13)
    masks[:, known] = 1
    for day in range(14):
        weekday = (first + timedelta(days=day)).weekday()
        values[day, 11:13] = (np.sin(weekday * 2 * np.pi / 7), np.cos(weekday * 2 * np.pi / 7))
    previous_stamp = None
    for event in sample["history"]:
        timestamp = datetime.fromisoformat(event["timestamp"])
        if timestamp.utcoffset() is None or timestamp > cutoff or timestamp.date() < first:
            raise ValueError("Event outside historical cutoff/window")
        if previous_stamp is not None and timestamp < previous_stamp:
            raise ValueError("Events must be chronological")
        previous_stamp = timestamp
        kind, source, payload = event["kind"], event["source"], event["payload"]
        if kind not in SOURCES or source not in SOURCES[kind]:
            raise ValueError("Unknown event kind or invalid provenance")
        index = (timestamp.date() - first).days
        if kind == "goal_declared":
            values[index, :2] = payload["minutes"]
            masks[index, :2] = 1
        elif kind == "context_reported":
            values[index, 5:7] = (payload["conflict"], payload["obstacle"])
            masks[index, 5:7] = 1
        elif kind == "routine_changed":
            values[index, 10] = 1
        elif kind == "adjustment_decided":
            if payload["decision"] == "approved":
                if source != "user_approved":
                    raise ValueError("Approval provenance mismatch")
                values[index, 1], values[index, 7] = payload["proposed_minutes"], 1
            elif payload["decision"] == "rejected":
                if source != "user_rejected":
                    raise ValueError("Rejection provenance mismatch")
                values[index, 8] = 1
            else:
                raise ValueError("Unknown adjustment decision")
        elif kind == "progress_reported":
            if not 0 <= payload["minutes"] <= 1440:
                raise ValueError("Invalid declared minutes")
            if time(18) <= timestamp.time() <= time(23):
                if not masks[index, 1] or values[index, 1] <= 0:
                    raise ValueError("Report without a valid goal")
                values[index, 2:5] = (payload["minutes"], payload["minutes"] / values[index, 1], 1)
                masks[index, 2:4] = 1
            else:
                values[index, 13] = 1
        elif kind == "verification_result":
            values[index, 9], masks[index, 9] = payload["passed"], 1
    return values, masks


def raw_arrays(samples):
    pairs = [raw_window(sample) for sample in samples]
    return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])


class FeatureEncoder:
    def fit(self, values, masks):
        self.mean = np.zeros(N, dtype=np.float32)
        self.scale = np.ones(N, dtype=np.float32)
        for column in CONTINUOUS:
            observed = values[:, :, column][masks[:, :, column].astype(bool)]
            if observed.size:
                self.mean[column] = observed.mean()
                self.scale[column] = max(float(observed.std()), 1e-6)
        return self

    def transform(self, values, masks):
        result = np.where(masks.astype(bool), (values - self.mean) / self.scale, 0)
        return np.concatenate([result, masks], axis=-1).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                                         "features": FEATURES}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text())
        instance = cls()
        instance.mean, instance.scale = np.array(state["mean"]), np.array(state["scale"])
        return instance


def aggregate(sequence):
    """Strong summary baseline: means, recency, trend and observed fractions."""
    values, masks = sequence[:, :, :N], sequence[:, :, N:]
    features = []
    for width in (14, 7, 3):
        features.append((values[:, -width:] * masks[:, -width:]).sum(axis=1) / masks[:, -width:].sum(axis=1).clip(1))
    positions = np.arange(values.shape[1], dtype=np.float32)[None, :, None]
    count = masks.sum(axis=1).clip(1)
    mean_time = (positions * masks).sum(axis=1) / count
    centered = positions - mean_time[:, None, :]
    slopes = (centered * values * masks).sum(axis=1) / (centered**2 * masks).sum(axis=1).clip(1)
    features.extend((values[:, -1], masks.mean(axis=1), slopes, masks[:, -1]))
    return np.concatenate(features, axis=1)


def ablate(sequence, mode, ids, seed):
    if mode == "full":
        return sequence.copy()
    if mode == "no_history":
        return sequence[:, -1:].copy()
    if mode != "shuffled":
        raise ValueError("Unknown history ablation")
    result = sequence.copy()
    for index, identifier in enumerate(ids):
        order = list(range(sequence.shape[1] - 1))
        rng(seed, identifier, "shuffle").shuffle(order)
        result[index, :-1] = sequence[index, order]
    return result


def drop_history_channels(values, masks, ids, seed, rate):
    """MCAR telemetry loss is unknown, never a falsely observed missing report."""
    values, masks = values.copy(), masks.copy()
    channels = (2, 3, 4, 9, 13)
    for index, identifier in enumerate(ids):
        draw = rng(seed, identifier, "telemetry_drop")
        for day in range(13):
            if draw.random() < rate:
                values[index, day, channels] = 0
                masks[index, day, channels] = 0
    return values, masks
