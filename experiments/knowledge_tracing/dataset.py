from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import context_of, events_array, perturb
from .utils import digest, read_json


def load_split(data, split):
    data = Path(data)
    manifest = read_json(data / "manifest.json")
    path = data / f"{split}.jsonl"
    if digest(path) != manifest["files"][path.name]:
        raise ValueError("Prepared split hash mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Windows(Dataset):
    """One target per completed learner transition. No target outcome enters X."""

    def __init__(self, histories, encoder, length, mode="full", seed=42):
        if length < 1:
            raise ValueError("Positive sequence length required")
        n = sum(len(h["events"]) - 1 for h in histories)
        self.x = np.zeros((n, length, len(encoder.event_names)), np.float32)
        self.lengths = np.zeros(n, np.int64)
        self.summary = np.zeros((n, 2 * len(encoder.summary_names)), np.float32)
        self.context = np.zeros((n, len(encoder.category_names("target"))), np.float32)
        self.y = np.zeros(n, np.float32)
        self.groups = np.zeros(n, np.int64)
        self.history_counts = np.zeros(n, np.int64)
        self.statistics = np.zeros((n, 2), np.float32)  # correctness sum and observed label count
        self.formats = []
        k = 0
        for group, learner in enumerate(histories):
            events = learner["events"]
            bundles = events_array(events)
            for i in range(1, len(events)):
                window = perturb(bundles[max(0, i-length):i], mode, f'{learner["user_id"]}:{i}', seed)
                encoded = encoder.encode_events(window)
                self.x[k, :len(encoded)] = encoded
                self.lengths[k] = len(encoded)
                self.summary[k] = encoder.encode_summary(window, i)
                self.context[k] = encoder.encode_context(context_of(events[i]), mode)
                self.y[k], self.groups[k], self.history_counts[k] = events[i]["correct"], group, i
                labels = np.asarray([e["correct"] for e in window])
                self.statistics[k] = np.nansum(labels), np.isfinite(labels).sum()
                self.formats.append(events[i].get("format", "<missing>"))
                k += 1

    @property
    def tabular(self):
        return np.concatenate([self.summary, self.context], axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.x[index]), torch.tensor(self.lengths[index]),
                torch.from_numpy(self.summary[index]), torch.from_numpy(self.context[index]),
                torch.tensor(self.y[index]))


def make_windows(histories, encoder, length, mode="full", seed=42):
    data = Windows(histories, encoder, length or 1, mode, seed)
    if length == 0:
        offset = 0
        for learner in histories:
            labels = np.asarray([e["correct"] for e in learner["events"]])
            n = len(labels) - 1
            if mode in ("full", "shuffled"):
                data.statistics[offset:offset+n, 0] = np.cumsum(labels)[:-1]
                data.statistics[offset:offset+n, 1] = np.arange(1, len(labels))
            else:
                bundles = events_array(learner["events"])
                for i in range(1, len(labels)):
                    window = perturb(bundles[:i], mode, f'{learner["user_id"]}:{i}', seed)
                    observed = np.asarray([e["correct"] for e in window])
                    data.statistics[offset+i-1] = np.nansum(observed), np.isfinite(observed).sum()
            offset += n
    return data
