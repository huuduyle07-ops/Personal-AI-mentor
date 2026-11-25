from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecurrentModel(nn.Module):
    """Packed valid events + common engineered summaries/context, one binary logit."""

    def __init__(self, input_size, summary_size, context_size, hidden_size=32, kind="gru"):
        super().__init__()
        self.config = dict(input_size=input_size, summary_size=summary_size,
                           context_size=context_size, hidden_size=hidden_size, kind=kind)
        self.kind = kind
        if kind == "mlp":
            self.recurrent = None
            self.head = nn.Sequential(nn.Linear(summary_size + context_size, hidden_size),
                                      nn.ReLU(), nn.Linear(hidden_size, 1))
        else:
            cell = nn.GRU if kind == "gru" else nn.LSTM
            self.recurrent = cell(input_size, hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden_size + summary_size + context_size, 1)

    def forward(self, x, lengths, summary, context):
        if self.kind == "mlp":
            return self.head(torch.cat([summary, context], dim=1)).squeeze(-1)
        # Empty histories use a zero recurrent state; packed padding cannot affect h.
        nonempty = lengths > 0
        state = x.new_zeros((x.shape[0], self.config["hidden_size"]))
        if nonempty.any():
            packed = pack_padded_sequence(x[nonempty], lengths[nonempty].cpu(),
                                          batch_first=True, enforce_sorted=False)
            _, hidden = self.recurrent(packed)
            if self.kind == "lstm":
                hidden = hidden[0]
            state[nonempty] = hidden[-1]
        return self.head(torch.cat([state, summary, context], dim=1)).squeeze(-1)


def make_gru(**kwargs):
    return RecurrentModel(kind="gru", **kwargs)
