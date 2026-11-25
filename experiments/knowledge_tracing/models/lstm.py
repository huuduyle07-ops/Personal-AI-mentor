from .gru import RecurrentModel


def make_lstm(**kwargs):
    return RecurrentModel(kind="lstm", **kwargs)
