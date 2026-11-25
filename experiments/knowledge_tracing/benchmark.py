"""Convenience entry point; individual stages remain usable offline."""
import argparse
from pathlib import Path

from .data_adapter import prepare
from .evaluate import evaluate
from .train import train
from .utils import DEFAULT_ARTIFACTS, DEFAULT_CONFIG, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_ARTIFACTS.parent / "cache")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    prepare(config, args.output / "data", args.cache, args.offline)
    train(config, args.output / "data", args.output / "models")
    evaluate(args.output / "data", args.output / "models", args.output / "evaluation")


if __name__ == "__main__":
    main()
