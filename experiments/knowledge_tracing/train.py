"""Freeze all choices using train/validation only, before opening held-out learners."""
from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import load_split, make_windows
from .features import FeatureEncoder
from .metrics import scores
from .models import FittedPredictor
from .models.gru import RecurrentModel
from .models.logistic import make_logistic
from .models.statistical import StatisticalModel
from .utils import (DEFAULT_ARTIFACTS, DEFAULT_CONFIG, MemoryMonitor, digest, environment,
                    read_json, source_hashes, write_json)


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fit_recurrent(kind, train, validation, encoder, length, mode, config):
    seed_everything(config["seed"], config["threads"])
    model = RecurrentModel(len(encoder.event_names), train.summary.shape[1], train.context.shape[1],
                           config["hidden_size"], kind)
    predictor = FittedPredictor(kind, model, encoder, length, mode, config["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True,
                        num_workers=0, generator=generator)
    best, stale, curves, state = float("inf"), 0, [], None
    best_epoch = None
    for epoch in range(config["epochs"]):
        model.train()
        total = 0.0
        for x, lengths, summary, context, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths, summary, context)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        validation_scores = scores(validation.y, predictor.predict_windows(validation))
        value = validation_scores[config["selection_metric"]]
        curves.append({"epoch": epoch+1, "train_log_loss": total / len(train),
                       "validation_brier": validation_scores["brier"],
                       "validation_log_loss": validation_scores["log_loss"]})
        if value < best - 1e-6:
            best, stale, best_epoch = value, 0, epoch+1
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(state)
    return predictor, curves, best_epoch


def fit_candidate(kind, train, validation, encoder, length, mode, config, prior, value=None):
    curves, epoch = [], None
    start = time.perf_counter()
    with MemoryMonitor() as memory:
        if kind == "statistical":
            model = StatisticalModel(prior, value)
            predictor = FittedPredictor(kind, model, encoder, length, mode, config["seed"])
        elif kind == "logistic":
            model = make_logistic(value, config["seed"])
            model.fit(train.tabular, train.y)
            predictor = FittedPredictor(kind, model, encoder, length, mode, config["seed"])
        else:
            predictor, curves, epoch = fit_recurrent(kind, train, validation, encoder, length, mode, config)
        validation_scores = scores(validation.y, predictor.predict_windows(validation), validation.groups,
                                   config["threshold"])
    parameters = (sum(p.numel() for p in predictor.model.parameters()) if kind in ("gru", "lstm", "mlp")
                  else (predictor.model.coef_.size + predictor.model.intercept_.size if kind == "logistic" else 2))
    record = {"kind": kind, "length": length, "mode": mode, "hyperparameter": value,
              "validation": validation_scores, "training_seconds": time.perf_counter()-start,
              "parameters": int(parameters), "selected_epoch": epoch, "curves": curves,
              "memory": memory.result}
    return predictor, record


def train(config, data, output):
    data, output = Path(data), Path(output)
    if (output / "selection.json").exists() or (output / "test_opened.json").exists():
        raise FileExistsError("Selection sealed; choose a fresh output for a new experiment")
    if config["device"] != "cpu":
        raise ValueError("This resource-limited pilot uses CPU; no CUDA required")
    seed_everything(config["seed"], config["threads"])
    output.mkdir(parents=True, exist_ok=True)
    # No code path here loads test.jsonl.
    train_histories, val_histories = load_split(data, "train"), load_split(data, "validation")
    train_ids = {h["user_id"] for h in train_histories}
    val_ids = {h["user_id"] for h in val_histories}
    if train_ids & val_ids:
        raise ValueError("Learner overlap between train and validation")
    manifest = read_json(data / "manifest.json")
    if "assignments" in manifest:
        groups = [set(manifest["assignments"][p]) for p in ("train", "validation", "test")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError("Learner assignments overlap")
        if train_ids != groups[0] or val_ids != groups[1]:
            raise ValueError("Loaded learners disagree with sealed assignments")
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Positive epochs and batch size required")
    encoder = FeatureEncoder.fit(train_histories)
    write_json(output / "encoder.json", asdict(encoder))
    write_json(output / "feature_contract.json", encoder.contract())
    prior = float(np.mean([e["correct"] for h in train_histories for e in h["events"]]))
    candidates, selected = [], {}
    with MemoryMonitor() as memory:
        for length in [0] + config["sequence_lengths"]:
            train_data = make_windows(train_histories, encoder, length, seed=config["seed"])
            val_data = make_windows(val_histories, encoder, length, seed=config["seed"])
            kinds = ["statistical"] if length == 0 else ["statistical", "logistic", "gru", "lstm"]
            for kind in kinds:
                values = (config["statistical_strength"] if kind == "statistical" else
                          config["logistic_c"] if kind == "logistic" else [None])
                for value in values:
                    predictor, record = fit_candidate(kind, train_data, val_data, encoder, length,
                                                       "full", config, prior, value)
                    record["id"] = f"{kind}_L{length}_{value}"
                    predictor.save(output / record["id"])
                    candidates.append(record)
                    if kind not in selected or record["validation"]["brier"] < selected[kind]["validation"]["brier"]:
                        selected[kind] = record
                    print(f'{record["id"]}: validation Brier={record["validation"]["brier"]:.6f}', flush=True)
            del train_data, val_data
        # A summary-only nonlinear model approximately matches GRU capacity.
        reference = selected["gru"]
        mlp_config = dict(config)
        denominator = 2 * len(encoder.summary_names) + len(encoder.category_names("target")) + 2
        mlp_config["hidden_size"] = max(1, round((reference["parameters"] - 1) / denominator))
        train_data = make_windows(train_histories, encoder, reference["length"], seed=config["seed"])
        val_data = make_windows(val_histories, encoder, reference["length"], seed=config["seed"])
        predictor, record = fit_candidate("mlp", train_data, val_data, encoder, reference["length"],
                                         "full", mlp_config, prior)
        record["id"] = f'mlp_capacity_L{reference["length"]}'
        predictor.save(output / record["id"])
        candidates.append(record)
        selected["mlp"] = record
        del train_data, val_data
        # Freeze main hyperparameters first. Controls are retrained at these fixed choices.
        controls = []
        for mode in ["last"] + config["ablations"]:
            for kind in ("logistic", "gru", "lstm"):
                main = selected[kind]
                train_data = make_windows(train_histories, encoder, main["length"], mode, config["seed"])
                val_data = make_windows(val_histories, encoder, main["length"], mode, config["seed"])
                predictor, record = fit_candidate(kind, train_data, val_data, encoder, main["length"],
                                                   mode, config, prior, main["hyperparameter"])
                record["id"] = f'{kind}_{mode}_L{main["length"]}'
                predictor.save(output / record["id"])
                controls.append(record)
                print(f'{record["id"]}: validation Brier={record["validation"]["brier"]:.6f}', flush=True)
                del train_data, val_data
    all_files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    selection = {"config": config, "data_manifest_sha256": digest(data / "manifest.json"),
                 "source_sha256": source_hashes(), "environment": environment(),
                 "prior": prior, "selection": "minimum validation Brier; threshold fixed at 0.5",
                 "selected": {k: v["id"] for k, v in selected.items()}, "candidates": candidates,
                 "controls": controls, "artifact_sha256": all_files, "resources": memory.result}
    write_json(output / "selection.json", selection)
    print("Selection sealed. Test has not been loaded.", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_ARTIFACTS / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "models")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sequence-length", type=int, choices=[16, 32, 64])
    args = parser.parse_args()
    config = read_json(args.config)
    for key in ("batch_size", "epochs"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    if args.sequence_length:
        config["sequence_lengths"] = [args.sequence_length]
    train(config, args.data, args.output)


if __name__ == "__main__":
    main()
