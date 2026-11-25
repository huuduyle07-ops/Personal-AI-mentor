"""Train/validation selection only. Does not open the sealed test files."""
import argparse
import copy
import hashlib
import json
import random
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
import torch

from experiments.decision_model.data import load_partition, sha
from experiments.decision_model.features import FeatureEncoder, ablate, raw_arrays
from experiments.decision_model.metrics import evaluate_predictions, selection_loss
from experiments.decision_model.models import HistoricalProbability, LogisticHeads, LSTM, masked_bce

HERE = Path(__file__).resolve().parent


def code_hashes():
    paths = [*HERE.glob("*.py"), HERE / "pyproject.toml", HERE / "uv.lock"]
    return {path.name: sha(path) for path in sorted(paths)}


def arrays(rows):
    return np.asarray([row["targets"] for row in rows], dtype=np.float32), np.asarray([row["target_mask"] for row in rows], dtype=np.float32)


def setup_torch(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)


class MemoryMonitor:
    """Sample process RSS, not Python-only allocations or isolated model RAM."""
    def __enter__(self):
        self.process = psutil.Process()
        self.start = self.peak = self.process.memory_info().rss
        self.done = threading.Event()

        def sample():
            while not self.done.wait(.02):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.done.set()
        self.thread.join()

    def metrics(self):
        return {"rss_start_bytes": self.start, "rss_sampled_peak_bytes": self.peak,
                "rss_peak_delta_bytes": self.peak - self.start}


def fit_lstm(x, y, mask, val_x, val_y, val_mask, config, seed, hidden_size, head=None):
    setup_torch(config, seed)
    model = LSTM(hidden_size, heads=2 if head is None else 1)
    if head is not None:
        y, mask = y[:, head:head + 1], mask[:, head:head + 1]
        val_y, val_mask = val_y[:, head:head + 1], val_mask[:, head:head + 1]
    tx, ty, tm = [torch.as_tensor(array, dtype=torch.float32) for array in (x, y, mask)]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(seed + 1000)
    best, best_loss, best_epoch, losses = None, float("inf"), None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(tx), generator=generator)
        for indices in order.split(config["batch_size"]):
            if not tm[indices].sum():
                continue
            optimizer.zero_grad()
            loss = masked_bce(model(tx[indices]), ty[indices], tm[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        probabilities = model.predict_proba(val_x)
        if head is None:
            val_loss = selection_loss(val_y, val_mask, probabilities)
        else:
            from experiments.decision_model.metrics import head_metrics
            selected = val_mask[:, 0].astype(bool)
            val_loss = head_metrics(val_y[selected, 0], probabilities[selected, 0])["log_loss"]
        losses.append(val_loss)
        if val_loss < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
    model.load_state_dict(best)
    model.eval()
    return model, {"hidden_size": hidden_size, "selected_epoch": best_epoch,
                   "validation_selection_log_loss": best_loss, "validation_loss_by_epoch": losses}


def save_network(model, path):
    torch.save({"state_dict": model.state_dict(), "hidden_size": model.hidden_size, "heads": model.heads}, path)


def train(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("Training selection is sealed; choose a new output directory")
    manifest = json.loads((data / "manifest.json").read_text())
    config = manifest["config"]
    # These are deliberately the only dataset partitions read in this script.
    train_rows, val_rows = load_partition(data, "train"), load_partition(data, "validation")
    raw_x, raw_mask = raw_arrays(train_rows)
    raw_val, raw_val_mask = raw_arrays(val_rows)
    encoder = FeatureEncoder().fit(raw_x, raw_mask)
    encoder.save(output / "encoder.json")
    x, val_x = encoder.transform(raw_x, raw_mask), encoder.transform(raw_val, raw_val_mask)
    y, mask = arrays(train_rows)
    val_y, val_mask = arrays(val_rows)
    ids, val_ids = [row["id"] for row in train_rows], [row["id"] for row in val_rows]
    protocol = {"config": config, "dataset_manifest_sha256": sha(data / "manifest.json"),
                "source_hashes": code_hashes(), "shadow_mode": True,
                "selection": "train preprocessing only; validation log loss; no test reads; lower hidden size/C breaks ties",
                "embargo": "Do not inspect test outcomes or retune after evaluate.py opens test"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    entries = {}
    start = time.perf_counter()
    with MemoryMonitor() as monitor:
        for width in (0, 1, 7, 14):
            model = HistoricalProbability(width).fit(raw_x, raw_mask, y, mask)
            name = f"statistics_{width}"
            joblib.dump(model, output / f"{name}.joblib")
            entries[name] = {"kind": "statistics", "file": f"{name}.joblib", "width": width,
                             "validation": evaluate_predictions(val_y, val_mask, model.predict_raw(raw_val, raw_val_mask))}
        for mode in ("full", "no_history", "shuffled"):
            train_view = ablate(x, mode, ids, config["shuffle_seed"])
            val_view = ablate(val_x, mode, val_ids, config["shuffle_seed"])
            candidates = []
            for c in config["logistic_C"]:
                model = LogisticHeads(c).fit(train_view, y, mask)
                candidates.append((selection_loss(val_y, val_mask, model.predict_proba(val_view)), c, model))
            score, c, model = min(candidates, key=lambda row: (row[0], row[1]))
            name = f"logistic_{mode}"
            joblib.dump(model, output / f"{name}.joblib")
            entries[name] = {"kind": "logistic", "file": f"{name}.joblib", "mode": mode, "C": c,
                             "validation_selection_log_loss": score,
                             "candidate_validation": [{"C": candidate[1], "log_loss": candidate[0]} for candidate in candidates],
                             "validation": evaluate_predictions(val_y, val_mask, model.predict_proba(val_view))}
            for seed in config["model_seeds"]:
                candidates = []
                for hidden_size in config["hidden_sizes"]:
                    joint, details = fit_lstm(train_view, y, mask, val_view, val_y, val_mask, config, seed, hidden_size)
                    candidates.append((details["validation_selection_log_loss"], hidden_size, joint, details))
                _, _, joint, details = min(candidates, key=lambda row: (row[0], row[1]))
                name = f"lstm_joint_{mode}_s{seed}"
                save_network(joint, output / f"{name}.pt")
                entries[name] = {"kind": "lstm_joint", "file": f"{name}.pt", "mode": mode, "seed": seed,
                                 **details, "validation": evaluate_predictions(val_y, val_mask, joint.predict_proba(val_view))}
                separate_models, separate_details, files = [], [], []
                for head in range(2):
                    candidates = []
                    for hidden_size in config["hidden_sizes"]:
                        model, detail = fit_lstm(train_view, y, mask, val_view, val_y, val_mask, config, seed + head * 100,
                                                hidden_size, head)
                        candidates.append((detail["validation_selection_log_loss"], hidden_size, model, detail))
                    _, _, model, detail = min(candidates, key=lambda row: (row[0], row[1]))
                    file = f"lstm_separate_{mode}_s{seed}_h{head}.pt"
                    save_network(model, output / file)
                    files.append(file)
                    separate_models.append(model)
                    separate_details.append(detail)
                name = f"lstm_separate_{mode}_s{seed}"
                predictions = np.concatenate([model.predict_proba(val_view) for model in separate_models], axis=1)
                entries[name] = {"kind": "lstm_separate", "files": files, "mode": mode, "seed": seed,
                                 "head_selection": separate_details, "validation": evaluate_predictions(val_y, val_mask, predictions)}
                print(f"Selected {mode}, seed {seed} using validation only", flush=True)
    if protocol["source_hashes"] != code_hashes():
        raise RuntimeError("Benchmark source changed during training")
    artifacts = {file.name: sha(file) for file in output.iterdir() if file.suffix in {".pt", ".joblib"} or file.name == "encoder.json"}
    selection = {"protocol": protocol, "models": entries, "artifact_sha256": artifacts,
                 "training_seconds": time.perf_counter() - start, "memory": monitor.metrics()}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("experiments/decision_model/artifacts/data"))
    parser.add_argument("--output", type=Path, default=Path("experiments/decision_model/artifacts/models"))
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(f"Sealed {len(result['models'])} models; test has not been opened")


if __name__ == "__main__":
    main()
