"""Grouped splits, 14-day samples and hashes. No production state access."""
import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path

from experiments.decision_model.simulator import FAMILIES, KST, episode, persona, rng

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["history_days"] != 14 or config["episode_days"] < 15:
        raise ValueError("Benchmark v0.1 requires 14 historical days and a future day")
    for key in ("personas_per_family", "episodes_per_persona", "epochs", "batch_size", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["report_window_start"] != "18:00" or config["report_window_end"] != "23:00":
        raise ValueError("v0.1 reporting window is fixed at 18:00–23:00 KST")
    return config


def family_split(config):
    families = list(FAMILIES)
    rng(config["split_seed"], "family_assignment").shuffle(families)
    return {"train": families[:6], "validation": families[6:9], "test": families[9:]}


def samples_from_episode(item, config):
    samples = []
    for next_index in range(14, len(item["daily"])):
        future = item["daily"][next_index]
        forecast_day = datetime.fromisoformat(future["day"]).replace(tzinfo=KST)
        cutoff = forecast_day - timedelta(seconds=1)
        first = forecast_day - timedelta(days=14)
        history = [event for event in item["events"] if first <= datetime.fromisoformat(event["timestamp"]) <= cutoff]
        available = future["available"]
        samples.append({"id": f"{item['id']}:forecast:{future['day']}", "episode_id": item["id"],
                        "group_id": item["group_id"], "persona_id": item["persona_id"], "family": item["family"],
                        "pair_branch": item["kind"], "cutoff": cutoff.isoformat(),
                        "report_window_start": forecast_day.replace(hour=18).isoformat(),
                        "report_window_end": forecast_day.replace(hour=23).isoformat(),
                        "history": history, "targets": [available, future["complete"] if available else 0],
                        "target_mask": [1, available], "oracle": [future["oracle_a"], future["oracle_b"]]})
    return samples


def generate_partition(config, split):
    natural, paired = [], []
    for family in family_split(config)[split]:
        for index in range(config["personas_per_family"]):
            person = persona(config, family, index)
            for repetition in range(config["episodes_per_persona"]):
                natural.extend(samples_from_episode(episode(config, person, repetition), config))
            for branch in ("high", "low"):
                paired.extend(samples_from_episode(episode(config, person, branch=branch), config))
    return natural, paired


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_partition(folder, split, *, paired=False):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    key = split + ("_pairs" if paired else "")
    path = folder / f"{key}.jsonl"
    if sha(path) != manifest["files"][key]:
        raise ValueError("Dataset differs from its sealed manifest")
    return read_jsonl(path)


def identities(rows, key):
    return {row[key] for row in rows}
