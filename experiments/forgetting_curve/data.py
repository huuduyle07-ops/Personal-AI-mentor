"""Stream a prefix of the Duolingo half-life regression learning traces and split it by learner.

Source: B. Settles and B. Meeder (2016), "A Trainable Spaced Repetition Model for Language Learning", ACL.
Data: Duolingo, CC BY-NC 4.0, https://doi.org/10.7910/DVN/N8XJME (settles.acl16.learning_traces.13m.csv.gz).

Only the first --rows rows are read; the download stops there, so the full 379 MB file is never fetched.
"""
import argparse
import csv
import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://dataverse.harvard.edu/api/access/datafile/3091087"
SOURCE = {"doi": "10.7910/DVN/N8XJME", "file": "settles.acl16.learning_traces.13m.csv.gz",
          "license": "CC BY-NC 4.0 (Duolingo)", "citation": "Settles & Meeder (2016), ACL"}
ARTIFACTS = Path(__file__).with_name("artifacts")
COLUMNS = ["p_recall", "timestamp", "delta", "user_id", "learning_language", "ui_language", "lexeme_id",
           "lexeme_string", "history_seen", "history_correct", "session_seen", "session_correct"]
SPLITS = ("train", "validation", "test")


class CountingReader:
    """Wraps the HTTP response to count compressed bytes actually downloaded."""

    def __init__(self, raw):
        self.raw, self.bytes = raw, 0

    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.bytes += len(chunk)
        return chunk


def download(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    request = urllib.request.Request(URL, headers={"User-Agent": "forgetting-curve-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        counter = CountingReader(response)
        with gzip.GzipFile(fileobj=counter) as stream, open(output, "w", newline="", encoding="utf-8") as out:
            reader = csv.reader(line.decode("utf-8") for line in stream)
            header = next(reader)
            if header != COLUMNS:
                raise ValueError(f"Unexpected columns: {header}")
            writer = csv.writer(out)
            writer.writerow(header)
            written = 0
            for row in reader:
                writer.writerow(row)
                written += 1
                if written >= rows:
                    break
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance = {**SOURCE, "url": URL, "rows": written, "compressed_bytes_read": counter.bytes,
                  "seconds": round(time.time() - start, 1), "sha256": digest}
    (output.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def split_of(user_id, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Deterministic learner-level split: every learner's rows land in exactly one partition."""
    value = int(hashlib.sha256(f"split:{seed}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return SPLITS[0] if value < fractions[0] else SPLITS[1] if value < fractions[0] + fractions[1] else SPLITS[2]


def load(path):
    """Load the sample into numpy arrays: one dict of arrays per split."""
    columns = {name: [] for name in ("p", "delta_days", "right", "wrong", "lexeme", "user", "split")}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seen, correct = int(row["history_seen"]), int(row["history_correct"])
            columns["p"].append(float(row["p_recall"]))
            columns["delta_days"].append(int(row["delta"]) / 86400)
            columns["right"].append(correct)
            columns["wrong"].append(seen - correct)
            columns["lexeme"].append(f'{row["learning_language"]}:{row["lexeme_id"]}')
            columns["user"].append(row["user_id"])
            columns["split"].append(split_of(row["user_id"]))
    split = np.array(columns.pop("split"))
    arrays = {key: np.array(value) for key, value in columns.items()}
    return {name: {key: value[split == name] for key, value in arrays.items()} for name in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "sample.csv")
    options = parser.parse_args()
    if options.output.exists():
        raise SystemExit(f"{options.output} exists; refusing to overwrite a sample that results may depend on.")
    print(json.dumps(download(options.rows, options.output), indent=2))


if __name__ == "__main__":
    main()
