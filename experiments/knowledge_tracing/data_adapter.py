"""Pinned, column-projected HTTP Range Parquet acquisition; no full-file fallback."""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .utils import (DEFAULT_ARTIFACTS, DEFAULT_CONFIG, MemoryMonitor, digest,
                    read_json, stable_hash, write_json)

PARTITIONS = ("train", "validation", "test")
COLUMNS = ["user_id", "interaction_index", "days", "correct",
           "response_time_seconds", "session", "format"]
EVIDENCE = ["README.md", "LICENSE.md", "CITING.md", "convert_to_hf.py"]


def fetch(url, headers=None):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "knowledge-tracing-research/0.1", **(headers or {})}), timeout=45)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))


class RangeFile(io.RawIOBase):
    """Seekable exact-range reader. Disk cache supports strict offline reproduction."""

    def __init__(self, url, cache, offline=False):
        super().__init__()
        self.url, self.cache, self.offline = url, Path(cache), offline
        self.cache.mkdir(parents=True, exist_ok=True)
        self.position = 0
        self.network_bytes = 0
        meta = self.cache / "size.json"
        if meta.exists():
            self.size = read_json(meta)["size"]
        else:
            if offline:
                raise FileNotFoundError(f"Uncached remote metadata: {url}")
            with fetch(url + "?range=0-0", {"Range": "bytes=0-0"}) as response:
                match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match:
                    raise RuntimeError("Server did not honor Range; refusing full dataset download")
                self.size = int(match[1])
                self.network_bytes += len(response.read(2))
            write_json(meta, {"url": url, "size": self.size})

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        path = self.cache / f"{start}-{end}.bin"
        if path.exists():
            data = path.read_bytes()
        else:
            if self.offline:
                raise FileNotFoundError(f"Uncached range {path.name}; run preparation online first")
            with fetch(self.url + f"?range={start}-{end}", {"Range": f"bytes={start}-{end}"}) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                    raise RuntimeError("Mismatched Range response; full-file fallback forbidden")
                data = response.read(size + 1)
            if len(data) != size:
                raise RuntimeError("Incomplete or excessive range response")
            self.network_bytes += len(data)
            path.write_bytes(data)
        if len(data) != size:
            raise RuntimeError("Corrupt range cache")
        self.position += size
        return data


def reconstruct(rows):
    """Preserve official partition/file order; fail on ambiguity instead of sorting by days."""
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["user_id"], row["source_partition"], row["interaction_index"]
        if key in seen:
            raise ValueError("Duplicate source interaction key")
        seen.add(key)
        grouped[row["user_id"]].append(row)
    audit = {"learners": len(grouped), "equal_day_transitions": 0,
             "cross_partition_transitions": 0, "local_index_gaps": 0,
             "ordering": "official train -> validation -> test; original file order within each"}
    for history in grouped.values():
        for index, row in enumerate(history):
            if (row["correct"] not in (0, 1) or row["days"] is None
                    or not math.isfinite(row["days"]) or row["days"] < 0):
                raise ValueError("Missing/invalid label or ordering time")
            row["sequence_index"] = index
            if index == 0:
                continue
            prev = history[index - 1]
            if row["days"] < prev["days"]:
                raise ValueError("Nonmonotone learner time; refusing ambiguous merge")
            audit["equal_day_transitions"] += int(row["days"] == prev["days"])
            if PARTITIONS.index(row["source_partition"]) < PARTITIONS.index(prev["source_partition"]):
                raise ValueError("Reversed official source partitions")
            if row["source_partition"] == prev["source_partition"]:
                if row["interaction_index"] <= prev["interaction_index"]:
                    raise ValueError("Nonmonotone split-local index")
                audit["local_index_gaps"] += int(row["interaction_index"] != prev["interaction_index"] + 1)
            else:
                audit["cross_partition_transitions"] += 1
    return dict(grouped), audit


def learner_split(ids, seed, fractions=(0.7, 0.15, 0.15)):
    if len(ids) < 7 or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Need >=7 learners and split fractions summing to one")
    ordered = sorted(ids, key=lambda uid: stable_hash("split", seed, uid))
    ntrain, nval = int(len(ids) * fractions[0]), int(len(ids) * fractions[1])
    return dict(zip(PARTITIONS, (ordered[:ntrain], ordered[ntrain:ntrain+nval], ordered[ntrain+nval:])))


def split_statistics(histories):
    rows = [r for h in histories for r in h]
    spans = [h[-1]["days"] - h[0]["days"] for h in histories]
    import numpy as np
    return {"learners": len(histories), "interactions": len(rows),
            "prediction_examples": sum(max(0, len(h) - 1) for h in histories),
            "positive_rate_all_interactions": sum(r["correct"] for r in rows) / len(rows),
            "positive_rate_prediction_targets": float(np.mean([r["correct"] for h in histories for r in h[1:]])),
            "missing": {c: sum(r[c] is None for r in rows) for c in COLUMNS},
            "relative_days_min": min(r["days"] for r in rows),
            "relative_days_max": max(r["days"] for r in rows),
            "learner_span_days_min_median_max": [min(spans), float(np.median(spans)), max(spans)],
            "source_partition_counts": {p: sum(r["source_partition"] == p for r in rows) for p in PARTITIONS}}


def prepare(config, output, cache, offline=False):
    output, cache = Path(output), Path(cache)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared data is sealed; use a new output directory")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pin a full dataset commit SHA")
    base = f'https://huggingface.co/datasets/{config["dataset_id"]}/resolve/{config["revision"]}'
    cache = cache / config["revision"]
    evidence = cache / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    evidence_hashes = {}
    for name in EVIDENCE:
        path = evidence / name
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            with fetch(f"{base}/{name}") as response:
                path.write_bytes(response.read())
        evidence_hashes[name] = digest(path)
    if "CC BY-NC 4.0" not in (evidence / "LICENSE.md").read_text(encoding="utf-8"):
        raise ValueError("Unexpected license; audit revision before proceeding")
    sources, id_groups, universe, schemas = {}, {}, set(), {}
    with MemoryMonitor() as memory:
        for partition in PARTITIONS:
            remote = RangeFile(f"{base}/{partition}/data.parquet", cache / partition, offline)
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            sources[partition] = (remote, parquet)
            schemas[partition] = str(parquet.schema_arrow)
            groups = []
            for group in range(parquet.num_row_groups):
                ids = set(parquet.read_row_group(group, columns=["user_id"], use_threads=False)["user_id"].to_pylist())
                universe.update(ids)
                groups.append(ids)
            id_groups[partition] = groups
        selected = set(sorted(universe, key=lambda uid: stable_hash("sample", config["seed"], uid))[:config["learners"]])
        if len(selected) != config["learners"]:
            raise ValueError("Requested more learners than available")
        rows, offsets = [], defaultdict(int)
        import pyarrow as pa
        for partition in PARTITIONS:
            remote, parquet = sources[partition]
            for group, ids in enumerate(id_groups[partition]):
                count = parquet.metadata.row_group(group).num_rows
                if selected.intersection(ids):
                    table = parquet.read_row_group(group, columns=COLUMNS, use_threads=False)
                    source_rows = pa.array(range(offsets[partition], offsets[partition] + count), type=pa.int64())
                    table = table.append_column("source_row", source_rows)
                    table = table.filter(pc.is_in(table["user_id"], value_set=pa.array(sorted(selected))))
                    for row in table.to_pylist():
                        row["source_partition"] = partition
                        rows.append(row)
                offsets[partition] += count
        grouped, ordering = reconstruct(rows)
        assignments = learner_split(selected, config["seed"], config["split_fractions"])
        output.mkdir(parents=True, exist_ok=True)
        stats, hashes = {}, {}
        for partition, learners in assignments.items():
            path = output / f"{partition}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for uid in learners:
                    handle.write(json.dumps({"user_id": uid, "events": grouped[uid]}) + "\n")
            stats[partition] = split_statistics([grouped[uid] for uid in learners])
            hashes[path.name] = digest(path)
    provenance = {"dataset_id": config["dataset_id"], "revision": config["revision"],
                  "license": "CC BY-NC 4.0", "copyright": "2016 Duolingo, Inc.",
                  "source_doi": "10.7910/DVN/8SWHNO", "evidence_sha256": evidence_hashes,
                  "schemas": schemas, "universe_learners": len(universe), "source_rows": dict(offsets),
                  "sampling": "smallest SHA256(sample:seed:original_user_id) over union of source IDs",
                  "selection_sha256": stable_hash(*sorted(selected)),
                  "projected_columns": COLUMNS, "ordering": ordering,
                  "network_bytes_this_run": sum(r.network_bytes for r, _ in sources.values()),
                  "full_parquet_bytes": sum(r.size for r, _ in sources.values()),
                  "memory": memory.result}
    manifest = {"version": "0.1", "config": config, "provenance": provenance,
                "splits": stats, "files": hashes, "assignments": assignments}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"splits": stats, "network_bytes": provenance["network_bytes_this_run"]}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "data")
    parser.add_argument("--cache", type=Path, default=DEFAULT_ARTIFACTS.parent / "cache")
    parser.add_argument("--learners", type=int)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.learners:
        config["learners"] = args.learners
    prepare(config, args.output, args.cache, args.offline)


if __name__ == "__main__":
    main()
