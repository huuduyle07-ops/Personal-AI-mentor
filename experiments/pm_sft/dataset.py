"""Prepare, audit and seal local multilingual scenario families."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile
import unicodedata

from .contract import SYSTEM, disposition, dumps, validate

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CONFIG = ROOT / "config.json"
LANGUAGES = {"en": "en-US", "vi": "vi-VN", "ko": "ko-KR"}
PARTITIONS = ("train", "validation", "holdout")

# Reviewed aligned train families. Only PM-compatible intent/date semantics are mapped.
MASSIVE_MAP = {
    "45": ("general_greet", "GENERAL_CONVERSATION", None),
    "166": ("general_greet", "GENERAL_CONVERSATION", None),
    "719": ("general_greet", "GENERAL_CONVERSATION", None),
    "1100": ("general_greet", "GENERAL_CONVERSATION", "today"),
    "1921": ("general_greet", "GENERAL_CONVERSATION", None),
    "1274": ("calendar_query", "CLARIFY", "tomorrow"),
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reviewed_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        split, family, pending, intents, slots, missing, *utterances = line.split("|")
        if len(utterances) != len(LANGUAGES):
            raise ValueError(f"{family}: expected one utterance per language")
        slots = json.loads(slots)
        for index, (language, utterance) in enumerate(zip(LANGUAGES, utterances, strict=True)):
            gold = {
                "i": intents.split(","),
                "s": {key: value[index] if type(value) is list else value for key, value in slots.items()},
                "m": missing.split(",") if missing else [],
            }
            rows.append({
                "id": f"{family}:{language}", "family": family, "language": language,
                "split": split, "text": utterance, "pending": pending.strip() or None,
                "source": "authored", "gold": gold,
            })
    return rows


def massive_rows(path):
    rows = []
    with tarfile.open(path, "r:gz") as archive:
        for language, locale in LANGUAGES.items():
            member = next(item for item in archive if item.name.endswith(f"/{locale}.jsonl"))
            with archive.extractfile(member) as stream:
                for raw in stream:
                    item = json.loads(raw)
                    source_id = str(item["id"])
                    if source_id not in MASSIVE_MAP:
                        continue
                    source_intent, intent, relative_day = MASSIVE_MAP[source_id]
                    if (item["partition"] != "train" or item["intent"] != source_intent
                            or (relative_day and "[date :" not in item["annot_utt"])):
                        raise ValueError(f"Unexpected MASSIVE annotation for {source_id}")
                    family = f"massive-{source_id}"
                    rows.append({
                        "id": f"{family}:{language}", "family": family, "language": language,
                        "split": "train", "text": item["utt"], "pending": None,
                        "source": "massive-1.0", "gold": {
                            "i": [intent], "s": {"date": relative_day} if relative_day else {},
                            "m": ["request"] if intent == "CLARIFY" else [],
                        },
                    })
    if len(rows) != len(MASSIVE_MAP) * len(LANGUAGES):
        raise ValueError("Incomplete MASSIVE alignment")
    return rows


def audit(rows):
    ids, inputs = set(), set()
    family_splits, family_languages = {}, {}
    counts = {split: {"languages": Counter(), "intents": Counter(), "sources": Counter()}
              for split in PARTITIONS}
    for row in rows:
        split, family, language = row["split"], row["family"], row["language"]
        validate(row["gold"])
        pending = row["pending"]
        if pending is not None and (type(pending) is not str or not re.fullmatch(r"P-[0-9]{2,4}", pending)):
            raise ValueError(f"Invalid pending proposal in {row['id']}")
        if bool(row["gold"]["m"]) != (disposition(row["gold"], pending=pending) == "clarify"):
            raise ValueError(f"Gold and policy disagree for {row['id']}")
        normalized = unicodedata.normalize("NFKC", row["text"]).casefold()
        normalized = " ".join(normalized.split())
        key = (language, normalized)
        if not normalized or key in inputs or row["id"] in ids:
            raise ValueError(f"Empty or duplicate example {row['id']}")
        ids.add(row["id"])
        inputs.add(key)
        if family_splits.setdefault(family, split) != split:
            raise ValueError(f"Scenario family crosses partitions: {family}")
        languages = family_languages.setdefault(family, set())
        if language not in LANGUAGES or language in languages:
            raise ValueError(f"Unexpected or repeated language for {family}")
        languages.add(language)
        counts[split]["languages"][language] += 1
        counts[split]["intents"].update(row["gold"]["i"])
        counts[split]["sources"][row["source"]] += 1
    if any(values != LANGUAGES.keys() for values in family_languages.values()):
        raise ValueError("Expected aligned scenarios in all languages")
    for split, group in counts.items():
        group["families"] = sum(value == split for value in family_splits.values())
        if not group["families"]:
            raise ValueError(f"Empty partition: {split}")
    return counts


def protocol_hash():
    sha = sha256()
    sha.update(SYSTEM.encode("utf-8"))
    for name in ("config.json", "contract.py", "model.py", "evaluate.py"):
        # Hash text with universal newlines so a Git line-ending conversion is harmless.
        sha.update((ROOT / name).read_text(encoding="utf-8").encode("utf-8"))
    return sha.hexdigest()


def prepare(reviewed, massive, output):
    rows = reviewed_rows(reviewed)
    if massive:
        rows.extend(massive_rows(massive))
    counts = audit(rows)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split in PARTITIONS:
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                if row["split"] == split:
                    stream.write(dumps(row) + "\n")
        hashes[split] = digest(path)
    manifest = {
        "counts": counts, "sha256": hashes, "protocol_sha256": protocol_hash(),
        "reviewed_source_sha256": digest(reviewed),
        "massive_archive_sha256": digest(massive) if massive else None,
        "massive_ids": list(MASSIVE_MAP) if massive else [],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(directory, split):
    if split not in PARTITIONS:
        raise ValueError(f"Invalid partition {split}")
    manifest = read_json(directory / "manifest.json")
    path = directory / f"{split}.jsonl"
    if digest(path) != manifest["sha256"][split] or protocol_hash() != manifest["protocol_sha256"]:
        raise ValueError(f"Sealed dataset or evaluation protocol mismatch for {split}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--reviewed", type=Path, default=ARTIFACTS / "reviewed.psv")
    args.add_argument("--massive", type=Path, default=ARTIFACTS / "massive-1.0.tar.gz")
    args.add_argument("--authored-only", action="store_true")
    args.add_argument("--output", type=Path, default=ARTIFACTS / "dataset")
    options = args.parse_args()
    manifest = prepare(options.reviewed, None if options.authored_only else options.massive, options.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
