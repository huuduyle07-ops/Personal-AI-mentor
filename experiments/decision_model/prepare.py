"""Author and seal synthetic train/validation/test files before model training."""
import argparse
import json
from pathlib import Path

from experiments.decision_model.data import DEFAULT_CONFIG, family_split, generate_partition, load_config, sha, write_jsonl


def prepare(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Dataset already sealed; choose a new output directory")
    manifest = {"version": "0.1", "config": config, "family_split": family_split(config), "files": {}, "counts": {},
                "sealing": "Process-level test embargo; no cryptographic access control. Training loads train/validation only."}
    for split in ("train", "validation", "test"):
        natural, paired = generate_partition(config, split)
        for key, rows in ((split, natural), (split + "_pairs", paired)):
            path = output / f"{key}.jsonl"
            write_jsonl(path, rows)
            manifest["files"][key] = sha(path)
            manifest["counts"][key] = {"families": len({row["family"] for row in rows}),
                                       "personas": len({row["persona_id"] for row in rows}),
                                       "episodes": len({row["episode_id"] for row in rows}), "samples": len(rows)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("experiments/decision_model/artifacts/data"))
    args = parser.parse_args()
    manifest = prepare(args.output, load_config(args.config))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
