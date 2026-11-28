"""Score vocabulary extraction from study notes against hand-labelled pairs.

A predicted pair is correct when its Korean matches a gold item exactly and at least one predicted
meaning is among that item's accepted meanings. Precision = correct / predicted, recall = matched
gold items / gold items. Korean-only scores ignore the meaning.

Text samples live in samples/ with labels in gold.json (frozen: samples.sha256). Photos can be
added to images/ as <name>.png|jpg plus <name>.json holding [[korean, [meanings]], ...]; they are
scored with the local vision model when it is installed and reported as skipped otherwise.

    uv run --locked python -m experiments.notes_extraction.evaluate [--output results.json]
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from config import Config
from study_notes import IMAGE_TYPES, NoteImporter, VisionUnavailable

HERE = Path(__file__).parent


def score(predicted, gold):
    gold_map = {korean: set(meanings) for korean, meanings in gold}
    correct = [k for k, meanings in predicted if k in gold_map and gold_map[k] & set(meanings)]
    found = {k for k, _ in predicted}
    return {"predicted": len(predicted), "gold": len(gold), "correct": len(correct),
            "korean_correct": len(found & set(gold_map)),
            "missed": sorted(set(gold_map) - set(correct)),
            "false_positives": [[k, m] for k, m in predicted if k not in gold_map or not gold_map[k] & set(m)]}


def totals(rows):
    predicted, gold = sum(r["predicted"] for r in rows), sum(r["gold"] for r in rows)
    correct, korean = sum(r["correct"] for r in rows), sum(r["korean_correct"] for r in rows)
    precision = correct / predicted if predicted else None
    recall = correct / gold if gold else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"files": len(rows), "gold_pairs": gold, "predicted_pairs": predicted, "precision": round(precision or 0, 4),
            "recall": round(recall or 0, 4), "f1": round(f1, 4),
            "korean_only_precision": round(korean / predicted, 4) if predicted else None,
            "korean_only_recall": round(korean / gold, 4) if gold else None}


SETS = {"samples": ("samples", "gold.json", "samples.sha256"), "heldout": ("heldout", "heldout_gold.json", "heldout.sha256")}


async def evaluate(config, which="samples"):
    folder, labels_file, lock_file = SETS[which]
    lock = (HERE / lock_file).read_text().split()[0]
    content = b"".join(path.read_bytes() for path in sorted((HERE / folder).iterdir())) + (HERE / labels_file).read_bytes()
    if hashlib.sha256(content).hexdigest() != lock:
        raise SystemExit("Samples or labels changed after they were frozen; refusing to score them.")
    importer = NoteImporter(config)
    gold = json.loads((HERE / labels_file).read_text(encoding="utf-8"))
    text_rows = {}
    for name, pairs in gold.items():
        predicted, _ = await importer.extract(name, (HERE / folder / name).read_bytes())
        text_rows[name] = score(predicted, pairs)
    image_rows, skipped = {}, None
    for image in sorted(p for p in (HERE / "images").glob("*") if p.suffix.lower() in IMAGE_TYPES):
        labels = image.with_suffix(".json")
        if not labels.exists():
            continue
        try:
            predicted, _ = await importer.extract(image.name, image.read_bytes())
        except VisionUnavailable as reason:
            skipped = str(reason)
            break
        image_rows[image.name] = score(predicted, json.loads(labels.read_text(encoding="utf-8")))
    report = {"text": {"overall": totals(list(text_rows.values())), "by_file": text_rows}}
    report["images"] = ({"overall": totals(list(image_rows.values())), "by_file": image_rows,
                         "vision_model": config.ollama_vision_model}
                        if image_rows else {"skipped": skipped or "No labelled photos in images/."})
    return report


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--set", dest="which", choices=sorted(SETS), default="samples",
                        help="samples: development set (parser was fixed on it); heldout: frozen unseen set")
    options = parser.parse_args()
    report = asyncio.run(evaluate(Config.from_env(), options.which))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if options.output:
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
