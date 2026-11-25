"""Load the provisional, scenario-aligned reference dataset."""

from pathlib import Path

from experiments.pm_sft.contract import dumps, parse

from .annotation import validate

ROOT = Path(__file__).parent
LANGUAGES = ("en", "vi", "ko")
ANCHOR_DAY = "2026-11-24"
ANCHOR_ZONE = "Asia/Seoul"


def load_reference():
    rows = []
    families = parse((ROOT / "reference.json").read_text(encoding="utf-8"))
    for family in families:
        for lang in LANGUAGES:
            text, intent_quotes, slot_quotes = family[lang]
            slots = {**family.get("s", {}), **{key: slot_quotes[key] for key in family.get("verbatim", [])}}
            row = {
                "id": f"{family['family']}/{lang}", "family": family["family"], "language": lang,
                "text": text, "pending": family.get("pending"), "anchor_day": ANCHOR_DAY,
                "anchor_zone": ANCHOR_ZONE, "review_status": "provisional",
                "gold": {"i": family["i"], "s": slots, "m": family.get("m", []),
                         "a": family.get("a", False), "e": {"i": intent_quotes, "s": slot_quotes}},
            }
            checks = validate(dumps(row["gold"]), row)
            if checks["errors"] or checks["policy_flags"]:
                raise ValueError(f"Invalid reference {row['id']}: {checks['errors'] + checks['policy_flags']}")
            rows.append(row)
    if len(rows) != 60 or len({row["id"] for row in rows}) != 60:
        raise ValueError("Reference must contain 20 distinct families with three languages each")
    return rows
