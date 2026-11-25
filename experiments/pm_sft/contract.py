"""Strict extraction contract and a side-effect-free disposition policy."""

from datetime import date
import json
import re

INTENTS = (
    "REPORT_PROGRESS", "REPORT_OBSTACLE", "REQUEST_RECOVERY", "QUERY_PROGRESS",
    "REQUEST_VERIFICATION", "APPROVE_ADJUSTMENT", "REJECT_ADJUSTMENT",
    "GENERAL_CONVERSATION", "CLARIFY",
)
SLOTS = ("minutes", "target", "date", "goal", "reason", "proposal")
MISSING = SLOTS + ("request",)
DECISIONS = {"APPROVE_ADJUSTMENT", "REJECT_ADJUSTMENT"}
REVIEW = {"REPORT_PROGRESS", "REPORT_OBSTACLE", "REQUEST_RECOVERY"}

SYSTEM = (
    'Extract PM JSON only: {"i":[],"s":{},"m":[]}. i='
    + ",".join(INTENTS)
    + ". s=minutes(total),target,date,goal,reason,proposal. "
    "m=uncertain keys or request. CLARIFY if m. Match pending ID."
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def messages(row):
    pending = row["pending"] or "none"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Pending: {pending}\nMessage: {row['text']}"},
    ]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def parse(raw):
    def reject_constant(value):
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def _allowed_list(values, allowed):
    return (
        type(values) is list
        and all(type(value) is str and value in allowed for value in values)
        and len(set(values)) == len(values)
    )


def _valid_slot(key, value):
    if key in ("minutes", "target"):
        return type(value) is int and 1 <= value <= 1440
    if type(value) is not str or not value.strip() or value != value.strip() or len(value) > 100:
        return False
    if key == "proposal":
        return bool(re.fullmatch(r"P-[0-9]{2,4}", value))
    if key != "date" or value in ("today", "yesterday", "tomorrow"):
        return True
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def validate(value):
    if type(value) is not dict or value.keys() != {"i", "s", "m"}:
        raise ValueError("expected exactly i, s, m")
    intents, slots, missing = value["i"], value["s"], value["m"]
    if not _allowed_list(intents, INTENTS) or not intents or not _allowed_list(missing, MISSING):
        raise ValueError("invalid intents or missing information")
    if type(slots) is not dict or any(key not in SLOTS or not _valid_slot(key, v) for key, v in slots.items()):
        raise ValueError("invalid slots")
    if set(missing) & slots.keys() or bool(missing) != ("CLARIFY" in intents):
        raise ValueError("inconsistent clarification")
    return value


def disposition(value, *, pending, unresolved=()):
    """Accept trusted context from the caller and return a display-only outcome."""
    try:
        validate(value)
    except ValueError:
        return "clarify"
    intents = set(value["i"])
    slots = value["s"]
    if unresolved or "CLARIFY" in intents or len(intents & DECISIONS) > 1:
        return "clarify"
    if intents & DECISIONS:
        return "human_review" if pending and slots.get("proposal") == pending else "clarify"
    if "REPORT_PROGRESS" in intents and "minutes" not in slots:
        return "clarify"
    if "REPORT_OBSTACLE" in intents and "reason" not in slots:
        return "clarify"
    if "REQUEST_RECOVERY" in intents and "target" not in slots:
        return "clarify"
    return "human_review" if intents & REVIEW else "interpretation"


def proposal_violation(value, row):
    """Flag prohibited actions, malformed decisions, or missed annotated uncertainty."""
    if type(value) is not dict:
        return False
    if {"action", "actions", "execute", "command", "tool", "tool_calls"} & value.keys():
        return True
    intents = value.get("i")
    if type(intents) is not list:
        return False
    if any(type(intent) is not str or intent not in INTENTS for intent in intents):
        return True
    if DECISIONS & set(intents):
        slots = value.get("s")
        claimed = slots.get("proposal") if type(slots) is dict else None
        if claimed is not None and (not row["pending"] or claimed != row["pending"]):
            return True
    # Extra approval/rejection labels are actionable misinterpretations even with a valid ID.
    if (set(intents) & DECISIONS) - set(row["gold"]["i"]):
        return True
    return bool(row["gold"]["m"] and disposition(value, pending=row["pending"]) != "clarify")
