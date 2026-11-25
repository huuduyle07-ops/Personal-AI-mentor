"""New, purely synthetic chronological simulator. No production imports or I/O."""
import hashlib
import math
import random
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9), "Asia/Seoul")
# Twelve authored temporal schedule mechanisms, not numbered text templates.
FAMILIES = (
    "steady_cadence", "alternating_duty", "weekly_peak", "deadline_ramp",
    "interruption_blocks", "routine_relocation", "weekend_obligations",
    "rotating_shift", "obligation_pulses", "late_reporting", "persistent_workload", "study_cycle",
)


def rng(seed, *keys):
    material = "|".join(map(str, (seed, *keys))).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def persona(config, family, index):
    draw = rng(config["persona_seed"], family, index)
    return {"id": f"{family}:person:{index}", "family": family,
            "report_bias": draw.gauss(.45, .8), "completion_bias": draw.gauss(.2, .7),
            "late_probability": draw.uniform(.02, .16), "context_missing": draw.uniform(.05, .25),
            "goal_minutes": draw.choice((20, 30, 45, 60)), "routine_phase": draw.randrange(7)}


def conflict_probability(family, day, previous):
    return {
        "steady_cadence": .12,
        "alternating_duty": .65 if day % 2 else .05,
        "weekly_peak": .6 if day % 7 in (1, 2) else .08,
        "deadline_ramp": .05 + .65 * (day % 7) / 6,
        "interruption_blocks": .7 if day % 9 in (3, 4, 5) else .05,
        "routine_relocation": .5 if 10 <= day % 28 < 16 else .1,
        "weekend_obligations": .7 if day % 7 in (5, 6) else .1,
        "rotating_shift": .35 + .3 * math.sin(day * math.pi / 3),
        "obligation_pulses": .8 if day % 11 in (1, 2) else .08,
        "late_reporting": .18,
        "persistent_workload": .75 if previous else .12,
        "study_cycle": .3 + .25 * math.sin(day * 2 * math.pi / 5),
    }[family]


def episode(config, person, index=0, branch=None):
    """Paired branches are history interventions, not edited future-labelled rows.

    Both branches use common future random draws. Hidden probabilities never
    enter emitted events; they are retained separately for diagnostic evaluation.
    """
    paired = branch is not None
    length = 15 if paired else config["episode_days"]
    group = f"{person['id']}:pair" if paired else f"{person['id']}:episode:{index}"
    start = datetime(2026, 1, 5, tzinfo=KST) + timedelta(days=index * 35)
    events, daily = [], []
    previous_conflict = False
    for day in range(length):
        stamp = start + timedelta(days=day)
        draw = rng(config["behavior_seed"], group, day)
        numeric = rng(config["augmentation_seed"], group, day)
        conflict = draw.random() < conflict_probability(person["family"], day + person["routine_phase"], previous_conflict)
        obstacle = draw.random() < (.12 + .45 * conflict)
        routine = day in (10, 20)
        decision = "none"
        if obstacle and draw.random() < .55:
            decision = "approved" if draw.random() < .65 else "rejected"
        context_seen = draw.random() >= person["context_missing"]
        base_target = person["goal_minutes"]
        target = max(1, base_target // 2) if decision == "approved" else base_target
        recent_reports = daily[-7:]
        report_rate = sum(row["available"] for row in recent_reports) / len(recent_reports) if recent_reports else .5
        recent_valid = [row for row in daily[-7:] if row["available"]]
        completion_rate = sum(row["complete"] for row in recent_valid) / len(recent_valid) if recent_valid else .5
        recent3 = daily[-3:]
        short_rate = sum(row["available"] for row in recent3) / len(recent3) if recent3 else .5
        # Only past outcomes and today's exogenous circumstances affect today's
        # probabilities. Tomorrow's context is not announced in today's history.
        routine_effect = -.5 if 10 <= day % 28 < 14 else .15 if 14 <= day % 28 < 20 else 0
        p_a = sigmoid(person["report_bias"] + 1.4 * (report_rate - .5) + .8 * (short_rate - .5)
                      - .9 * conflict - .3 * obstacle + routine_effect)
        p_b = sigmoid(person["completion_bias"] + 1.3 * (completion_rate - .5)
                      - 1.0 * conflict - .4 * obstacle + .65 * (decision == "approved") + routine_effect)
        u_report, u_complete, u_zero, u_late, u_verify = [draw.random() for _ in range(5)]
        available = u_report < p_a
        complete = u_complete < p_b
        # A reported zero is an observed non-completion; silence is unobserved.
        if available and u_zero < .08:
            complete, minutes = False, 0
        elif complete:
            minutes = target + numeric.randrange(0, 11)
        else:
            minutes = numeric.randrange(0, target)
        late_probability = person["late_probability"] + (.18 if person["family"] == "late_reporting" else 0)
        outside = not available and u_late < late_probability
        if paired and day < 13:
            available = branch == "high" or day % 5 == 0
            complete = branch == "high"
            minutes = target + 3 if complete else 0
            outside = False
        if paired and day == 13:
            # Identical last-day event histories in both branches, including time.
            available, complete, minutes = True, False, 10
            conflict, obstacle, routine, context_seen, decision, outside = False, False, False, True, "none", False
            base_target = target = 30
        day_events = []

        def emit(hour, minute, kind, source, payload):
            day_events.append({"timestamp": stamp.replace(hour=hour, minute=minute).isoformat(),
                               "kind": kind, "source": source, "payload": payload})

        emit(9, 0, "goal_declared", "user_reported", {"minutes": base_target})
        if context_seen:
            emit(10, 0, "context_reported", "user_reported", {"conflict": conflict, "obstacle": obstacle})
        if routine:
            emit(10, 30, "routine_changed", "user_reported", {"changed": True})
        if decision != "none":
            emit(11, 0, "adjustment_decided", "user_approved" if decision == "approved" else "user_rejected",
                 {"decision": decision, "proposed_minutes": max(1, base_target // 2)})
        if available:
            emit(20, 0, "progress_reported", "user_reported", {"minutes": minutes})
            if not (paired and day == 13) and u_verify < .35:
                quiz_passed = draw.random() < sigmoid(person["completion_bias"] + .5)
                emit(21, 0, "verification_result", "quiz_result", {"passed": quiz_passed})
        elif outside:
            emit(23, 30, "progress_reported", "user_reported", {"minutes": minutes})
        events.extend(day_events)
        # Zero override changes the conditional completion probability as well.
        daily.append({"day": stamp.date().isoformat(), "available": int(available),
                      "complete": int(complete) if available else None,
                      "effective_target": target, "oracle_a": p_a, "oracle_b": .92 * p_b})
        previous_conflict = conflict
    return {"id": group + (f":{branch}" if paired else ""), "group_id": group,
            "persona_id": person["id"], "family": person["family"],
            "kind": branch or "natural", "events": events, "daily": daily}
