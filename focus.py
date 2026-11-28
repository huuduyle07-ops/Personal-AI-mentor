"""Today's focus: one read-only summary of study status, vocabulary queue and check-ins.

Everything here is a SELECT (through chat_context, the recall report and direct queries). It never
calls GoalStore.summary()/plan(), which create today's plan row, so asking for focus writes nothing.
"""
from datetime import date

import chat_context
import confusion
from checkin_policy import recent_obstacle
from personalize import MIN_EVENTS
from time_utils import now_kst


def snapshot(store, verification, user_id, now=None):
    now = now or now_kst()
    day = now.date().isoformat()
    facts = chat_context.snapshot(store, verification, user_id, day, now)
    recall = verification.recall_report(user_id, now=now, limit=10_000)
    with store.connect() as db:
        checkins = db.execute("SELECT enabled, paused, midday, evening FROM checkin_settings WHERE user_id=?",
                              (user_id,)).fetchone()
        shadow = db.execute("SELECT COUNT(resolved_at) AS scored, COUNT(*) - COUNT(resolved_at) AS waiting "
                            "FROM bandit_shadow WHERE user_id=?", (user_id,)).fetchone()
        last = db.execute("SELECT engaged, slot FROM bandit_shadow WHERE user_id=? AND resolved_at IS NOT NULL "
                          "ORDER BY resolved_at DESC, id DESC LIMIT 1", (user_id,)).fetchone()
        obstacle = recent_obstacle(db, user_id, now)
    mixups = confusion.pairs(store, user_id, limit=2)
    statuses = [item["status"] for item in recall["words"]]
    return {
        "day": day, "goal": facts["goal_title"], "target": facts["target"], "minutes": facts["minutes"],
        "streak": facts["streak"], "obstacle": obstacle,
        "due": statuses.count("due"), "resting": statuses.count("resting"), "not_due": statuses.count("not due"),
        "seen": recall["seen"], "deck": recall["deck"], "next": recall["next"],
        "model": recall["model"], "reviews": recall["reviews"], "confusions": mixups,
        "checkins": dict(checkins) if checkins else None,
        "shadow": {"scored": shadow["scored"], "waiting": shadow["waiting"],
                   "last_engaged": None if last is None else bool(last["engaged"])},
    }


def suggestion(f):
    """One short, low-pressure next step."""
    if f["goal"] is None:
        return "Whenever you like, set a daily goal with /setgoal (for example /setgoal Korean 20)."
    if f["obstacle"]:
        return (f"You mentioned being {f['obstacle']} recently, so go easy: a smaller target is fine "
                "(say “can today's goal be smaller”), or just rest.")
    if f["minutes"] is None:
        return "No pressure: if you get a moment, even a short session counts. Tell me your total afterwards."
    remaining = f["target"] - f["minutes"]
    if remaining > 0:
        return f"You're {remaining} minute{'s' if remaining != 1 else ''} from today's goal, only if it fits your day."
    if f["due"]:
        return (f"Goal met. If you feel like it, a quick quiz would review {f['due']} due "
                f"word{'s' if f['due'] != 1 else ''}; otherwise you're done for today.")
    return "Goal met and nothing is due. You're done for today: rest counts too."


def format_focus(f):
    when = date.fromisoformat(f["day"]).strftime("%b %d").replace(" 0", " ")
    lines = [f"Today's focus ({when})"]
    if f["goal"] is None:
        lines.append("• Study: no daily goal yet.")
    else:
        logged = "nothing logged yet" if f["minutes"] is None else f"{f['minutes']} / {f['target']} min logged"
        lines.append(f"• Study: {logged} for {f['goal']} · streak {f['streak']} day{'s' if f['streak'] != 1 else ''}")
    upcoming = f["next"]
    if upcoming is None:
        up_next = "no words in the deck"
    elif upcoming["policy"] == "review":
        up_next = f"{upcoming['word']} (review, {upcoming['predicted_recall']:.0%} predicted recall)"
    else:
        up_next = f"{upcoming['word']} (new word)"
    lines.append(f"• Words: {f['due']} due · {f['resting']} resting · {f['seen']} of {f['deck']} seen · next up: {up_next}")
    if f.get("confusions"):
        lines.append("• Mix-ups: " + "; ".join(confusion.contrast(pair) for pair in f["confusions"])
                     + " (try “quiz me in Korean”)")
    if f["model"] == "personalized":
        model = f"personalized from your {f['reviews']} reviews"
    elif f["model"] == "population":
        model = f"Duolingo population model ({f['reviews']} of {MIN_EVENTS} genuine reviews needed to personalize)"
    else:
        model = "fixed Leitner rule (model weights unavailable)"
    lines.append(f"• Recall model: {model}")
    c, s = f["checkins"], f["shadow"]
    if not c or not c["enabled"]:
        checkins = "off (shadow-mode learning starts when you turn them on with /checkins)"
    else:
        state = "paused" if c["paused"] else f"on at {c['midday']} and {c['evening']} KST"
        last = "" if s["last_engaged"] is None else f", last window: {'you engaged' if s['last_engaged'] else 'no activity'}"
        checkins = f"{state} · shadow mode: {s['scored']} scored, {s['waiting']} waiting{last}"
    lines.append(f"• Check-ins: {checkins}")
    lines.append(f"Suggestion: {suggestion(f)}")
    return "\n".join(lines)
