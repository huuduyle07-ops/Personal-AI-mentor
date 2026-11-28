"""Read-only status facts that ground the conversational reply.

Everything here is a SELECT. It deliberately avoids GoalStore.summary()/plan(), which insert
today's plan row when it is missing: talking to the bot must never change the database.
"""
from datetime import date, timedelta


def _latest_minutes(db, user_id, day):
    row = db.execute("SELECT minutes FROM progress_events WHERE user_id=? AND day=? ORDER BY id DESC LIMIT 1",
                     (user_id, day)).fetchone()
    return row["minutes"] if row else None


def _target(db, user_id, day, default):
    row = db.execute("SELECT target FROM daily_goals WHERE user_id=? AND day=?", (user_id, day)).fetchone()
    return row["target"] if row else default


def streak(db, user_id, today, default_target, limit=365):
    """Consecutive days ending today (or yesterday, if today isn't done yet) whose reported total met the target."""
    met = lambda day: (m := _latest_minutes(db, user_id, day)) is not None and m >= _target(db, user_id, day, default_target)
    day = date.fromisoformat(today)
    if not met(today):
        day -= timedelta(days=1)
    count = 0
    while count < limit and met(day.isoformat()):
        count += 1
        day -= timedelta(days=1)
    return count


def snapshot(store, verification, user_id, today, now=None):
    with store.connect() as db:
        goal = db.execute("SELECT title, target_minutes FROM goals WHERE user_id=?", (user_id,)).fetchone()
        facts = {"date": today, "goal_title": None, "target": None, "minutes": None, "streak": 0}
        if goal:
            target = _target(db, user_id, today, goal["target_minutes"])
            facts.update(goal_title=goal["title"], target=target, minutes=_latest_minutes(db, user_id, today),
                         streak=streak(db, user_id, today, goal["target_minutes"]))
    report = verification.recall_report(user_id, now=now, limit=1000)
    facts.update(words_due=sum(item["status"] == "due" for item in report["words"]),
                 words_seen=report["seen"], deck_size=report["deck"])
    return facts


def describe(facts):
    """The facts as short lines for the model's system prompt."""
    if facts["goal_title"] is None:
        lines = ["No study goal has been set yet (they can set one with /setgoal)."]
    else:
        logged = ("nothing logged yet today" if facts["minutes"] is None
                  else f"{facts['minutes']} of {facts['target']} minutes logged today")
        lines = [f"Daily goal: {facts['goal_title']}, {facts['target']} minutes. Today ({facts['date']}): {logged}.",
                 f"Streak: {facts['streak']} day(s) in a row meeting the goal."]
    lines.append(f"Korean vocabulary: {facts['words_due']} word(s) due for review, "
                 f"{facts['words_seen']} of {facts['deck_size']} seen.")
    return "\n".join(lines)
