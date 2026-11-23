"""Explicit approval applies a daily reduction or a one-off move to tomorrow."""
from memory import record
from time_utils import following_day, now_kst, valid_day, valid_minutes


class Recovery:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def remaining_minutes(target, reported):
        return max(0, target - reported)

    def propose(self, user_id, day, reason, *, tomorrow=False, remaining_minutes=None, db=None):
        valid_day(day)
        if remaining_minutes is not None:
            valid_minutes(remaining_minutes, 1, 1440)
            if not tomorrow:
                raise ValueError("A transfer amount requires a move to tomorrow.")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
            raise ValueError("Describe the obstacle in 1 to 300 characters")
        with self.store.transaction(db) as db:
            plan = self.store.plan(db, user_id, day)
            if plan["target"] == 0:
                raise ValueError("Today's plan has no remaining target to reduce.")
            target = max(1, plan["target"] // 2)
            if tomorrow:
                report = db.execute("SELECT id,minutes FROM progress_events WHERE user_id=? AND day=? "
                                    "ORDER BY id DESC LIMIT 1", (user_id, day)).fetchone()
                if report is None:
                    raise ValueError("How many minutes have you studied today? Send today's total first.")
                target = report["minutes"]
                remaining = self.remaining_minutes(plan["target"], target)
                if remaining_minutes is not None and remaining_minutes != remaining:
                    raise ValueError("The requested minutes differ from the remaining target.")
                if remaining <= 0:
                    raise ValueError("Today's reported total already meets the target. There are no minutes left to move.")
                next_day = following_day(day)
                future = self._future_plan(db, user_id, next_day)
                if future["title"] != plan["title"] or future["target"] + remaining > 600:
                    raise ValueError("The remaining minutes do not fit tomorrow's goal. Choose a smaller plan.")
            db.execute(
                "UPDATE recovery_proposals SET status='cancelled',resolved_at=? WHERE user_id=? AND day=? AND status='pending'",
                (now_kst().isoformat(), user_id, day),
            )
            result = db.execute(
                """INSERT INTO recovery_proposals(user_id,day,reason,original_target,proposed_target,plan_revision,created_at)
                VALUES(?,?,?,?,?,?,?)""",
                (user_id, day, reason.strip(), plan["target"], target, plan["revision"], now_kst().isoformat()),
            )
            record(db, user_id, day, "recovery_proposed", {"proposal_id": result.lastrowid,
                   "reason": reason.strip(), "from": plan["target"], "to": target})
            if tomorrow:
                db.execute("INSERT INTO recovery_moves VALUES(?,?,?,?,?,?)",
                           (result.lastrowid, report["id"], next_day, future["title"], future["target"], future["revision"]))
                explanation = (f"Move the remaining {remaining} minutes to {next_day}? "
                               f"Today's target would be {target} minutes and tomorrow's would be "
                               f"{future['target'] + remaining}. Your recurring goal stays the same.")
            else:
                explanation = (f"Reduce today's target from {plan['target']} to {target} minutes to make room for the obstacle. "
                               "Your recurring goal stays the same. Approval is required.")
        proposal = {"id": result.lastrowid, "original_target": plan["target"], "proposed_target": target,
                    "explanation": explanation}
        if tomorrow:
            proposal.update(remaining=remaining, next_day=next_day, next_target=future["target"] + remaining)
        return proposal

    @staticmethod
    def _future_plan(db, user_id, day):
        snapshot = db.execute("SELECT title,target,revision FROM daily_goals WHERE user_id=? AND day=?",
                              (user_id, day)).fetchone()
        return snapshot or db.execute("SELECT title,target_minutes AS target,1 AS revision FROM goals WHERE user_id=?",
                                      (user_id,)).fetchone()

    def resolve(self, user_id, day, proposal_id, decision, *, db=None):
        valid_day(day)
        if decision not in {"approve", "reject", "cancel"}:
            raise ValueError("Choose approve, reject, or cancel")
        with self.store.transaction(db) as db:
            proposal = db.execute(
                "SELECT * FROM recovery_proposals WHERE id=? AND user_id=? AND day=?",
                (proposal_id, user_id, day),
            ).fetchone()
            if proposal is None or proposal["status"] != "pending":
                raise ValueError("No pending proposal found for you today")
            plan = self.store.plan(db, user_id, day)
            if decision == "approve" and plan["revision"] != proposal["plan_revision"]:
                raise ValueError("Your plan changed; request a new proposal with /recover")
            status = {"approve": "approved", "reject": "rejected", "cancel": "cancelled"}[decision]
            if decision == "approve":
                move = db.execute("SELECT * FROM recovery_moves WHERE proposal_id=?", (proposal_id,)).fetchone()
                if move:
                    latest = db.execute("SELECT MAX(id) FROM progress_events WHERE user_id=? AND day=?",
                                        (user_id, day)).fetchone()[0]
                    future = self._future_plan(db, user_id, move["next_day"])
                    if (latest != move["report_id"] or future["title"] != move["next_title"]
                            or future["target"] != move["next_target"] or future["revision"] != move["next_revision"]):
                        raise ValueError("Progress or tomorrow's plan changed. Please request a new proposal.")
                    self.store.plan(db, user_id, move["next_day"])
                    remaining = proposal["original_target"] - proposal["proposed_target"]
                    db.execute("UPDATE daily_goals SET target=target+?,revision=revision+1 WHERE user_id=? AND day=?",
                               (remaining, user_id, move["next_day"]))
                    record(db, user_id, move["next_day"], "minutes_rescheduled",
                           {"proposal_id": proposal_id, "from_day": day, "minutes": remaining})
                db.execute("UPDATE daily_goals SET target=?,revision=revision+1 WHERE user_id=? AND day=?",
                           (proposal["proposed_target"], user_id, day))
            db.execute("UPDATE recovery_proposals SET status=?,resolved_at=? WHERE id=?",
                       (status, now_kst().isoformat(), proposal_id))
            record(db, user_id, day, "recovery_" + status,
                   {"proposal_id": proposal_id, "from": proposal["original_target"], "to": proposal["proposed_target"]})
        return status
