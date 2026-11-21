"""Deterministic proposals change only today's plan after explicit approval."""
from memory import record
from time_utils import now_kst, valid_day


class Recovery:
    def __init__(self, store):
        self.store = store

    def propose(self, user_id, day, reason):
        valid_day(day)
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
            raise ValueError("Describe the obstacle in 1 to 300 characters")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            plan = self.store.plan(db, user_id, day)
            db.execute(
                "UPDATE recovery_proposals SET status='cancelled',resolved_at=? WHERE user_id=? AND day=? AND status='pending'",
                (now_kst().isoformat(), user_id, day),
            )
            target = max(1, plan["target"] // 2)
            result = db.execute(
                """INSERT INTO recovery_proposals(user_id,day,reason,original_target,proposed_target,plan_revision,created_at)
                VALUES(?,?,?,?,?,?,?)""",
                (user_id, day, reason.strip(), plan["target"], target, plan["revision"], now_kst().isoformat()),
            )
            record(db, user_id, day, "recovery_proposed", {"proposal_id": result.lastrowid,
                   "reason": reason.strip(), "from": plan["target"], "to": target})
        return {"id": result.lastrowid, "original_target": plan["target"], "proposed_target": target,
                "explanation": f"Reduce today's target from {plan['target']} to {target} minutes to make room for the obstacle. "
                               "Your recurring goal stays the same. Approval is required."}

    def resolve(self, user_id, day, proposal_id, decision):
        valid_day(day)
        if decision not in {"approve", "reject", "cancel"}:
            raise ValueError("Choose approve, reject, or cancel")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
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
                db.execute("UPDATE daily_goals SET target=?,revision=revision+1 WHERE user_id=? AND day=?",
                           (proposal["proposed_target"], user_id, day))
            db.execute("UPDATE recovery_proposals SET status=?,resolved_at=? WHERE id=?",
                       (status, now_kst().isoformat(), proposal_id))
            record(db, user_id, day, "recovery_" + status,
                   {"proposal_id": proposal_id, "from": proposal["original_target"], "to": proposal["proposed_target"]})
        return status
