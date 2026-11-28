"""Learned check-in timing: a disjoint LinUCB contextual bandit (plain Python).

At a decision point the policy chooses one of three actions:
    nudge_quiz   - suggest a quick vocabulary quiz
    nudge_study  - suggest logging or doing today's study
    stay_silent  - send nothing

Reward (what the policy is trained to maximise; see experiments/checkin_bandit/REPORT.md):
    +1 if the learner engages within 3 hours (logs study or answers a quiz), minus a nudge cost
    (0.15, or 0.45 if they recently reported being tired or busy). Staying silent costs nothing.

Disjoint LinUCB (Li et al., 2010): each action a keeps A_a = I + sum x x^T and b_a = sum r x.
    theta_a = A_a^-1 b_a,  score_a = theta_a . x  +  ALPHA * sqrt(x^T A_a^-1 x)
The first term is the predicted reward, the second the exploration bonus (uncertainty). Ties go to
the first action in ACTIONS, so a cold-start policy stays silent rather than nudging blindly.

This module only previews and learns. It never sends messages: the fixed check-ins in scheduler.py
are unchanged (AGENT.md: a learned decision model must not replace existing behaviour automatically).

Shadow mode (ShadowLearner) learns from what really happens: each delivered fixed check-in is logged
as a nudge_study decision and an 18:00 point with no message as stay_silent, with the context at that
moment. REWARD_WINDOW later, the outcome is scored with the reward above and the bandit is updated.
The bot never sends quiz nudges, so the nudge_quiz arm stays at cold start.
"""
import json
import logging
import math
from datetime import datetime, timedelta

import chat_context
from personalize import _solve
from time_utils import now_kst

ACTIONS = ("stay_silent", "nudge_study", "nudge_quiz")  # order = tie-break preference
FEATURES = ("bias", "afternoon", "evening", "night", "weekend", "goal_progress", "logged_today",
            "streak", "words_due", "tired_or_busy")
ALPHA = 0.3            # exploration strength; chosen on simulated validation learners (REPORT.md)
VERSION = 1            # stored state with another version or dimension is ignored (cold start)
OBSTACLE_WINDOW = timedelta(hours=24)
REWARD_WINDOW = timedelta(hours=3)
NUDGE_COST, ANNOYANCE = 0.15, 0.30   # same reward definition as experiments/checkin_bandit
SILENT_SLOT = (18, 20)                # hours: observe "no message" between 18:00 and 20:00
log = logging.getLogger(__name__)


# ---- context features (read-only) ------------------------------------------------------------

def time_bucket(now):
    hour = now.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def recent_obstacle(db, user_id, now):
    """'tired' or 'busy' if reported in the last OBSTACLE_WINDOW, else None. Read-only."""
    for row in db.execute("SELECT payload, created_at FROM events WHERE user_id=? AND kind='obstacle_reported' "
                          "ORDER BY id DESC LIMIT 20", (user_id,)):
        reason = json.loads(row["payload"]).get("reason")
        if reason in {"tired", "busy"} and timedelta(0) <= now - datetime.fromisoformat(row["created_at"]) <= OBSTACLE_WINDOW:
            return reason
    return None


def context(store, verification, user_id, now=None):
    """(feature dict, vector). Only SELECTs: computing the context never changes the database."""
    now = (now or now_kst())
    day = now.date().isoformat()
    facts = chat_context.snapshot(store, verification, user_id, day, now)
    with store.connect() as db:
        obstacle = recent_obstacle(db, user_id, now) is not None
    bucket = time_bucket(now)
    minutes, target = facts["minutes"], facts["target"]
    features = {
        "bias": 1.0,
        "afternoon": float(bucket == "afternoon"), "evening": float(bucket == "evening"), "night": float(bucket == "night"),
        "weekend": float(now.weekday() >= 5),
        "goal_progress": min(minutes / target, 1.5) if minutes is not None and target else 0.0,
        "logged_today": float(minutes is not None),
        "streak": min(facts["streak"], 14) / 14,
        "words_due": min(facts["words_due"], 20) / 20,
        "tired_or_busy": float(obstacle),
    }
    return features, [features[name] for name in FEATURES]


# ---- disjoint LinUCB -------------------------------------------------------------------------

class LinUCB:
    def __init__(self, dimension=len(FEATURES), alpha=ALPHA, actions=ACTIONS):
        self.d, self.alpha, self.actions = dimension, alpha, tuple(actions)
        self.A = {a: [[float(i == j) for j in range(dimension)] for i in range(dimension)] for a in self.actions}
        self.b = {a: [0.0] * dimension for a in self.actions}
        self.updates = {a: 0 for a in self.actions}
        self._inverse = {a: [row[:] for row in self.A[a]] for a in self.actions}  # A^-1, kept in sync

    def scores(self, x):
        """Per action: predicted reward (theta.x), exploration bonus, and their sum (UCB)."""
        out = {}
        for a in self.actions:
            inv = self._inverse[a]
            inv_x = [sum(inv[i][j] * x[j] for j in range(self.d)) for i in range(self.d)]
            theta = [sum(inv[i][j] * self.b[a][j] for j in range(self.d)) for i in range(self.d)]
            mean = sum(t * v for t, v in zip(theta, x))
            bonus = self.alpha * math.sqrt(max(0.0, sum(v * w for v, w in zip(x, inv_x))))
            out[a] = {"predicted_reward": mean, "exploration_bonus": bonus, "ucb": mean + bonus,
                      "updates": self.updates[a]}
        return out

    def choose(self, x):
        scores = self.scores(x)
        best = max(scores[a]["ucb"] for a in self.actions)
        # Deterministic: the first action (in ACTIONS order) within float noise of the best score.
        return next(a for a in self.actions if scores[a]["ucb"] >= best - 1e-12), scores

    def update(self, action, x, reward):
        """A_a += x x^T, b_a += r x; A_a^-1 is updated with the Sherman–Morrison formula."""
        if action not in self.A or len(x) != self.d or not math.isfinite(reward):
            raise ValueError("Unknown action or malformed update")
        A, b, inv = self.A[action], self.b[action], self._inverse[action]
        for i in range(self.d):
            b[i] += reward * x[i]
            for j in range(self.d):
                A[i][j] += x[i] * x[j]
        inv_x = [sum(inv[i][j] * x[j] for j in range(self.d)) for i in range(self.d)]
        denominator = 1.0 + sum(v * w for v, w in zip(x, inv_x))
        for i in range(self.d):
            for j in range(self.d):
                inv[i][j] -= inv_x[i] * inv_x[j] / denominator
        self.updates[action] += 1

    def theta(self, action):
        return _solve(self.A[action], self.b[action])

    # ---- persistence ----

    def to_rows(self):
        return {a: {"A": self.A[a], "b": self.b[a], "updates": self.updates[a]} for a in self.actions}

    @classmethod
    def from_rows(cls, rows, alpha=ALPHA):
        policy = cls(alpha=alpha)
        for action, state in rows.items():
            A, b = state["A"], state["b"]
            if (action not in policy.actions or len(b) != policy.d or len(A) != policy.d
                    or any(len(row) != policy.d for row in A)
                    or not all(math.isfinite(v) for row in A for v in row) or not all(math.isfinite(v) for v in b)):
                raise ValueError("Stored bandit state does not match this policy")
            policy.A[action], policy.b[action] = [list(map(float, r)) for r in A], list(map(float, b))
            policy.updates[action] = int(state["updates"])
            policy._inverse[action] = _inverse(policy.A[action])
        return policy


def _inverse(matrix):
    n = len(matrix)
    columns = [_solve(matrix, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def load(db, user_id, alpha=ALPHA):
    """The user's stored policy, or a cold-start policy if none (or it doesn't match this version)."""
    rows = db.execute("SELECT action, A, b, updates FROM bandit_state WHERE user_id=? AND version=?",
                      (user_id, VERSION)).fetchall()
    if not rows:
        return LinUCB(alpha=alpha)
    try:
        return LinUCB.from_rows({row["action"]: {"A": json.loads(row["A"]), "b": json.loads(row["b"]),
                                                 "updates": row["updates"]} for row in rows}, alpha)
    except (ValueError, KeyError, TypeError):
        return LinUCB(alpha=alpha)


def save(db, user_id, policy):
    for action, state in policy.to_rows().items():
        db.execute("""INSERT INTO bandit_state(user_id, action, version, A, b, updates, updated_at)
                      VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id, action) DO UPDATE SET version=excluded.version,
                      A=excluded.A, b=excluded.b, updates=excluded.updates, updated_at=excluded.updated_at""",
                   (user_id, action, VERSION, json.dumps(state["A"]), json.dumps(state["b"]), state["updates"],
                    now_kst().isoformat()))


def learn(store, user_id, action, x, reward):
    """Record one observed outcome. Not called by the bot yet: shadow-mode logging is a future step."""
    with store.transaction() as db:
        policy = load(db, user_id)
        policy.update(action, x, reward)
        save(db, user_id, policy)


# ---- explainable preview ---------------------------------------------------------------------

def preview(store, verification, user_id, now=None):
    """What the policy would do right now and why. Read-only: nothing is sent or stored."""
    features, x = context(store, verification, user_id, now)
    with store.connect() as db:
        policy = load(db, user_id)
        shadow = db.execute("SELECT COUNT(resolved_at) AS learned, COUNT(*) - COUNT(resolved_at) AS waiting "
                            "FROM bandit_shadow WHERE user_id=?", (user_id,)).fetchone()
    action, scores = policy.choose(x)
    return {"features": features, "scores": scores, "action": action, "updates": sum(policy.updates.values()),
            "alpha": policy.alpha, "shadow": {"learned": shadow["learned"], "waiting": shadow["waiting"]}}


# ---- shadow mode -----------------------------------------------------------------------------

def reward(engaged, action, tired_or_busy):
    cost = 0.0 if action == "stay_silent" else NUDGE_COST + ANNOYANCE * bool(tired_or_busy)
    return float(engaged) - cost


class ShadowLearner:
    """Learns from real check-ins without changing them. Every failure is logged and swallowed, so
    shadow mode can never block or alter a check-in."""

    def __init__(self, store, verification):
        self.store, self.verification = store, verification

    def observe(self, user_id, slot, action, now):
        """Log a decision point that just happened (a delivered check-in, or planned silence)."""
        try:
            features, x = context(self.store, self.verification, user_id, now)
            with self.store.transaction() as db:
                choice = load(db, user_id).choose(x)[0]
                db.execute("INSERT OR IGNORE INTO bandit_shadow(user_id, day, slot, action, policy_choice, context, "
                           "decided_at) VALUES(?,?,?,?,?,?,?)", (user_id, now.date().isoformat(), slot, action,
                                                                 choice, json.dumps(x), now.isoformat()))
        except Exception as error:  # shadow mode must never affect check-ins
            log.warning("Shadow observe failed (%s)", type(error).__name__)

    def observe_silence(self, user_id, now):
        """At the 18:00 point, if check-ins are on and nothing is sent, log a stay_silent observation."""
        if not SILENT_SLOT[0] <= now.hour < SILENT_SLOT[1]:
            return
        with self.store.connect() as db:
            setting = db.execute("SELECT enabled, paused FROM checkin_settings WHERE user_id=?", (user_id,)).fetchone()
        if setting and setting["enabled"] and not setting["paused"]:
            self.observe(user_id, "silent", "stay_silent", now)

    def resolve(self, user_id, now):
        """Score observations whose reward window has ended and update the bandit. Returns how many."""
        try:
            with self.store.transaction() as db:
                waiting = db.execute("SELECT * FROM bandit_shadow WHERE user_id=? AND resolved_at IS NULL ORDER BY id",
                                     (user_id,)).fetchall()
                due = [row for row in waiting if datetime.fromisoformat(row["decided_at"]) + REWARD_WINDOW <= now]
                if not due:
                    return 0
                policy = load(db, user_id)
                for row in due:
                    start = datetime.fromisoformat(row["decided_at"])
                    engaged = engaged_between(db, user_id, start, start + REWARD_WINDOW)
                    x = json.loads(row["context"])
                    r = reward(engaged, row["action"], x[FEATURES.index("tired_or_busy")])
                    policy.update(row["action"], x, r)
                    db.execute("UPDATE bandit_shadow SET reward=?, engaged=?, resolved_at=? WHERE id=?",
                               (r, int(engaged), now.isoformat(), row["id"]))
                save(db, user_id, policy)
                return len(due)
        except Exception as error:
            log.warning("Shadow resolve failed (%s)", type(error).__name__)
            return 0


def engaged_between(db, user_id, start, end):
    """Did the user log study or answer a quiz in [start, end)? Timestamps share one ISO format (KST)."""
    lo, hi = start.isoformat(), end.isoformat()
    return bool(db.execute(
        "SELECT 1 FROM progress_events WHERE user_id=? AND created_at>=? AND created_at<? "
        "UNION SELECT 1 FROM verification_attempts WHERE user_id=? AND answered_at>=? AND answered_at<? LIMIT 1",
        (user_id, lo, hi, user_id, lo, hi)).fetchone())


def format_preview(report):
    f = report["features"]
    bucket = next((name for name in ("afternoon", "evening", "night") if f[name]), "morning")
    lines = ["Check-in policy preview (LinUCB; nothing is sent, your fixed check-ins are unchanged).",
             f"Context: {bucket}{', weekend' if f['weekend'] else ''}, goal progress {f['goal_progress']:.0%}"
             f"{'' if f['logged_today'] else ' (nothing logged today)'}, streak {round(f['streak'] * 14)} day(s), "
             f"{round(f['words_due'] * 20)} word(s) due, {'recently tired/busy' if f['tired_or_busy'] else 'no recent obstacle'}."]
    for action in ACTIONS:
        s = report["scores"][action]
        lines.append(f"{action}: predicted reward {s['predicted_reward']:+.2f} + exploration bonus "
                     f"{s['exploration_bonus']:.2f} = {s['ucb']:.2f} ({s['updates']} updates)")
    shadow = report.get("shadow", {"learned": 0, "waiting": 0})
    lines.append(f"Shadow mode: {shadow['learned']} real outcome(s) learned, {shadow['waiting']} waiting for their "
                 f"{int(REWARD_WINDOW.total_seconds() // 3600)}-hour window (quiz nudges are never sent, so that arm can't learn yet).")
    if report["updates"] == 0:
        lines.append("Cold start: no outcomes learned yet, so every action ties and the policy stays silent by default.")
    lines.append(f"Would choose now: {report['action']}.")
    return "\n".join(lines)
