"""Persistent daily check-ins, bounded retries, and conservative crash recovery."""
import asyncio
from datetime import datetime, timedelta

from core import daily_review
from memory import record
from time_utils import KST, now_kst, valid_time


class DeliveryError(Exception):
    def __init__(self, code="delivery_failed", *, retryable=False, uncertain=False):
        super().__init__(code)
        self.code = code if code in {"forbidden", "not_found", "delivery_failed", "unavailable"} else "delivery_failed"
        self.retryable, self.uncertain = retryable, uncertain


class CheckinScheduler:
    MAX_ATTEMPTS = 3
    CATCHUP = timedelta(hours=6)

    def __init__(self, store, allowed_user_id, send, shadow=None):
        self.store, self.allowed_user_id, self.send = store, allowed_user_id, send
        self.shadow = shadow  # optional checkin_policy.ShadowLearner: observes check-ins, never changes them

    def configure(self, user_id, *, enabled=True, midday="12:00", evening="21:00", now=None):
        self._authorize(user_id)
        valid_time(midday)
        valid_time(evening)
        if type(enabled) is not bool:
            raise ValueError("Enabled must be true or false")
        now = self._now(now)
        with self.store.connect() as db:
            db.execute(
                """INSERT INTO checkin_settings(user_id,enabled,paused,midday,evening,active_since)
                VALUES(?,?,0,?,?,?) ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,
                paused=0,midday=excluded.midday,evening=excluded.evening,active_since=excluded.active_since""",
                (user_id, int(enabled), midday, evening, now.isoformat()),
            )
            record(db, user_id, now.date().isoformat(), "checkins_configured",
                   {"enabled": enabled, "midday": midday, "evening": evening})

    def pause(self, user_id, paused=True, now=None):
        self._authorize(user_id)
        if type(paused) is not bool:
            raise ValueError("Paused must be true or false")
        now = self._now(now)
        with self.store.connect() as db:
            result = db.execute("UPDATE checkin_settings SET paused=?,active_since=? WHERE user_id=? AND enabled=1",
                                (int(paused), now.isoformat(), user_id))
            if not result.rowcount:
                raise ValueError("Opt in first with /checkins action:enable")
            record(db, user_id, now.date().isoformat(), "checkins_paused" if paused else "checkins_resumed", {})

    def status(self, user_id):
        self._authorize(user_id)
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM checkin_settings WHERE user_id=?", (user_id,)).fetchone()
            deliveries = [dict(item) for item in db.execute(
                "SELECT day,slot,status,attempts,error_code FROM checkin_deliveries WHERE user_id=? ORDER BY id DESC LIMIT 4",
                (user_id,),
            )]
        return {"settings": dict(row) if row else None, "deliveries": deliveries}

    def _authorize(self, user_id):
        if not self.allowed_user_id or user_id != self.allowed_user_id:
            raise PermissionError("This is a private assistant")

    @staticmethod
    def _now(now):
        now = now or now_kst()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Scheduler requires a timezone-aware datetime")
        return now.astimezone(KST)

    def _recover(self, now):
        # A claim is committed before sending. A crash can leave delivery uncertain.
        with self.store.connect() as db:
            stale = db.execute(
                "SELECT * FROM checkin_deliveries WHERE user_id=? AND status='sending' AND updated_at<?",
                (self.allowed_user_id, (now - timedelta(minutes=2)).isoformat()),
            ).fetchall()
            for row in stale:
                db.execute("UPDATE checkin_deliveries SET status='unknown',error_code='interrupted',updated_at=? WHERE id=?",
                           (now.isoformat(), row["id"]))
                record(db, row["user_id"], row["day"], "checkin_unknown", {"slot": row["slot"], "reason": "interrupted"}, "system")

    def _claim(self, user_id, day, slot, scheduled, now):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            setting = db.execute("SELECT * FROM checkin_settings WHERE user_id=?", (user_id,)).fetchone()
            if not setting or not setting["enabled"] or setting["paused"] or setting[slot] != scheduled.strftime("%H:%M"):
                return None
            if scheduled < datetime.fromisoformat(setting["active_since"]):
                return None
            row = db.execute("SELECT * FROM checkin_deliveries WHERE user_id=? AND day=? AND slot=?",
                             (user_id, day, slot)).fetchone()
            if row:
                if row["status"] != "failed" or row["attempts"] >= self.MAX_ATTEMPTS:
                    return None
                if not row["retry_at"] or now < datetime.fromisoformat(row["retry_at"]):
                    return None
                db.execute("UPDATE checkin_deliveries SET status='sending',attempts=attempts+1,updated_at=? WHERE id=?",
                           (now.isoformat(), row["id"]))
                delivery_id = row["id"]
            else:
                delivery_id = db.execute(
                    """INSERT INTO checkin_deliveries(user_id,day,slot,status,attempts,updated_at)
                    VALUES(?,?,?,'sending',1,?)""", (user_id, day, slot, now.isoformat()),
                ).lastrowid
            record(db, user_id, day, "checkin_attempt", {"slot": slot, "delivery_id": delivery_id}, "system")
            return delivery_id

    def _due(self, setting, slot, now):
        """The scheduled time this slot is due for, if now is within its catch-up window.

        The window may cross midnight (an evening slot at 23:30 can still go out at 01:00),
        and the delivery belongs to the day the slot was scheduled on.
        """
        hour, minute = map(int, setting[slot].split(":"))
        today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        for scheduled in (today, today - timedelta(days=1)):
            if scheduled <= now <= scheduled + self.CATCHUP:
                return scheduled
        return None

    def _active_settings(self):
        with self.store.connect() as db:
            return db.execute("SELECT * FROM checkin_settings WHERE user_id=? AND enabled=1 AND paused=0",
                              (self.allowed_user_id,)).fetchall()

    def _message(self, user_id, day, slot):
        if slot == "evening":
            return ("Evening review (KST)\n" + daily_review(self.store.summary(user_id, day))
                    + "\nHow did today go? You can reply here with your total or ask for a smaller target.")
        return ("How is studying going today? Tell me your total so far, for example "
                "“today's total is 20 minutes”. If something got in the way, you can tell me that too.")

    def _finish(self, user_id, day, slot, delivery_id, status, code, retry_at, now):
        with self.store.connect() as db:
            db.execute("UPDATE checkin_deliveries SET status=?,updated_at=?,retry_at=?,error_code=? WHERE id=?",
                       (status, now.isoformat(), retry_at, code, delivery_id))
            record(db, user_id, day, "checkin_" + status,
                   {"slot": slot, "delivery_id": delivery_id, "error_code": code}, "system")

    async def tick(self, now=None):
        # SQLite work runs in worker threads so a locked database never stalls the event loop.
        now = self._now(now)
        await asyncio.to_thread(self._recover, now)
        for setting in await asyncio.to_thread(self._active_settings):
            for slot in sorted(("midday", "evening"), key=lambda key: setting[key]):
                scheduled = self._due(setting, slot, now)
                if scheduled is None:
                    continue
                day, user_id = scheduled.date().isoformat(), setting["user_id"]
                delivery_id = await asyncio.to_thread(self._claim, user_id, day, slot, scheduled, now)
                if delivery_id is None:
                    continue
                code, retry_at = None, None
                try:
                    message = await asyncio.to_thread(self._message, user_id, day, slot)
                    await asyncio.wait_for(self.send(user_id, message), timeout=20)
                    status = "delivered"
                except DeliveryError as exc:
                    code = exc.code
                    status = "unknown" if exc.uncertain else ("failed" if exc.retryable else "blocked")
                    if status == "failed":
                        retry_at = (now + timedelta(minutes=5)).isoformat()
                except Exception:
                    # Never log exception text: it may contain private message content.
                    status, code = "unknown", "delivery_uncertain"
                await asyncio.to_thread(self._finish, user_id, day, slot, delivery_id, status, code, retry_at, now)
                if self.shadow and status == "delivered":
                    await asyncio.to_thread(self.shadow.observe, user_id, slot, "nudge_study", now)
        if self.shadow and self.allowed_user_id:
            await asyncio.to_thread(self.shadow.observe_silence, self.allowed_user_id, now)
            await asyncio.to_thread(self.shadow.resolve, self.allowed_user_id, now)
