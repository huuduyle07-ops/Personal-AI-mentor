import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core import GoalStore
from scheduler import CheckinScheduler, DeliveryError
from time_utils import KST

USER = 123
START = datetime(2026, 11, 22, 11, 0, tzinfo=KST)
NOON = START.replace(hour=12)


def test_restart_dedup_evening_and_authorization(store):
    send = AsyncMock()
    scheduler = CheckinScheduler(store, USER, send)
    scheduler.configure(USER, now=START)
    asyncio.run(scheduler.tick(NOON.astimezone(timezone.utc)))
    restart = CheckinScheduler(GoalStore(store.path), USER, send)
    asyncio.run(restart.tick(NOON + timedelta(minutes=1)))
    assert send.await_count == 1
    asyncio.run(restart.tick(NOON.replace(hour=21)))
    assert send.await_count == 2 and "Evening review" in send.call_args.args[1]
    with pytest.raises(PermissionError):
        restart.configure(999, now=START)


def test_restart_catches_up_same_day_only(store):
    send = AsyncMock()
    scheduler = CheckinScheduler(store, USER, send)
    scheduler.configure(USER, now=START)
    restart = CheckinScheduler(GoalStore(store.path), USER, send)
    asyncio.run(restart.tick(NOON + timedelta(hours=1)))
    assert send.await_count == 1
    asyncio.run(restart.tick(NOON + timedelta(days=1, hours=-2)))
    assert send.await_count == 1


def test_opt_in_pause_resume_and_disable(store):
    send = AsyncMock()
    scheduler = CheckinScheduler(store, USER, send)
    asyncio.run(scheduler.tick(NOON))
    assert send.await_count == 0
    scheduler.configure(USER, now=START)
    scheduler.pause(USER, True, START)
    asyncio.run(scheduler.tick(NOON))
    scheduler.pause(USER, False, NOON + timedelta(minutes=1))
    asyncio.run(scheduler.tick(NOON + timedelta(minutes=2)))
    assert send.await_count == 0  # no backlog when resuming
    asyncio.run(scheduler.tick(NOON.replace(hour=21)))
    assert send.await_count == 1
    scheduler.configure(USER, enabled=False, now=NOON)
    asyncio.run(scheduler.tick(NOON + timedelta(days=1)))
    assert send.await_count == 1


def test_bounded_definite_failure_retries(store):
    send = AsyncMock(side_effect=DeliveryError("unavailable", retryable=True))
    scheduler = CheckinScheduler(store, USER, send)
    scheduler.configure(USER, now=START)
    for minutes in (0, 1, 5, 6, 10, 15, 20):
        asyncio.run(scheduler.tick(NOON + timedelta(minutes=minutes)))
    assert send.await_count == 3
    row = scheduler.status(USER)["deliveries"][0]
    assert row["attempts"] == 3 and row["status"] == "failed"


@pytest.mark.parametrize("error,status", [(DeliveryError("forbidden"), "blocked"), (TimeoutError(), "unknown")])
def test_failed_discord_delivery_does_not_change_progress(store, error, status):
    store.set_goal(USER, "Korean", 30, "2026-11-22")
    send = AsyncMock(side_effect=error)
    scheduler = CheckinScheduler(store, USER, send)
    scheduler.configure(USER, now=START)
    asyncio.run(scheduler.tick(NOON))
    asyncio.run(scheduler.tick(NOON + timedelta(minutes=5)))
    assert send.await_count == 1
    assert scheduler.status(USER)["deliveries"][0]["status"] == status
    assert store.summary(USER, "2026-11-22")["status"] == "not_reported"


def test_crash_after_claim_never_resends(store):
    send = AsyncMock()
    scheduler = CheckinScheduler(store, USER, send)
    scheduler.configure(USER, now=START)
    scheduler._claim(USER, "2026-11-22", "midday", NOON, NOON)
    restart = CheckinScheduler(GoalStore(store.path), USER, send)
    asyncio.run(restart.tick(NOON + timedelta(minutes=3)))
    assert send.await_count == 0
    assert restart.status(USER)["deliveries"][0]["status"] == "unknown"


def test_concurrent_ticks_deduplicate(store):
    send = AsyncMock()
    first = CheckinScheduler(store, USER, send)
    second = CheckinScheduler(GoalStore(store.path), USER, send)
    first.configure(USER, now=START)

    async def run():
        await asyncio.gather(first.tick(NOON), second.tick(NOON))

    asyncio.run(run())
    assert send.await_count == 1


@pytest.mark.parametrize("value", ["25:00", "9:00", "12:60", "oops"])
def test_invalid_schedule(store, value):
    with pytest.raises(ValueError):
        CheckinScheduler(store, USER, AsyncMock()).configure(USER, midday=value)


def test_late_evening_slot_catches_up_after_midnight(store):
    send = AsyncMock()
    scheduler = CheckinScheduler(store, USER, send)
    scheduler.configure(USER, evening="23:30", now=START)
    asyncio.run(scheduler.tick(datetime(2026, 11, 23, 0, 30, tzinfo=KST)))
    assert send.await_count == 1 and "Evening review" in send.call_args.args[1]
    assert scheduler.status(USER)["deliveries"][0]["day"] == "2026-11-22"
    asyncio.run(scheduler.tick(datetime(2026, 11, 23, 5, 31, tzinfo=KST)))  # window closed, no duplicate
    assert send.await_count == 1
