import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

import bot as integration
from config import Config
from core import GoalStore
from memory import Memory
from scheduler import DeliveryError
from time_utils import KST


@pytest.fixture
def bot(store, monkeypatch):
    monkeypatch.setattr(integration, "today", lambda: "2026-11-22")
    return integration.create_bot(Config(token="synthetic-token", allowed_user_id=123), store)


def interaction(user_id=123):
    return SimpleNamespace(user=SimpleNamespace(id=user_id),
                           response=SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock(), is_done=lambda: False),
                           followup=SimpleNamespace(send=AsyncMock()))


@pytest.mark.parametrize("name,args", [
    ("setgoal", ("Korean", 30)), ("report", (30,)), ("summary", ()),
    ("checkins", ("enable",)), ("quiz", ()), ("answer", (1, "water")),
    ("recover", ("Work shift",)), ("adjust", (1, "approve")), ("history", ()),
    ("ask", ("Help with my plan",)),
])
def test_every_command_rejects_other_users(bot, name, args):
    denied = interaction(999)
    asyncio.run(bot.tree.get_command(name).callback(denied, *args))
    assert denied.response.send_message.call_args.args[0] == "This is a private assistant."
    assert Memory(bot.store).history(999) == []


def test_end_to_end_slash_workflow(bot):
    async def invoke(name, *args):
        request = interaction()
        await bot.tree.get_command(name).callback(request, *args)
        sent = request.response.send_message.call_args or request.followup.send.call_args
        return sent.args[0]

    async def run():
        assert "Goal saved" in await invoke("setgoal", "Korean", 30)
        bot.scheduler.send = AsyncMock()
        bot.scheduler.configure(123, now=datetime(2026, 11, 22, 11, tzinfo=KST))
        await bot.scheduler.tick(datetime(2026, 11, 22, 12, tzinfo=KST))
        assert bot.scheduler.send.await_count == 1
        assert "/quiz" in await invoke("report", 30)
        await invoke("quiz")
        with bot.store.connect() as db:
            attempt_id = db.execute("SELECT MAX(id) FROM verification_attempts").fetchone()[0]
        assert "verified" in await invoke("answer", attempt_id, "water")
        await invoke("recover", "Synthetic late work shift")
        with bot.store.connect() as db:
            proposal_id = db.execute("SELECT MAX(id) FROM recovery_proposals").fetchone()[0]
        assert "approved" in await invoke("adjust", proposal_id, "approve")
        review = await invoke("summary")
        assert "30 / 15" in review and "verified" in review and "Original daily target: 30" in review
        await invoke("ask", "Can you help me plan")
        await bot.scheduler.tick(datetime(2026, 11, 22, 21, tzinfo=KST))
        assert "30 / 15" in bot.scheduler.send.call_args.args[1]
        restart = integration.create_bot(bot.config, GoalStore(bot.store.path))
        restart.scheduler.send = AsyncMock()
        await restart.scheduler.tick(datetime(2026, 11, 22, 21, 1, tzinfo=KST))
        restart.scheduler.send.assert_not_called()
        assert restart.store.summary(123, "2026-11-22")["verification_state"] == "verified"

    asyncio.run(run())
    assert bot.intents.message_content is False
    assert bot.store.summary(123, "2026-11-22")["target"] == 15


def test_dm_forbidden_is_safe(bot, monkeypatch):
    response = SimpleNamespace(status=403, reason="Forbidden")
    user = SimpleNamespace(send=AsyncMock(side_effect=discord.Forbidden(response, "synthetic blocked DM")))
    monkeypatch.setattr(bot, "get_user", lambda user_id: user)
    with pytest.raises(DeliveryError) as error:
        asyncio.run(bot.send_checkin(123, "Synthetic check-in"))
    assert error.value.code == "forbidden" and not error.value.retryable


def test_syncs_global_and_optional_guild(bot, monkeypatch):
    bot.tree.sync = AsyncMock()
    monkeypatch.setattr(bot, "checkin_loop", AsyncMock())

    async def run():
        await bot.setup_hook()
        await bot.close()

    asyncio.run(run())
    bot.tree.sync.assert_awaited_once_with()
