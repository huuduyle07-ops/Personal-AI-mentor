import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
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
    ("recover", ("Work shift",)), ("adjust", (1, "approve")), ("history", ()), ("recall", ()), ("policy", ()), ("focus", ()),
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


def test_setup_hook_skips_unchanged_commands_and_clears_guild_copies(store):
    config = Config(token="synthetic-token", allowed_user_id=123, guild_id=456)

    async def start(bot):
        bot.tree.sync = AsyncMock()
        bot.checkin_loop = AsyncMock()
        await bot.setup_hook()
        await bot.close()
        return bot.tree.sync

    first = asyncio.run(start(integration.create_bot(config, store)))
    assert first.await_count == 2 and first.await_args_list[1].kwargs["guild"].id == 456
    restart = asyncio.run(start(integration.create_bot(config, GoalStore(store.path))))
    restart.assert_not_awaited()


def test_discord_text_fits_limit_without_dangling_escape():
    assert integration.discord_text("short *bold*") == "short \\*bold\\*"
    for text in ("x" * 2500, "*" * 1500, "a" + "*" * 1500):
        capped = integration.discord_text(text)
        assert len(capped) <= 2000 and capped.endswith("…")
        body = capped[:-1]
        assert (len(body) - len(body.rstrip("\\"))) % 2 == 0


def http_error(kind, status):
    return kind(SimpleNamespace(status=status, reason="synthetic"), "synthetic")


@pytest.mark.parametrize("error,code,retryable,uncertain", [
    (http_error(discord.HTTPException, 400), "delivery_failed", False, False),
    (http_error(discord.DiscordServerError, 503), "delivery_failed", False, True),
    (aiohttp.ClientConnectorError(SimpleNamespace(host="discord.com", port=443, ssl=True),
                                  OSError(61, "refused")), "unavailable", True, False),
])
def test_checkin_delivery_error_mapping(bot, monkeypatch, error, code, retryable, uncertain):
    user = SimpleNamespace(send=AsyncMock(side_effect=error))
    monkeypatch.setattr(bot, "get_user", lambda user_id: user)
    with pytest.raises(DeliveryError) as raised:
        asyncio.run(bot.send_checkin(123, "Synthetic check-in"))
    assert (raised.value.code, raised.value.retryable, raised.value.uncertain) == (code, retryable, uncertain)


def test_long_checkin_is_capped(bot, monkeypatch):
    user = SimpleNamespace(send=AsyncMock())
    monkeypatch.setattr(bot, "get_user", lambda user_id: user)
    asyncio.run(bot.send_checkin(123, "_" * 3000))
    assert len(user.send.call_args.args[0]) <= 2000


def test_report_command_says_when_total_is_unchanged(bot):
    async def invoke(name, *args):
        request = interaction()
        await bot.tree.get_command(name).callback(request, *args)
        return request.response.send_message.call_args.args[0]

    async def run():
        await invoke("setgoal", "Korean", 30)
        assert "Reported 20" in await invoke("report", 20)
        assert "already recorded" in await invoke("report", 20)

    asyncio.run(run())
    assert len(Memory(bot.store).history(123, kind="progress_reported")) == 1
