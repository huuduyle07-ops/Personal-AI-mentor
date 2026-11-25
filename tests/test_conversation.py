import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

import bot
from config import Config
from core import GoalStore
from dm_intents import CLARIFY, OBSTACLE, RECOVER, REPORT, action, interpret, rule_actions
from llm_adapter import OllamaAdapter
from memory import Memory
from time_utils import KST

DAY, TOMORROW, USER = "2026-11-24", "2026-11-25", 123
VI_LIVE = "Hôm nay tôi học tiếng Hàn được 20 phút, nhưng hơi mệt nên muốn chuyển 10 phút còn lại sang ngày mai."
VI_FREE = "Chiều nay tôi ôn tiếng Hàn 20 phút, nhưng hơi mệt nên muốn chuyển 10 phút còn lại sang ngày mai."
MOVE_ACTIONS = [action(REPORT, minutes=20), action(OBSTACLE, reason="tired"),
                action(RECOVER, when="tomorrow", remaining_minutes=10)]


@pytest.fixture
def pm(store, monkeypatch):
    monkeypatch.setattr(bot, "today", lambda: DAY)
    store.set_goal(USER, "Korean", 30, DAY)
    return bot.create_bot(Config(token="synthetic", allowed_user_id=USER), store)


def message(content, number, user=USER, private=True, robot=False):
    return SimpleNamespace(content=content, id=number, author=SimpleNamespace(bot=robot, id=user),
                           guild=None, type=discord.MessageType.default,
                           created_at=datetime.fromisoformat(DAY).replace(hour=12, tzinfo=KST),
                           channel=SimpleNamespace(type=discord.ChannelType.private if private else discord.ChannelType.group,
                                                   send=AsyncMock(return_value=SimpleNamespace(id=10_000 + number))))


async def dm(pm, text, number):
    incoming = message(text, number)
    await pm.on_message(incoming)
    call = incoming.channel.send.call_args
    return call.args[0] if call else None


@pytest.mark.parametrize("text,confirm,approve", [
    ("I studied Korean for 20 minutes today, but I'm tired and want to finish tomorrow.", "confirm", "approve"),
    ("Hôm nay tôi đã học tiếng Hàn 20 phút, nhưng tôi mệt và muốn học tiếp vào ngày mai.", "xác nhận", "đồng ý"),
    ("오늘 한국어를 20분 공부했는데 피곤해서 내일 마저 하고 싶어요.", "확인", "승인"),
])
def test_acceptance_through_discord_and_restart(pm, text, confirm, approve):
    async def run():
        assert pm.intents.dm_messages and not pm.intents.guild_messages and not pm.intents.message_content
        draft = await dm(pm, text, 100)
        expected = {"confirm": ("20 minutes as today's total", "tired", "tomorrow", "remaining 10 minutes", "approved"),
                    "xác nhận": ("20/30 phút", "mệt", "ngày mai", "10 phút còn lại", "Đã duyệt"),
                    "확인": ("30분 중 20분", "피곤함", "내일", "남은 10분", "승인했어요")}[confirm]
        assert all(phrase in draft for phrase in expected[:3])
        assert await dm(pm, text, 100) is None
        assert pm.store.summary(USER, DAY)["minutes"] is None
        restart = bot.create_bot(pm.config, GoalStore(pm.store.path))
        proposal = await dm(restart, f"{confirm} #1", 101)
        assert expected[3] in proposal and "40" in proposal
        before = pm.store.summary(USER, DAY)
        assert (before["minutes"], before["target"], before["verification_state"]) == (20, 30, "reported")
        with pm.store.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM daily_goals WHERE day=?", (TOMORROW,)).fetchone()[0] == 0
        assert expected[4] in await dm(restart, f"{approve} #1", 102)
        assert await dm(pm, f"{approve} #1", 102) is None
        today, tomorrow = pm.store.summary(USER, DAY), pm.store.summary(USER, TOMORROW)
        assert (today["target"], today["original_target"]) == (20, 30)
        assert (tomorrow["target"], tomorrow["original_target"]) == (40, 30)
        assert pm.store.summary(USER, "2026-11-26")["target"] == 30
        assert restart.scheduler.status(USER)["settings"] is None
        history = Memory(pm.store)
        for kind in ("progress_reported", "obstacle_reported", "goal_set", "recovery_approved", "minutes_rescheduled"):
            assert len(history.history(USER, kind=kind)) == 1
        assert "reported 20 minutes" in await dm(restart, "How am I doing?", 103)
        assert not pm.config.ollama_enabled
    asyncio.run(run())


def test_private_gate_precedes_inference_and_receipts(pm):
    pm.conversation.adapter.interpret = AsyncMock()

    async def run():
        messages = [message("private content", 1, user=999), message("private content", 2, robot=True),
                    message("private content", 3, private=False), message("  ", 4), message("guild content", 5)]
        messages[-1].guild = object()
        for incoming in messages:
            await pm.on_message(incoming)
            incoming.channel.send.assert_not_called()
    asyncio.run(run())
    pm.conversation.adapter.interpret.assert_not_called()
    with pm.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM dm_receipts").fetchone()[0] == 0


def test_multiple_clients_and_replayed_events(pm):
    second = bot.create_bot(pm.config, GoalStore(pm.store.path))

    async def run():
        replies = await asyncio.gather(dm(pm, "Today's total is 15 minutes", 1),
                                       dm(second, "Today's total is 15 minutes", 1))
        assert sum(reply is not None for reply in replies) == 1
        assert "currently 15" in await dm(pm, "I studied for 5 minutes today", 2)
        pm.store.report(USER, DAY, 18)
        assert "changed" in await dm(second, "confirm #2", 3)
        assert pm.store.summary(USER, DAY)["minutes"] == 18
        assert "currently 18" in await dm(pm, "Today's total is 7 minutes", 4)
        assert "discarded" in await dm(pm, "discard #4", 5)
        assert "couldn't find" in await dm(second, "confirm #4", 6)
    asyncio.run(run())
    assert len(Memory(pm.store).history(USER, kind="progress_reported")) == 2


def test_controls_stale_moves_and_quiz(pm):
    async def run():
        await dm(pm, "I studied for 12 minutes", 1)
        assert "water" not in (await dm(pm, "quiz me", 2)).lower()
        assert "Would you like" in await dm(pm, "answer #1 water and approve #1", 200)
        assert "verified" in await dm(pm, "answer #1 water", 3)
        assert "Vocabulary quiz: verified" in await dm(pm, "my progress", 4)
        await dm(pm, "finish tomorrow", 5)
        await dm(pm, "yes", 6)
        assert pm.store.summary(USER, DAY)["target"] == 30
        assert "rejected" in await dm(pm, "reject #1", 7)
        await dm(pm, "finish tomorrow", 8)
        pm.store.report(USER, DAY, 13)
        assert "changed" in await dm(pm, "approve #2", 9)
        assert pm.store.summary(USER, TOMORROW)["target"] == 30
        await dm(pm, "finish tomorrow", 10)
        pm.store.set_goal(USER, "Piano", 45, TOMORROW)
        assert "changed" in await dm(pm, "approve #3", 11)
        assert pm.store.summary(USER, TOMORROW)["target"] == 45
    asyncio.run(run())


def test_verified_model_suggestion_requires_human_confirmation(pm):
    private_text = "I spent 22 minutes of my afternoon practicing vocabulary."
    transport = AsyncMock(return_value={"response": json.dumps({"actions": [action(REPORT, minutes=22)]})})
    pm.conversation.adapter = OllamaAdapter(Config(ollama_enabled=True), transport)
    pm.store.set_goal(999, "unrelated private record", 30, DAY)

    async def run():
        assert "22 minutes as today's total" in await dm(pm, private_text, 99)
        assert pm.store.summary(USER, DAY)["minutes"] is None
        assert "22 self-reported" in await dm(pm, "confirm #1", 100)
        assert "No vocabulary result" in await dm(pm, "my progress", 101)
        assert "no saved plan" in await dm(pm, "yesterday's progress", 102)
    asyncio.run(run())
    payload = transport.call_args.args[0]
    assert payload["prompt"] == private_text and "unrelated private record" not in str(payload)
    assert private_text not in str(Memory(pm.store).history(USER))
    assert transport.await_count == 1


@pytest.mark.parametrize("response", [
    "[]", "null", "bad json", '{"actions":[]}',
    '{"actions":[{"intent":"APPROVE_ADJUSTMENT","id":1}]}',
    '{"actions":[{"intent":"RUN_CODE","code":"ignored"}]}',
    '{"actions":[{"intent":"REPORT_PROGRESS","minutes":true}]}',
    '{"actions":[{"intent":"REPORT_PROGRESS","minutes":8,"verified":true}]}',
    '{"actions":[{"intent":"REQUEST_RECOVERY","when":"everyday"}]}',
    '{"actions":[{"intent":"REPORT_PROGRESS","minutes":8},{"intent":"REPORT_PROGRESS","minutes":9}]}',
])
def test_invalid_model_proposals_are_never_dispatched(pm, response):
    pm.conversation.adapter = OllamaAdapter(Config(ollama_enabled=True), AsyncMock(return_value={"response": response}))
    assert "Would you like" in asyncio.run(dm(pm, "Something unusual", 1))
    assert pm.store.summary(USER, DAY)["minutes"] is None
    assert [e["kind"] for e in Memory(pm.store).history(USER)] == ["dm_interpretation", "goal_set"]


def test_timeout_and_partial_grammar():
    async def slow(payload):
        await asyncio.sleep(10)
    adapter = OllamaAdapter(Config(ollama_enabled=True, ollama_timeout=0.1), slow)
    result = asyncio.run(interpret("I studied for 20 minutes, but don't record it.", adapter))
    assert result.source == "timeout" and result.actions == [action(CLARIFY)]
    assert asyncio.run(interpret(" " * 2000 + "hello", adapter)).source == "input_limit"


@pytest.mark.parametrize("text", ["I recall practicing.", "Perhaps I practiced 12 or 23 minutes.",
                                  "I reviewed 19 words. How many minutes would that be?", "I practiced for hours."])
def test_model_cannot_invent_numeric_report(pm, text):
    output = {"response": json.dumps({"actions": [action(REPORT, minutes=19)]})}
    pm.conversation.adapter = OllamaAdapter(Config(ollama_enabled=True), AsyncMock(return_value=output))
    assert "Would you like" in asyncio.run(dm(pm, text, 1))
    with pm.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM dm_drafts").fetchone()[0] == 0


def test_bare_duration_and_stale_day(pm):
    async def run():
        assert "today's total" in await dm(pm, "20 minutes", 1)
        pm.conversation.today = lambda: TOMORROW
        assert "KST day changed" in await dm(pm, "confirm #1", 2)
        assert "couldn't find" in await pm.conversation.receive(USER, 3, "confirm #1", TOMORROW)
        assert pm.store.summary(USER, DAY)["minutes"] is None
    asyncio.run(run())


def test_slash_adjust_can_resolve_dm_move(pm):
    async def run():
        await dm(pm, "today's total is 20 minutes", 10)
        await dm(pm, "finish tomorrow", 11)
        request = SimpleNamespace(user=SimpleNamespace(id=USER), response=SimpleNamespace(send_message=AsyncMock()))
        await pm.tree.get_command("adjust").callback(request, 1, "approve")
        assert "approved" in request.response.send_message.call_args.args[0]
        assert pm.store.summary(USER, TOMORROW)["target"] == 40
    asyncio.run(run())


def test_atomic_failure_and_retry(pm, monkeypatch):
    report = pm.store.report

    def interrupted(*args, **kwargs):
        report(*args, **kwargs)
        raise RuntimeError("synthetic failure after report")
    monkeypatch.setattr(pm.store, "report", interrupted)
    assert "couldn't process" in asyncio.run(dm(pm, "Today's total is 22 minutes", 1))
    assert pm.store.summary(USER, DAY)["minutes"] is None
    monkeypatch.setattr(pm.store, "report", report)
    assert "22 self-reported" in asyncio.run(dm(pm, "Today's total is 22 minutes", 1))


def test_checkin_reply_and_paused_schedule(pm):
    async def run():
        start = datetime.fromisoformat(DAY).replace(hour=11, tzinfo=KST)
        pm.scheduler.send = AsyncMock()
        pm.scheduler.configure(USER, now=start)
        await pm.scheduler.tick(start.replace(hour=12))
        assert "Tell me your total" in pm.scheduler.send.call_args.args[1]
        assert "10 self-reported" in await dm(pm, "10 minutes total", 1)
        await pm.scheduler.tick(start.replace(hour=12, minute=1))
        pm.scheduler.pause(USER, True, now=start.replace(hour=13))
        await pm.scheduler.tick(start.replace(hour=21))
        assert pm.scheduler.send.await_count == 1
    asyncio.run(run())


def saved_draft(pm):
    with pm.store.connect() as db:
        row = db.execute("SELECT actions FROM dm_drafts WHERE user_id=?", (USER,)).fetchone()
        return json.loads(row["actions"]) if row else None


@pytest.mark.parametrize("text", [
    VI_LIVE,
    "I studied Korean for 20 minutes today, but I'm tired and want to move the remaining 10 minutes to tomorrow.",
    "오늘 한국어를 20분 공부했는데 피곤해서 남은 10분을 내일로 미루고 싶어요.",
])
def test_two_durations_are_separate_slots(pm, text):
    transport = AsyncMock(side_effect=OSError("private backend details"))
    pm.conversation.adapter = OllamaAdapter(Config(ollama_enabled=True), transport)
    assert rule_actions(text) == MOVE_ACTIONS
    reply = asyncio.run(dm(pm, text, 200))
    assert TOMORROW in reply and saved_draft(pm) == MOVE_ACTIONS
    assert pm.store.summary(USER, DAY)["minutes"] is None
    assert pm.store.summary(USER, DAY)["recoveries"] == []
    transport.assert_not_called()


def test_production_dm_followups_and_replays(pm, caplog):
    caplog.set_level("INFO")

    async def run():
        assert "chưa có bản nháp" in await dm(pm, "được chứ", 100)
        preview = await dm(pm, VI_LIVE, 101)
        assert all(part in preview for part in ("20/30 phút", "mệt", "10 phút còn lại", TOMORROW, "xác nhận #2"))
        assert await dm(pm, VI_LIVE, 101) is None
        assert await dm(pm, VI_LIVE, 102) == preview
        assert "xác nhận #2" in await dm(pm, "được chứ", 103)
        assert pm.store.summary(USER, DAY)["minutes"] is None
        proposal = await dm(pm, "xác nhận #2", 104)
        assert all(part in proposal for part in ("Đã ghi 20 phút", "10 phút còn lại", TOMORROW, "40 phút", "đồng ý #1"))
        assert await dm(pm, "xác nhận #2", 104) is None
        assert "Không tìm thấy" in await dm(pm, "xác nhận #2", 105)
        assert "đồng ý #1" in await dm(pm, "được chứ", 106)
        assert pm.store.summary(USER, DAY)["target"] == 30
        assert pm.store.summary(USER, DAY)["verification_state"] == "reported"
        assert "Đã duyệt" in await dm(pm, "đồng ý #1", 107)
        assert await dm(pm, "đồng ý #1", 107) is None
        assert "Không tìm thấy" in await dm(pm, "đồng ý #1", 108)
        assert "chưa có bản nháp" in await dm(pm, "được chứ", 109)
        assert pm.store.summary(USER, DAY)["target"] == 20
        assert pm.store.summary(USER, TOMORROW)["target"] == 40
        assert pm.store.summary(USER, DAY)["original_target"] == 30
        for kind in ("progress_reported", "obstacle_reported", "recovery_proposed", "recovery_approved", "minutes_rescheduled"):
            assert len(Memory(pm.store).history(USER, kind=kind)) == 1
    asyncio.run(run())
    assert "stage=rules outcome=matched intents=REPORT_PROGRESS,REPORT_OBSTACLE,REQUEST_RECOVERY" in caplog.text
    assert VI_LIVE not in caplog.text


@pytest.mark.parametrize("backend,reason", [
    ("valid", None), ("offline", "backend_unavailable"), ("bad_json", "invalid_json"), ("missing_slot", "invalid_schema"),
    ("drop_report", "invalid_evidence"), ("drop_obstacle", "invalid_evidence"), ("drop_amount", "invalid_evidence"),
    ("wrong_amount", "invalid_evidence"), ("disabled", "disabled"), ("timeout", "timeout"),
])
def test_model_route_validates_entire_multi_intent_dm(pm, backend, reason, caplog):
    caplog.set_level("INFO")
    outputs = {"valid": MOVE_ACTIONS, "drop_report": MOVE_ACTIONS[1:],
               "drop_obstacle": MOVE_ACTIONS[::2], "drop_amount": [*MOVE_ACTIONS[:2], action(RECOVER, when="tomorrow")],
               "missing_slot": [action(REPORT), *MOVE_ACTIONS[1:]],
               "wrong_amount": [*MOVE_ACTIONS[:2], action(RECOVER, when="tomorrow", remaining_minutes=20)]}
    transport = AsyncMock(return_value={"response": json.dumps({"actions": outputs.get(backend)})})
    if backend == "offline":
        transport.side_effect = OSError("synthetic-private-error")
    elif backend == "timeout":
        transport.side_effect = TimeoutError()
    elif backend == "bad_json":
        transport.return_value = {"response": "synthetic-private-malformed-output"}
    pm.conversation.adapter = OllamaAdapter(Config(ollama_enabled=backend != "disabled"), transport)
    assert rule_actions(VI_FREE) is None
    reply = asyncio.run(dm(pm, VI_FREE, 100))
    assert transport.await_count == int(backend != "disabled")
    assert "stage=rules outcome=miss" in caplog.text
    assert VI_FREE not in caplog.text and "synthetic-private" not in caplog.text
    assert pm.store.summary(USER, DAY)["target"] == 30
    assert pm.store.summary(USER, DAY)["minutes"] is None
    assert pm.store.summary(USER, DAY)["recoveries"] == []
    if backend == "valid":
        assert saved_draft(pm) == MOVE_ACTIONS
        assert "20/30 phút" in reply and "10 phút còn lại" in reply
        assert "đồng ý #1" in asyncio.run(dm(pm, "xác nhận #1", 101))
        assert pm.store.summary(USER, DAY)["target"] == 30
    else:
        assert saved_draft(pm) is None
        assert "học tổng cộng bao nhiêu phút" in reply and "chuyển 10 phút sang ngày mai" in reply
        assert "reason=" + reason in caplog.text


def test_conflicting_remainder_requires_clarification(pm):
    reply = asyncio.run(dm(pm, VI_LIVE.replace("10 phút còn lại", "7 phút còn lại"), 100))
    assert "còn 10 phút" in reply and "xác nhận lại" in reply
    assert saved_draft(pm) is None
    assert pm.store.summary(USER, DAY)["minutes"] is None
    assert pm.store.summary(USER, DAY)["recoveries"] == []


@pytest.mark.parametrize("condition", ["valid", "stale", "unrelated", "question", "superseded"])
def test_vietnamese_assent_bound_to_delivered_discord_prompt(pm, condition):
    async def run():
        await dm(pm, VI_LIVE, 100)
        await dm(pm, "xác nhận #1", 101)
        if condition == "stale":
            pm.store.report(USER, DAY, 12)
        if condition == "superseded":
            await dm(pm, "finish tomorrow", 102)
        restart = bot.create_bot(pm.config, GoalStore(pm.store.path))
        incoming = message("được chứ?" if condition == "question" else "được chứ", 200)
        incoming.type = discord.MessageType.reply
        incoming.reference = SimpleNamespace(message_id=9999 if condition == "unrelated" else 10101)
        await restart.on_message(incoming)
        await restart.on_message(incoming)
        assert incoming.channel.send.await_count == 1
        approved = condition == "valid"
        assert ("Đã duyệt" in incoming.channel.send.call_args.args[0]) == approved
        assert pm.store.summary(USER, DAY)["target"] == (20 if approved else 30)
        assert pm.store.summary(USER, TOMORROW)["target"] == (40 if approved else 30)
        assert len(Memory(pm.store).history(USER, kind="recovery_approved")) == int(approved)
    asyncio.run(run())


def test_discord_utc_timestamp_and_kst_midnight(pm):
    async def deliver(text, number, timestamp):
        incoming = message(text, number)
        incoming.created_at = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
        await pm.on_message(incoming)
        return incoming.channel.send.call_args.args[0]

    async def run():
        assert TOMORROW in await deliver(VI_LIVE, 100, DAY + "T14:59:58")
        assert "đồng ý #1" in await deliver("xác nhận #1", 101, DAY + "T14:59:59")
        pm.conversation.today = lambda: TOMORROW
        assert "ngày mới" in await deliver("đồng ý #1", 102, DAY + "T14:59:59")
        assert "Không tìm thấy" in await deliver("đồng ý #1", 103, DAY + "T15:00:00")
        assert pm.store.summary(USER, DAY)["target"] == 30
        assert "2026-11-26" in await deliver(VI_LIVE, 104, DAY + "T15:00:01")
        assert pm.store.summary(USER, TOMORROW)["minutes"] is None
    asyncio.run(run())


def test_kst_day_changes_while_ollama_is_running(pm):
    async def cross_midnight(payload):
        pm.conversation.today = lambda: TOMORROW
        return {"response": json.dumps({"actions": MOVE_ACTIONS})}
    pm.conversation.adapter = OllamaAdapter(Config(ollama_enabled=True), cross_midnight)
    reply = asyncio.run(dm(pm, VI_FREE, 100))
    assert "ngày mới" in reply and saved_draft(pm) is None
    for day in (DAY, TOMORROW):
        data = pm.store.summary(USER, day)
        assert data["minutes"] is None and data["target"] == 30 and data["recoveries"] == []
