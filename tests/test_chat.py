import asyncio

import pytest

import chat
from config import Config
from memory import Memory

USER = 123
DAY = "2026-11-22"


@pytest.fixture
def local(store, monkeypatch):
    monkeypatch.setattr(chat, "today", lambda: DAY)
    return chat.LocalChat(Config(), store, USER)


def say(local, *lines):
    async def run():
        return [await local.handle(line) for line in lines]
    return asyncio.run(run())


def test_commands_and_plain_messages_share_the_store(local):
    goal, report, summary = say(local, "/setgoal Korean grammar 30", "/report 20", "/summary")
    assert goal == "Goal saved: Korean grammar — 30 min/day"
    assert "Reported 20 minutes" in report
    assert "Reported: 20 / 30 min" in summary
    assert "You reported 20 minutes" in say(local, "how am I doing?")[0]
    quiz = say(local, "quiz me")[0]
    assert "물" in quiz and "Just reply" in quiz
    assert say(local, "water")[0].startswith("Correct!")
    assert local.store.summary(USER, DAY)["verification_state"] == "verified"


def test_recovery_flow_and_history(local):
    say(local, "/setgoal Korean 30", "/report 10")
    proposal = say(local, "/recover late shift")[0]
    number = proposal.split("#")[1].split(":")[0]
    assert say(local, f"/adjust {number} approve")[0] == f"Proposal #{number}: approved."
    assert local.store.summary(USER, DAY)["target"] == 15
    assert "recovery_approved" in say(local, "/history")[0]


def test_errors_are_shown_not_raised(local):
    assert "Create a goal first" in say(local, "/report 20")[0]
    assert "Usage: /setgoal" in say(local, "/setgoal 30")[0]
    assert "whole number" in say(local, "/report lots")[0]
    assert "Unknown command /nope" in say(local, "/nope")[0]
    assert say(local, "   ")[0] is None


def test_local_message_ids_are_unique_and_outside_discord_range(local):
    ids = [local.next_message_id() for _ in range(1000)]
    assert len(set(ids)) == 1000 and ids == sorted(ids)
    assert all(chat.LOCAL_ID_BASE < value < 2**63 for value in ids)


def test_replacing_a_recorded_total_needs_confirmation(local):
    replies = say(local, "/setgoal Korean 30", "today's total is 20 minutes", "today's total is 25 minutes")
    assert "Recorded 20" in replies[1] and "confirm #" in replies[2]
    assert len(Memory(local.store).history(USER, kind="progress_reported")) == 1
    number = replies[2].split("confirm #")[1].split("”")[0]
    assert "Recorded 25" in say(local, f"confirm #{number}")[0]
    assert local.store.summary(USER, DAY)["minutes"] == 25


def test_unchanged_total_needs_no_confirmation(local):
    replies = say(local, "/setgoal Korean 30", "today's total is 20 minutes", "today's total is 20 minutes",
                  "/report 20")
    assert "Recorded 20" in replies[1]
    assert replies[2] == "20 minutes is already recorded as today's total, so nothing changed."
    assert replies[3] == replies[2]
    assert len(Memory(local.store).history(USER, kind="progress_reported")) == 1


def test_unchanged_total_still_applies_the_rest_of_the_message(local):
    say(local, "/setgoal Korean 30", "today's total is 20 minutes")
    reply = say(local, "20 minutes, I'm tired")[0]
    assert "noted the obstacle" in reply and "confirm #" not in reply
    assert len(Memory(local.store).history(USER, kind="obstacle_reported")) == 1
    assert len(Memory(local.store).history(USER, kind="progress_reported")) == 1


def test_quiz_accepts_natural_answers(local):
    say(local, "/setgoal Korean 30", "/report 30", "quiz me")
    assert say(local, "it means water?")[0].startswith("Correct! 물 means water")
    say(local, "quiz me")
    reply = say(local, "I don't know")[0]
    assert reply.startswith("Not quite: 학교 means school")
    assert local.store.summary(USER, DAY)["verification_state"] == "review_required"


def test_known_requests_still_work_while_a_quiz_waits(local):
    say(local, "/setgoal Korean 30", "/report 30", "quiz me")
    assert "You reported 30 minutes" in say(local, "how am I doing?")[0]
    assert "Would you like" in say(local, "hello")[0]
    assert say(local, "water")[0].startswith("Correct!")


def test_old_quiz_is_not_answered_by_a_stray_word(local):
    say(local, "/setgoal Korean 30", "/report 30", "quiz me")
    with local.store.connect() as db:
        db.execute("UPDATE verification_attempts SET created_at='2000-01-01T00:00:00+09:00'")
    assert "Correct" not in say(local, "water")[0]
    assert local.store.summary(USER, DAY)["verification_state"] == "reported"


def test_yes_and_no_handle_drafts(local):
    say(local, "/setgoal Korean 30", "today's total is 20 minutes")
    draft = say(local, "today's total is 25 minutes")[0]
    assert "Reply yes to go ahead" in draft
    assert "Recorded 25" in say(local, "yep")[0]
    say(local, "today's total is 40 minutes")
    assert "discarded" in say(local, "nope")[0]
    assert local.store.summary(USER, DAY)["minutes"] == 25


def test_plan_changes_need_approve_not_yes(local):
    say(local, "/setgoal Korean 30", "/report 10", "finish tomorrow")
    assert "explicit approval" in say(local, "yes")[0]
    assert local.store.summary(USER, DAY)["target"] == 30
    assert "approved" in say(local, "approve")[0]
    assert local.store.summary(USER, DAY)["target"] == 10


def test_no_rejects_a_plan_change_and_nothing_pending_is_explained(local):
    say(local, "/setgoal Korean 30", "/report 10", "finish tomorrow")
    assert "rejected" in say(local, "no")[0]
    assert "no pending draft or proposal" in say(local, "approve")[0]
    assert local.store.summary(USER, DAY)["target"] == 30


def test_requests_are_not_graded_as_quiz_answers(local):
    """Regression: "try quizzing me" was graded as a wrong answer while a quiz was waiting."""
    say(local, "/setgoal Korean 30", "/report 30", "quiz me")
    for request in ("try quizzing me", "another one please", "skip"):
        assert "Not quite" not in (say(local, request)[0] or "")
    assert local.store.summary(USER, DAY)["verification_state"] == "reported"
    assert say(local, "water")[0].startswith("Correct!")


def test_ai_understood_quiz_request_starts_without_confirmation(local):
    from unittest.mock import AsyncMock
    from dm_intents import Interpretation, QUIZ, action
    say(local, "/setgoal Korean 30", "/report 30")
    local.conversation.adapter.interpret = AsyncMock(return_value=Interpretation([action(QUIZ)], "ollama"))
    reply = say(local, "could you check my vocab for me")[0]
    assert "물" in reply and "Did I understand" not in reply
