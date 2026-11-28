"""Authorized DM delivery and atomic dispatch. The interpreter has no store handle."""
import asyncio
import json
import logging
import re
from collections import deque
from datetime import date, datetime, timedelta

import active_quiz
import chat_context
import focus

from dm_intents import (
    APPROVE, BARE_DURATION, CHAT, CLARIFY, OBSTACLE, QUERY, QUIZ, RECOVER, REJECT, REPORT,
    Interpretation, action, interpret, normalized, quiz_answer, rule_actions, validate,
)
from memory import record
from dm_replies import (
    AdjustmentPrompt, clarification, domain_message, draft_reply, language, obstacle, proposal_reply, say, unchanged_total,
)
import study_notes
from time_utils import now_kst, valid_day
from verification import user_deck

log = logging.getLogger(__name__)
QUIZ_REPLY_WINDOW = timedelta(hours=2)
CHAT_TURNS = 6  # last 6 messages (3 exchanges) are sent with each chit-chat reply


# Asking for a sentence quiz is detected after classification, so the intent prompt and rules are unchanged.
SENTENCE_REQUEST = re.compile(r"\b(?:sentences?|context|contextual|example)\b|문장|câu|ngữ cảnh", re.I)


# "What should I focus on today?" is answered from the read-only focus summary when the message was
# classified as a question, small talk or unclear (never for messages that record something).
FOCUS_REQUEST = re.compile(r"\bfocus\b|what should i (?:do|study|work on)\b|where should i start\b|"
                           r"집중|오늘 뭐(?:부터)? (?:할까|하지|해야)|nên (?:làm|học) gì|tập trung", re.I)


# Active production practice ("quiz me in Korean", "spelling quiz"): also detected after classification.
ACTIVE_REQUEST = re.compile(
    r"\b(?:spelling|active|production|writing|hangul)\s+(?:quiz|test|practice)\b|"
    r"\b(?:quiz|test)\s+me\s+in\s+(?:korean|hangul)\b|\b(?:type|write|spell)\b.*\bin\s+(?:korean|hangul)\b|"
    r"한글로|받아쓰기|철자|chính tả|viết (?:bằng )?(?:tiếng hàn|chữ hàn)", re.I)
# A reply to an active quiz: one or two Hangul words, optional trailing punctuation.
HANGUL_ANSWER = re.compile(r"[\u3131-\u318e\uac00-\ud7a3]+(?:\s+[\u3131-\u318e\uac00-\ud7a3]+)?[.!?~。]*")


def active_request(result, text):
    return ({item["intent"] for item in result.actions} <= {QUIZ, QUERY, CHAT, CLARIFY}
            and bool(ACTIVE_REQUEST.search(normalized(text))))


def active_answer(text, passive_pending):
    """The answer in a reply to a waiting active quiz, or None if the message is something else."""
    text = text.strip()
    if HANGUL_ANSWER.fullmatch(text):
        return text
    short = quiz_answer(text)
    if short == "(don't know)":
        return active_quiz.DONT_KNOW
    # A short romanized attempt ("hakgyo") counts as an answer only when no flashcard is waiting for English.
    return short if short and not passive_pending else None


def focus_request(result, text):
    return ({item["intent"] for item in result.actions} <= {QUERY, CHAT, CLARIFY}
            and bool(FOCUS_REQUEST.search(normalized(text))))


def sentence_quiz_request(result, text):
    return {item["intent"] for item in result.actions} == {QUIZ} and bool(SENTENCE_REQUEST.search(normalized(text)))


def chit_chat(result, text):
    """Small talk or an unclear message with no numbers: nothing to record, so the model may just talk.
    Messages with numbers keep the clarification template, which asks for exact minutes."""
    return (result.source != "input_limit" and {item["intent"] for item in result.actions} <= {CHAT, CLARIFY}
            and not re.search(r"[0-9]", text))


class Conversation:
    def __init__(self, store, allowed_user_id, adapter, recovery, verification, today, importer=None):
        self.store, self.allowed_user_id = store, allowed_user_id
        self.adapter, self.recovery, self.verification = adapter, recovery, verification
        self.importer = importer
        self.active = active_quiz.ActiveQuiz(store, verification)
        self.today = today
        self.lock = asyncio.Lock()
        # Recent turns for natural chit-chat replies. Kept in memory only, so chatting writes nothing.
        self.history = deque(maxlen=CHAT_TURNS)
        # Translations of generated quiz sentences, shown after the answer. Memory only: never stored.
        self.sentence_feedback = {}

    async def receive(self, user, message_id, text, sent_day, *, reply_to=None):
        if not self.allowed_user_id or user != self.allowed_user_id:
            return None
        if type(message_id) is not int or not 0 < message_id < 2**63:
            raise ValueError("Invalid Discord event ID")
        valid_day(sent_day)
        lang = language(text)
        async with self.lock:
            # SQLite work runs in a worker thread so a locked database never stalls the event loop.
            if await asyncio.to_thread(self._seen, message_id):
                return None
            result = None
            if sent_day == self.today():
                # While a quiz is waiting, a short reply that isn't a known request is the answer.
                pending = await asyncio.to_thread(self._pending_quiz, user, sent_day)
                active = await asyncio.to_thread(self.active.pending, user, sent_day, window=QUIZ_REPLY_WINDOW)
                known = rule_actions(text) is not None
                typed = active_answer(text, pending) if active and not known else None
                answer = quiz_answer(text) if pending and not known else None
                if typed:
                    result = Interpretation([action("ANSWER_ACTIVE", id=active, answer=typed)], "rule")
                elif answer:
                    result = Interpretation([action("ANSWER_QUIZ", id=pending, answer=answer)], "rule")
                else:
                    result = await interpret(text, self.adapter)
                log.warning(
                    "DM interpretation: source=%s, intents=%s, latency_ms=%s",
                    result.source,
                    [action.get("intent") for action in result.actions],
                    result.latency_ms,
                )
            if result is not None and focus_request(result, text):
                if not await asyncio.to_thread(self._accept, user, message_id, sent_day, result):
                    return None
                snapshot = await asyncio.to_thread(focus.snapshot, self.store, self.verification, user)
                reply = focus.format_focus(snapshot)
            elif result is not None and active_request(result, text):
                if not await asyncio.to_thread(self._accept, user, message_id, sent_day, result):
                    return None
                reply = (await self.active_quiz_reply(user, sent_day, lang))[1]
            elif result is not None and sentence_quiz_request(result, text):
                if not await asyncio.to_thread(self._accept, user, message_id, sent_day, result):
                    return None
                reply = (await self.quiz_reply(user, sent_day, lang, sentence=True))[1]
            elif result is not None and chit_chat(result, text):
                if not await asyncio.to_thread(self._accept, user, message_id, sent_day, result):
                    return None
                reply = await self._converse(user, text, sent_day, lang)
            else:
                reply = await asyncio.to_thread(self._commit, user, message_id, text, sent_day, lang, result, reply_to)
            if reply is not None:
                self.history.extend([("user", text), ("assistant", getattr(reply, "text", reply))])
            return reply

    async def quiz_reply(self, user, day, lang, *, sentence=False):
        """(task or None, reply text). With sentence=True the scheduler's next word is previewed read-only,
        the model writes a practice sentence for it, and the quiz is then recorded exactly like /quiz.
        Any generation or validation failure falls back to the plain flashcard."""
        request = None
        try:
            if sentence:
                upcoming = await asyncio.to_thread(self.verification.next_up, user, day)
                if upcoming is not None:  # a quiz already waiting is simply shown again
                    request = await self.adapter.sentence_quiz(upcoming["word"], upcoming["answers"])
                    request["question_id"] = upcoming["question_id"]
            task = await asyncio.to_thread(self.verification.start, user, day, sentence=request)
        except ValueError as error:
            return None, domain_message(lang, error)
        if task["style"] == "sentence":
            self.sentence_feedback[task["id"]] = task["translation"]
            word = task["word"]
            return task, say(lang, f"{task['sentence']}\nWhat does {word} mean in this sentence? Just reply with the English meaning. "
                                   "(AI-written example; it may occasionally sound unnatural.)",
                             f"{task['sentence']}\nTừ {word} trong câu này nghĩa là gì? Trả lời bằng tiếng Anh nhé. "
                             "(Câu ví dụ do AI viết, đôi khi có thể chưa tự nhiên.)",
                             f"{task['sentence']}\n이 문장에서 {word}은(는) 무슨 뜻일까요? 영어 뜻만 답해 주세요. "
                             "(AI가 만든 예문이라 가끔 어색할 수 있어요.)")
        note = ""
        if sentence and task["policy"] != "pending":
            note = say(lang, "I couldn't write a good sentence this time, so here's the word on its own. ",
                       "Lần này mình chưa viết được câu ví dụ tốt, nên hỏi từ đơn nhé. ",
                       "이번에는 좋은 예문을 만들지 못해서 단어만 물어볼게요. ")
        return task, note + task["prompt"] + " " + say(lang, "Just reply with the English meaning.",
                                                       "Chỉ cần trả lời nghĩa tiếng Anh.", "영어 뜻만 답해 주세요.")

    async def active_quiz_reply(self, user, day, lang):
        """(task or None, reply text) for an active production quiz: English meaning -> typed Hangul."""
        try:
            task = await asyncio.to_thread(self.active.start, user, day)
        except ValueError as error:
            return None, domain_message(lang, error)
        meaning = task["meaning"]
        return task, say(lang, task["prompt"] + " (Spaces and a final period are fine.)",
                         f"“{meaning}” trong tiếng Hàn viết thế nào? Hãy gõ bằng chữ Hàn.",
                         f"“{meaning}”을(를) 한국어로 어떻게 쓸까요? 한글로 입력해 주세요.")

    def _accept(self, user, message_id, sent_day, result):
        """Replay protection and the interpretation audit, exactly as for any message; nothing else."""
        with self.store.transaction() as db:
            if not db.execute("INSERT OR IGNORE INTO dm_receipts(message_id,user_id) VALUES(?,?)",
                              (message_id, user)).rowcount:
                return False
            if result.source not in {"rule", "input_limit", "disabled"}:
                record(db, user, sent_day, "dm_interpretation",
                       {"outcome": result.source, "reason": result.reason, "latency_ms": result.latency_ms},
                       "model_interpretation" if result.source == "ollama" else "system")
            return True

    async def _converse(self, user, text, day, lang):
        """A natural reply grounded in read-only status; the old clarification template if the model can't help."""
        facts = await asyncio.to_thread(chat_context.snapshot, self.store, self.verification, user, day)
        reply = await self.adapter.converse(list(self.history), text, chat_context.describe(facts), lang)
        log.info("DM stage=dispatch outcome=%s", "chat" if reply else "clarify")
        return reply or clarification(lang, text)

    async def import_notes(self, user, message_id, filename, data, sent_day):
        """Extract vocabulary from an uploaded file or photo and ask the user to confirm it."""
        if not self.allowed_user_id or user != self.allowed_user_id or self.importer is None:
            return None
        if type(message_id) is not int or not 0 < message_id < 2**63:
            raise ValueError("Invalid Discord event ID")
        valid_day(sent_day)
        async with self.lock:
            if await asyncio.to_thread(self._seen, message_id):
                return None
            try:
                pairs, source = await self.importer.extract(filename, data)
            except (ValueError, study_notes.VisionUnavailable) as error:
                return str(error)
            return await asyncio.to_thread(self._propose_import, user, message_id, pairs, source)

    def _propose_import(self, user, message_id, pairs, source):
        with self.store.transaction() as db:
            if not db.execute("INSERT OR IGNORE INTO dm_receipts(message_id,user_id) VALUES(?,?)",
                              (message_id, user)).rowcount:
                return None
            known = [question["word"] for question in user_deck(db, user).values()]
            log.info("Notes import stage=extract pairs=%s", len(pairs))
            return study_notes.propose(db, user, pairs, source, known)

    def _seen(self, message_id):
        with self.store.connect() as db:
            return db.execute("SELECT 1 FROM dm_receipts WHERE message_id=?", (message_id,)).fetchone() is not None

    def _commit(self, user, message_id, text, sent_day, lang, result, reply_to):
        with self.store.transaction() as db:
            receipt = db.execute("INSERT OR IGNORE INTO dm_receipts(message_id,user_id) VALUES(?,?)",
                                 (message_id, user))
            if not receipt.rowcount:
                return None
            if result is None or sent_day != self.today():
                log.info("DM stage=dispatch outcome=expired_day")
                return say(lang, "The KST day changed. Please send your request again for today.",
                           "Đã sang ngày mới theo giờ Hàn Quốc. Bạn hãy gửi lại yêu cầu cho hôm nay.",
                           "한국 시간으로 날짜가 바뀌었어요. 오늘 요청을 다시 보내 주세요.")
            if result.source not in {"rule", "input_limit", "disabled"}:
                record(db, user, sent_day, "dm_interpretation",
                       {"outcome": result.source, "reason": result.reason, "latency_ms": result.latency_ms},
                       "model_interpretation" if result.source == "ollama" else "system")
            db.execute("SAVEPOINT dm_action")
            try:
                return self._dispatch(db, user, sent_day, receipt.lastrowid, result, text, lang, reply_to)
            except ValueError as error:
                db.execute("ROLLBACK TO dm_action")
                log.info("DM stage=dispatch outcome=domain_rejected")
                return domain_message(lang, error)

    def _dispatch(self, db, user, day, receipt, result, text, lang, reply_to):
        actions = result.actions
        first = actions[0]
        kind = first["intent"]
        if result.source == "rule" and len(actions) == 1:
            # A word list extracted from uploaded notes is answered before drafts and plan changes.
            if kind == "DROP_IMPORT_ITEMS":
                return study_notes.drop(db, user, first["items"])
            if kind in {"PENDING_DECISION", "PENDING_NO", "CONFIRM_DRAFT", "DISCARD_DRAFT"} \
                    and first.get("id") is None and study_notes.pending(db, user):
                if kind in {"PENDING_DECISION", "CONFIRM_DRAFT"}:
                    return study_notes.confirm(db, user, day)
                return study_notes.cancel(db, user)
            if kind == "PENDING_DECISION":
                if reply_to is not None and normalized(text).rstrip(".! ") == "được chứ":
                    prompt = db.execute("SELECT json_extract(payload,'$.proposal_id') AS proposal_id FROM events "
                                        "WHERE user_id=? AND day=? AND kind='dm_approval_prompt' AND source='system' "
                                        "AND json_extract(payload,'$.message_id')=? LIMIT 1", (user, day, reply_to)).fetchone()
                    if prompt:
                        self.recovery.resolve(user, day, prompt["proposal_id"], "approve", db=db)
                        log.info("DM stage=dispatch outcome=approved_from_referenced_prompt")
                        return f"Đã duyệt đề xuất #{prompt['proposal_id']}."
            if kind in {"PENDING_DECISION", "PENDING_NO"} or first.get("id", 0) is None:
                first = self._resolve_reply(db, user, day, kind)
                if first is None:
                    return self._pending_prompt(db, user, day, lang)
                kind = first["intent"]
            if kind in {"CONFIRM_DRAFT", "DISCARD_DRAFT"}:
                return self._confirm(db, user, day, first, lang)
            if kind in {APPROVE, REJECT}:
                decision = "approve" if kind == APPROVE else "reject"
                status = self.recovery.resolve(user, day, first["id"], decision, db=db)
                number = first["id"]
                return say(lang, f"Proposal #{number} {status}. Say “how am I doing?” to review your plan.",
                           f"Đã {'duyệt' if decision == 'approve' else 'từ chối'} đề xuất #{number}.",
                           f"제안 #{number}을 {'승인' if decision == 'approve' else '거절'}했어요.")
            if kind == "ANSWER_ACTIVE":
                return active_quiz.feedback(self.active.answer(user, day, first["id"], first["answer"], db=db))
            if kind == "ANSWER_QUIZ":
                answer = self.verification.answer(user, day, first["id"], first["answer"], db=db)
                word, meaning = answer["word"], answer["meaning"]
                translation = self.sentence_feedback.pop(first["id"], None)
                context = "" if translation is None else say(
                    lang, f" The sentence means: “{translation}”", f" Câu này nghĩa là: “{translation}”",
                    f" 문장 뜻: “{translation}”")
                if answer["correct"]:
                    return say(lang, f"Correct! {word} means {meaning}.{context} Say “quiz me” for another word.",
                               f"Đúng rồi! {word} nghĩa là {meaning}.{context} Nói “kiểm tra từ vựng” để học từ khác.",
                               f"정답이에요! {word}은(는) {meaning}이에요.{context} 다른 단어는 “퀴즈 내줘”라고 해 주세요.")
                return say(lang, f"Not quite: {word} means {meaning}.{context} Say “quiz me” to try it again.",
                           f"Chưa đúng: {word} nghĩa là {meaning}.{context} Nói “kiểm tra từ vựng” để thử lại.",
                           f"아쉬워요. {word}은(는) {meaning}이에요.{context} “퀴즈 내줘”라고 하면 다시 풀 수 있어요.")

        validate({"actions": actions})
        kinds = {item["intent"] for item in actions}
        if kinds == {QUERY}:
            return self._progress(db, user, day, first["day"], lang)
        if kinds == {QUIZ}:
            # Starting a quiz records nothing about progress or plans, so it needs no confirmation.
            return self._apply(db, user, day, actions, lang)
        if kinds == {CHAT}:
            return clarification(lang, text)
        if CLARIFY in kinds or CHAT in kinds:
            log.info("DM stage=dispatch outcome=clarify source=%s reason=%s", result.source, result.reason or kind)
            return clarification(lang, text)
        plan = self.store.plan(db, user, day)
        prior = self._latest(db, user, day)
        if prior is not None and any(item["intent"] == REPORT and item["minutes"] == prior["minutes"] for item in actions):
            # Repeating the recorded total changes nothing, so it needs no draft or confirmation.
            actions = [item for item in actions if item["intent"] != REPORT]
            if not actions:
                log.info("DM stage=dispatch outcome=unchanged_total")
                return unchanged_total(lang, prior["minutes"])
            kinds = {item["intent"] for item in actions}
        by_kind = {item["intent"]: item for item in actions}
        move = by_kind.get(RECOVER, {})
        remaining = None
        if move.get("when") == "tomorrow":
            report = by_kind.get(REPORT, prior)
            total = report["minutes"] if report is not None else None
            if total is not None:
                remaining = self.recovery.remaining_minutes(plan["target"], total)
            if "remaining_minutes" in move and remaining != move["remaining_minutes"]:
                log.info("DM stage=dispatch outcome=clarify reason=remaining_mismatch")
                if remaining is None:
                    return clarification(lang, text)
                return say(lang, f"With {total} minutes reported, {remaining} would remain. How many minutes did you study today, and how many should move?",
                           f"Nếu đã học {total} phút thì còn {remaining} phút. Bạn xác nhận lại số phút đã học và số phút muốn chuyển nhé?",
                           f"{total}분 공부했다면 {remaining}분이 남아요. 공부한 시간과 내일로 미룰 시간을 다시 알려 주세요.")
        # A named subject is safe to log directly only for the matching goal.
        words = normalized(text).strip(".! ")
        named_korean = any(word in words for word in ("korean", "tiếng hàn", "한국어"))
        other_goal = named_korean and normalized(plan["title"]) not in {"korean", "tiếng hàn", "한국어"}
        if (len(actions) > 1 or result.source == "ollama"
                or (REPORT in kinds and (prior or other_goal or re.fullmatch(BARE_DURATION, words)))):
            draft = json.dumps(actions)
            report_id = prior["id"] if prior else None
            previous = db.execute("SELECT receipt_id FROM dm_drafts WHERE user_id=? AND day=? AND actions=? "
                                  "AND report_id IS ? AND revision=?", (user, day, draft, report_id, plan["revision"])).fetchone()
            if previous:
                receipt = previous["receipt_id"]
            else:
                db.execute("INSERT OR REPLACE INTO dm_drafts VALUES(?,?,?,?,?,?)",
                           (user, day, receipt, draft, report_id, plan["revision"]))
            log.info("DM stage=dispatch outcome=draft")
            return draft_reply(lang, actions, plan, prior, remaining, day, receipt)
        return self._apply(db, user, day, actions, lang)

    def _resolve_reply(self, db, user, day, kind):
        """Turn a reply without a number into a decision on today's single pending item, or None.

        yes      -> save the draft (never approves a plan change)
        no       -> discard the draft, else reject the plan change (saying no is always safe)
        approve  -> approve the plan change, else save the draft
        reject   -> reject the plan change, else discard the draft
        confirm / discard -> the draft only
        """
        draft, proposal = self._pending_draft(db, user, day), self._pending_proposal(db, user, day)
        save, drop = {"intent": "CONFIRM_DRAFT", "id": draft}, {"intent": "DISCARD_DRAFT", "id": draft}
        options = {
            "PENDING_DECISION": [(draft, save)],
            "PENDING_NO": [(draft, drop), (proposal, {"intent": REJECT, "id": proposal})],
            APPROVE: [(proposal, {"intent": APPROVE, "id": proposal}), (draft, save)],
            REJECT: [(proposal, {"intent": REJECT, "id": proposal}), (draft, drop)],
            "CONFIRM_DRAFT": [(draft, save)],
            "DISCARD_DRAFT": [(draft, drop)],
        }[kind]
        return next((decision for pending, decision in options if pending is not None), None)

    @staticmethod
    def _pending_draft(db, user, day):
        row = db.execute("SELECT receipt_id FROM dm_drafts WHERE user_id=? AND day=?", (user, day)).fetchone()
        return row["receipt_id"] if row else None

    @staticmethod
    def _pending_proposal(db, user, day):
        # Proposing cancels earlier pending ones, so there is at most one per day.
        row = db.execute("SELECT id FROM recovery_proposals WHERE user_id=? AND day=? AND status='pending' "
                         "ORDER BY id DESC LIMIT 1", (user, day)).fetchone()
        return row["id"] if row else None

    def _pending_quiz(self, user, day):
        """The quiz attempt waiting for an answer, if it was offered in the last two hours."""
        with self.store.connect() as db:
            row = db.execute("SELECT id, created_at FROM verification_attempts WHERE user_id=? AND day=? "
                             "AND state='pending' ORDER BY id DESC LIMIT 1", (user, day)).fetchone()
        if row and now_kst() - datetime.fromisoformat(row["created_at"]) <= QUIZ_REPLY_WINDOW:
            return row["id"]
        return None

    @staticmethod
    def _pending_prompt(db, user, day, lang):
        draft = db.execute("SELECT receipt_id FROM dm_drafts WHERE user_id=? AND day=?", (user, day)).fetchone()
        if draft:
            n = draft["receipt_id"]
            return say(lang, f"To save your draft, reply yes (or “confirm #{n}”).",
                       f"Để xác nhận nội dung vừa tóm tắt, hãy trả lời “đúng” (hoặc “xác nhận #{n}”).",
                       f"요약한 내용을 저장하려면 “네”라고 답해 주세요 (또는 “확인 #{n}”).")
        proposal = db.execute("SELECT id FROM recovery_proposals WHERE user_id=? AND day=? AND status='pending' "
                              "ORDER BY id DESC LIMIT 1", (user, day)).fetchone()
        if proposal:
            n = proposal["id"]
            return say(lang, f"Plan changes need an explicit approval: reply approve or reject (or “approve #{n}”).",
                       f"Để duyệt đề xuất đang chờ, hãy trả lời “đồng ý” (hoặc “đồng ý #{n}”).",
                       f"대기 중인 제안을 승인하려면 “승인”이라고 답해 주세요 (또는 “승인 #{n}”).")
        return say(lang, "There is no pending draft or proposal today. What would you like to record or move?",
                   "Hôm nay chưa có bản nháp hoặc đề xuất nào chờ xác nhận. Bạn muốn ghi nhận hay chuyển bao nhiêu phút?",
                   "오늘 확인할 초안이나 제안이 없어요. 기록하거나 미루고 싶은 시간을 알려 주세요.")

    @staticmethod
    def _latest(db, user, day):
        return db.execute("SELECT id,minutes FROM progress_events WHERE user_id=? AND day=? "
                          "ORDER BY id DESC LIMIT 1", (user, day)).fetchone()

    def _confirm(self, db, user, day, decision, lang):
        draft = db.execute("SELECT * FROM dm_drafts WHERE user_id=? AND day=? AND receipt_id=?",
                           (user, day, decision["id"])).fetchone()
        if draft is None:
            return say(lang, "I couldn't find that pending draft for today. Please send the request again.",
                       "Không tìm thấy bản nháp đó trong hôm nay. Bạn hãy gửi lại yêu cầu.", "오늘 해당 초안을 찾을 수 없어요. 요청을 다시 보내 주세요.")
        db.execute("DELETE FROM dm_drafts WHERE user_id=?", (user,))
        if decision["intent"] == "DISCARD_DRAFT":
            return say(lang, "Draft discarded.", "Đã hủy bản nháp.", "초안을 취소했어요.")
        plan = self.store.plan(db, user, day)
        latest = self._latest(db, user, day)
        if plan["revision"] != draft["revision"] or draft["report_id"] != (latest["id"] if latest else None):
            return say(lang, "Your plan or reported total changed since that draft. Please send an updated request.",
                       "Kế hoạch hoặc số phút đã ghi đã thay đổi. Bạn hãy gửi lại yêu cầu.", "목표나 공부 기록이 바뀌었어요. 요청을 다시 보내 주세요.")
        actions = validate({"actions": json.loads(draft["actions"])})
        return self._apply(db, user, day, actions, lang)

    def _apply(self, db, user, day, actions, lang):
        # Fixed order ensures the proposal and quiz refer to the confirmed report.
        slots = {item["intent"]: item for item in actions}
        messages = []
        proposal = None
        if REPORT in slots:
            minutes = slots[REPORT]["minutes"]
            if self.store.report(user, day, minutes, db=db).changed:
                messages.append(say(lang, f"Recorded {minutes} self-reported minutes for today.",
                                    f"Đã ghi {minutes} phút bạn tự báo cáo hôm nay.", f"오늘 직접 보고한 공부 시간 {minutes}분을 기록했어요."))
            else:
                messages.append(unchanged_total(lang, minutes))
        if OBSTACLE in slots:
            reason = slots[OBSTACLE]["reason"]
            record(db, user, day, "obstacle_reported", {"reason": reason})
            label = obstacle(lang, reason)
            messages.append(say(lang, f"I've noted the obstacle ({reason}).", f"Đã ghi nhận trở ngại: {label}.", f"어려움도 기록했어요: {label}."))
        if RECOVER in slots:
            reason = slots.get(OBSTACLE, {}).get("reason", "Requested a plan adjustment")
            try:
                proposal = self.recovery.propose(user, day, reason, tomorrow=slots[RECOVER]["when"] == "tomorrow",
                                                 remaining_minutes=slots[RECOVER].get("remaining_minutes"), db=db)
            except ValueError as exc:
                messages.append(domain_message(lang, exc))
            else:
                messages.append(proposal_reply(lang, proposal))
        if QUIZ in slots:
            task = self.verification.start(user, day, db=db)
            messages.append(task["prompt"] + " " + say(lang, "Just reply with the English meaning.",
                            "Chỉ cần trả lời nghĩa tiếng Anh.", "영어 뜻만 답해 주세요."))
        if QUERY in slots:
            messages.append(self._progress(db, user, day, slots[QUERY]["day"], lang))
        reply = " ".join(messages)
        if proposal and lang == "vi":
            return AdjustmentPrompt(reply, proposal["id"])
        return reply

    def _progress(self, db, user, day, period, lang):
        if period == "yesterday":
            day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
            if not db.execute("SELECT 1 FROM daily_goals WHERE user_id=? AND day=?", (user, day)).fetchone():
                return say(lang, f"I have no saved plan or report for {day}.", f"Chưa có kế hoạch hay báo cáo cho ngày {day}.", f"{day}의 계획이나 보고 기록이 없어요.")
        data = self.store.summary(user, day, db=db)
        if data is None:
            return domain_message(lang, "Create a goal first with /setgoal.")
        minutes = data["minutes"]
        progress = ("You haven't reported a total" if minutes is None else f"You reported {minutes} minutes")
        message = (f"{progress} for {data['title']} on {day}. The target is {data['target']} minutes "
                   f"(originally {data['original_target']}).")
        title, target, original = data["title"], data["target"], data["original_target"]
        vi = "Chưa có báo cáo" if minutes is None else f"Bạn đã tự báo cáo {minutes} phút"
        ko = "보고한 시간이 없어요" if minutes is None else f"직접 보고한 시간은 {minutes}분이에요"
        message = say(lang, message, f"{vi} cho {title} ngày {day}. Mục tiêu là {target} phút (ban đầu {original}).",
                      f"{day} {title}: {ko}. 목표는 {target}분(원래 {original}분)이에요.")
        if minutes is not None:
            state = data["verification_state"]
            message += (say(lang, " No vocabulary result yet.", " Chưa có kết quả kiểm tra từ vựng.", " 아직 단어 퀴즈 결과가 없어요.")
                        if state == "reported" else say(lang, f" Vocabulary quiz: {state}.", f" Kết quả từ vựng: {state}.", f" 단어 퀴즈: {state}."))
            if lang == "en":
                message += " Study time is self-reported."
        if period == "today":
            n = data["streak"]
            message += say(lang, f" Streak: {n} day{'s' if n != 1 else ''} in a row meeting your goal.",
                           f" Chuỗi ngày liên tiếp đạt mục tiêu: {n}.", f" 연속 목표 달성: {n}일.")
        if period == "today" and data["recoveries"]:
            proposal = data["recoveries"][-1]
            if proposal["status"] == "pending":
                n = proposal["id"]
                message += say(lang, f" Proposal #{n} is awaiting your decision.", f" Đề xuất #{n} đang chờ duyệt.", f" 제안 #{n}이 승인 대기 중이에요.")
        return message
