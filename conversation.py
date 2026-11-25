"""Authorized DM delivery and atomic dispatch. The interpreter has no store handle."""
import asyncio
import json
import logging
import re
from datetime import date, timedelta

from dm_intents import (
    APPROVE, BARE_DURATION, CHAT, CLARIFY, OBSTACLE, QUERY, QUIZ, RECOVER, REJECT, REPORT,
    interpret, normalized, validate,
)
from memory import record
from dm_replies import AdjustmentPrompt, clarification, domain_message, draft_reply, language, obstacle, proposal_reply, say
from time_utils import valid_day

log = logging.getLogger(__name__)


class Conversation:
    def __init__(self, store, allowed_user_id, adapter, recovery, verification, today):
        self.store, self.allowed_user_id = store, allowed_user_id
        self.adapter, self.recovery, self.verification = adapter, recovery, verification
        self.today = today
        self.lock = asyncio.Lock()

    async def receive(self, user, message_id, text, sent_day, *, reply_to=None):
        if not self.allowed_user_id or user != self.allowed_user_id:
            return None
        if type(message_id) is not int or not 0 < message_id < 2**63:
            raise ValueError("Invalid Discord event ID")
        valid_day(sent_day)
        lang = language(text)
        async with self.lock:
            with self.store.connect() as db:
                if db.execute("SELECT 1 FROM dm_receipts WHERE message_id=?", (message_id,)).fetchone():
                    return None
            result = None
            if sent_day == self.today():
                result = await interpret(text, self.adapter)
                log.warning(
                    "DM interpretation: source=%s, intents=%s, latency_ms=%s",
                    result.source,
                    [action.get("intent") for action in result.actions],
                    result.latency_ms,
                )
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
            if kind == "PENDING_DECISION":
                if reply_to is not None and normalized(text).rstrip(".! ") == "được chứ":
                    prompt = db.execute("SELECT json_extract(payload,'$.proposal_id') AS proposal_id FROM events "
                                        "WHERE user_id=? AND day=? AND kind='dm_approval_prompt' AND source='system' "
                                        "AND json_extract(payload,'$.message_id')=? LIMIT 1", (user, day, reply_to)).fetchone()
                    if prompt:
                        self.recovery.resolve(user, day, prompt["proposal_id"], "approve", db=db)
                        log.info("DM stage=dispatch outcome=approved_from_referenced_prompt")
                        return f"Đã duyệt đề xuất #{prompt['proposal_id']}."
                return self._pending_prompt(db, user, day, lang)
            if kind in {"CONFIRM_DRAFT", "DISCARD_DRAFT"}:
                return self._confirm(db, user, day, first, lang)
            if kind in {APPROVE, REJECT}:
                decision = "approve" if kind == APPROVE else "reject"
                status = self.recovery.resolve(user, day, first["id"], decision, db=db)
                number = first["id"]
                return say(lang, f"Proposal #{number} {status}. Say “how am I doing?” to review your plan.",
                           f"Đã {'duyệt' if decision == 'approve' else 'từ chối'} đề xuất #{number}.",
                           f"제안 #{number}을 {'승인' if decision == 'approve' else '거절'}했어요.")
            if kind == "ANSWER_QUIZ":
                answer = self.verification.answer(user, day, first["id"], first["answer"], db=db)
                state = answer["state"]
                return say(lang, f"Vocabulary result: {state}. {answer['message']} Study time is self-reported.",
                           f"Kết quả từ vựng: {state}. Thời gian học do bạn tự báo cáo.",
                           f"단어 퀴즈 결과: {state}. 공부 시간은 직접 보고한 기록이에요.")

        validate({"actions": actions})
        kinds = {item["intent"] for item in actions}
        if kinds == {QUERY}:
            return self._progress(db, user, day, first["day"], lang)
        if kinds == {CHAT}:
            return clarification(lang, text)
        if CLARIFY in kinds or CHAT in kinds:
            log.info("DM stage=dispatch outcome=clarify source=%s reason=%s", result.source, result.reason or kind)
            return clarification(lang, text)
        plan = self.store.plan(db, user, day)
        prior = self._latest(db, user, day)
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

    @staticmethod
    def _pending_prompt(db, user, day, lang):
        draft = db.execute("SELECT receipt_id FROM dm_drafts WHERE user_id=? AND day=?", (user, day)).fetchone()
        if draft:
            n = draft["receipt_id"]
            return say(lang, f"To confirm your draft, reply “confirm #{n}”.",
                       f"Để xác nhận nội dung vừa tóm tắt, hãy trả lời “xác nhận #{n}”.",
                       f"요약한 내용을 확인하려면 “확인 #{n}”라고 답해 주세요.")
        proposal = db.execute("SELECT id FROM recovery_proposals WHERE user_id=? AND day=? AND status='pending' "
                              "ORDER BY id DESC LIMIT 1", (user, day)).fetchone()
        if proposal:
            n = proposal["id"]
            return say(lang, f"To approve the pending plan, reply “approve #{n}”.",
                       f"Để duyệt đề xuất đang chờ, hãy trả lời “đồng ý #{n}”.",
                       f"대기 중인 제안을 승인하려면 “승인 #{n}”라고 답해 주세요.")
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
            self.store.report(user, day, minutes, db=db)
            messages.append(say(lang, f"Recorded {minutes} self-reported minutes for today.",
                                f"Đã ghi {minutes} phút bạn tự báo cáo hôm nay.", f"오늘 직접 보고한 공부 시간 {minutes}분을 기록했어요."))
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
            n = task["id"]
            messages.append(task["prompt"] + " " + say(lang, f"Reply “answer #{n} <your answer>”. This checks vocabulary only.",
                            f"Trả lời “trả lời #{n} <đáp án>”. Bài này chỉ kiểm tra từ vựng.",
                            f"“정답 #{n} <답>”이라고 답해 주세요. 단어 지식만 확인해요."))
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
        if period == "today" and data["recoveries"]:
            proposal = data["recoveries"][-1]
            if proposal["status"] == "pending":
                n = proposal["id"]
                message += say(lang, f" Proposal #{n} is awaiting your decision.", f" Đề xuất #{n} đang chờ duyệt.", f" 제안 #{n}이 승인 대기 중이에요.")
        return message
