"""Local DM wording. Goal values and proposal amounts come from domain services."""
import re
from dataclasses import dataclass

from dm_intents import OBSTACLE, QUERY, QUIZ, RECOVER, REPORT, clause_action, clauses, normalized
from time_utils import following_day


@dataclass(frozen=True)
class AdjustmentPrompt:
    text: str
    proposal_id: int


def language(text):
    words = normalized(text[:2000]) if isinstance(text, str) else ""
    if re.search(r"[가-힣]", words):
        return "ko"
    if re.search(r"[ăâđêôơư\u1ea0-\u1ef9]|\b(?:tôi|mai|hôm|phút|hàn|chào|hơi|bận|giờ)\b", words):
        return "vi"
    return "en"


def say(lang, en, vi, ko):
    return {"en": en, "vi": vi, "ko": ko}[lang]


def clarification(lang, text):
    words = normalized(text) if isinstance(text, str) and len(text) <= 2000 else ""
    for piece in clauses(words):
        known = clause_action(piece)
        if known and 0 < known.get("remaining_minutes", 0) <= 1440:
            n = known["remaining_minutes"]
            return say(lang, f"Are you asking to move {n} minutes to tomorrow? How many minutes did you actually study today?",
                       f"Bạn muốn chuyển {n} phút sang ngày mai phải không? Hôm nay bạn đã học tổng cộng bao nhiêu phút?",
                       f"{n}분을 내일로 미루고 싶으신가요? 오늘 실제로 공부한 시간은 총 몇 분인가요?")
    if re.search(r"[0-9]|phút|minutes?|분|mai|tomorrow|내일", words):
        return say(lang, "Would you like to record a daily total or move time to tomorrow? "
                   "Please give today's studied minutes and the minutes to move separately.",
                   "Hôm nay bạn đã học tổng cộng bao nhiêu phút? Nếu muốn dời phần còn lại, hãy cho biết riêng số phút muốn chuyển sang ngày mai.",
                   "오늘 공부한 시간은 총 몇 분인가요? 내일로 미루고 싶은 시간이 있다면 몇 분인지 따로 알려 주세요.")
    return say(lang, "Would you like to log study time, check your progress, or change today's plan? "
               "For example, say “today's total is 20 minutes”, “how am I doing?” or “finish tomorrow”.",
               "Bạn muốn ghi thời gian đã học, xem tiến độ hay điều chỉnh kế hoạch? Ví dụ: “Hôm nay tôi học 20 phút”.",
               "공부 시간을 기록할까요, 진도를 확인할까요, 아니면 계획을 조정할까요? 예를 들어 “오늘 20분 공부했어요”라고 보내 주세요.")


def obstacle(lang, reason):
    return {"tired": say(lang, "tired", "mệt", "피곤함"), "busy": say(lang, "busy", "bận", "바쁨"),
            "other": say(lang, "other", "trở ngại khác", "기타 어려움")}[reason]


def domain_message(lang, error):
    message = str(error)
    if "/setgoal" in message:
        return say(lang, message, "Hãy tạo mục tiêu trước bằng /setgoal.", "/setgoal로 목표를 먼저 만들어 주세요.")
    if "No pending proposal" in message:
        return say(lang, message, "Không tìm thấy đề xuất đó đang chờ duyệt trong hôm nay.", "오늘 승인 대기 중인 해당 제안이 없어요.")
    if "already meets" in message or "no remaining target" in message:
        return say(lang, message, "Bạn đã đạt mục tiêu hôm nay. Không còn phút nào để chuyển.", "오늘 목표를 달성해 미룰 시간이 없어요.")
    if "first" in message and ("total" in message or "progress" in message.lower()):
        return say(lang, message, "Bạn đã học bao nhiêu phút hôm nay? Hãy gửi tổng số phút trước.", "오늘 공부한 총시간을 먼저 알려 주세요.")
    if "fit tomorrow" in message:
        return say(lang, message, "Số phút còn lại không phù hợp với mục tiêu ngày mai. Bạn muốn điều chỉnh thế nào?",
                   "남은 시간이 내일 목표에 맞지 않아요. 어떻게 조정할까요?")
    return say(lang, message, "Không thể áp dụng yêu cầu này cho kế hoạch hiện tại. Bạn hãy kiểm tra tiến độ và gửi lại yêu cầu.",
               "현재 계획에 이 요청을 적용할 수 없어요. 진도를 확인하고 다시 요청해 주세요.")


def draft_reply(lang, actions, plan, prior, remaining, day, number):
    parts = []
    for item in actions:
        kind = item["intent"]
        if kind == REPORT:
            minutes, target, title = item["minutes"], plan["target"], plan["title"]
            part = say(lang, f"record {minutes} minutes as today's total for {title} (target {target})",
                       f"hôm nay bạn đã học {minutes}/{target} phút cho mục tiêu {title}",
                       f"오늘 {title} 목표 {target}분 중 {minutes}분을 공부하셨고")
            if prior:
                previous = prior["minutes"]
                part += say(lang, f" (currently {previous})", f" (hiện đã ghi {previous} phút)", f" (현재 기록 {previous}분)")
        elif kind == OBSTACLE:
            reason = obstacle(lang, item["reason"])
            vi = f"bạn thấy {reason}" if item["reason"] == "tired" else f"bạn gặp trở ngại ({reason})"
            part = say(lang, f"note the obstacle ({reason})", vi, f"어려움은 {reason}")
        elif kind == RECOVER and item["when"] == "tomorrow":
            date = following_day(day)
            if remaining is None:
                part = say(lang, f"propose moving the remaining minutes to tomorrow ({date})",
                           f"bạn muốn chuyển phần còn lại sang ngày mai ({date})", f"남은 시간을 내일({date})로 미루기")
            else:
                part = say(lang, f"propose moving the remaining {remaining} minutes to tomorrow ({date})",
                           f"bạn muốn chuyển {remaining} phút còn lại sang ngày mai ({date})",
                           f"남은 {remaining}분을 내일({date})로 미루기")
        elif kind == RECOVER:
            part = say(lang, "propose a smaller target for today", "bạn muốn giảm mục tiêu hôm nay", "오늘 목표 줄이기")
        elif kind == QUIZ:
            part = say(lang, "start a vocabulary quiz", "bạn muốn kiểm tra từ vựng", "단어 퀴즈 시작")
        elif kind == QUERY:
            period = item["day"]
            part = say(lang, f"show progress for {period}", "xem tiến độ " + ("hôm nay" if period == "today" else "hôm qua"),
                       ("오늘" if period == "today" else "어제") + " 진도 확인")
        parts.append(part)
    return say(lang, "Did I understand you correctly? You'd like me to " + ", and ".join(parts)
               + f". Reply “confirm #{number}” or “discard #{number}”. Any plan change needs a separate approval.",
               "Tôi hiểu là " + ", ".join(parts) + f". Đúng vậy không? Trả lời “xác nhận #{number}” để ghi nhận "
               f"hoặc “hủy #{number}”. Bạn sẽ duyệt riêng trước khi kế hoạch thay đổi.",
               "이렇게 이해했어요. " + ", ".join(parts) + f". 맞으면 “확인 #{number}”, 취소하려면 “취소 #{number}”라고 답해 주세요. 계획 변경은 별도로 승인해 주세요.")


def proposal_reply(lang, proposal):
    number, target = proposal["id"], proposal["proposed_target"]
    if "remaining" in proposal:
        remaining, date, future = proposal["remaining"], proposal["next_day"], proposal["next_target"]
        vi = f"Chuyển {remaining} phút còn lại sang ngày mai ({date})? Mục tiêu hôm nay sẽ là {target} phút, ngày mai là {future} phút."
        ko = f"남은 {remaining}분을 내일({date})로 미룰까요? 오늘 목표는 {target}분, 내일 목표는 {future}분이 됩니다."
    else:
        vi = f"Giảm mục tiêu hôm nay xuống {target} phút?"
        ko = f"오늘 목표를 {target}분으로 줄일까요?"
    return say(lang, f"Proposal #{number}: {proposal['explanation']} Reply “approve #{number}” or “reject #{number}”.",
               f"Đề xuất #{number}. {vi} Gửi “đồng ý #{number}” để duyệt hoặc “từ chối #{number}”. "
               "Bạn cũng có thể dùng Reply trên chính tin nhắn này và trả lời “được chứ” để duyệt.",
               f"제안 #{number}. {ko} “승인 #{number}” 또는 “거절 #{number}”라고 답해 주세요.")
