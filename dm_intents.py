"""Closed DM intent vocabulary and conservative whole-message grammar."""
import logging
import re
import unicodedata
from dataclasses import dataclass

REPORT = "REPORT_PROGRESS"
QUERY = "QUERY_PROGRESS"
OBSTACLE = "REPORT_OBSTACLE"
RECOVER = "REQUEST_RECOVERY"
APPROVE = "APPROVE_ADJUSTMENT"
REJECT = "REJECT_ADJUSTMENT"
QUIZ = "REQUEST_VERIFICATION"
CHAT = "GENERAL_CONVERSATION"
CLARIFY = "CLARIFY"
log = logging.getLogger(__name__)


def choice(*values):
    return {"type": "string", "enum": list(values)}


SLOTS = {
    REPORT: {"minutes": {"type": "integer", "minimum": 0, "maximum": 1440}},
    QUERY: {"day": choice("today", "yesterday")},
    OBSTACLE: {"reason": choice("tired", "busy", "other")},
    RECOVER: {"when": choice("today", "tomorrow")},
    QUIZ: {}, CHAT: {}, CLARIFY: {},
}
REMAINING = {"remaining_minutes": {"type": "integer", "minimum": 1, "maximum": 1440}}
# Approval, draft confirmation and quiz answers require literal user input.
SCHEMA = {"type": "object", "properties": {"actions": {
    "type": "array", "minItems": 1, "maxItems": 4, "items": {"anyOf": [
        {"type": "object", "properties": {"intent": choice(intent), **slots, **(REMAINING if intent == RECOVER else {})},
         "required": ["intent", *slots], "additionalProperties": False}
        for intent, slots in SLOTS.items()
    ]}}}, "required": ["actions"], "additionalProperties": False}


def action(intent, **slots):
    return {"intent": intent, **slots}


def validate(value):
    if type(value) is not dict or set(value) != {"actions"}:
        raise ValueError("Invalid intent envelope")
    actions = value["actions"]
    if type(actions) is not list or not 1 <= len(actions) <= 4:
        raise ValueError("Invalid intent count")
    seen = set()
    for item in actions:
        if type(item) is not dict or type(item.get("intent")) is not str:
            raise ValueError("Invalid intent")
        intent = item["intent"]
        if intent not in SLOTS or intent in seen:
            raise ValueError("Unknown, repeated or malformed intent")
        slots = SLOTS[intent]
        if intent == RECOVER and "remaining_minutes" in item and item.get("when") == "tomorrow":
            slots = {**slots, **REMAINING}
        if set(item) != {"intent", *slots}:
            raise ValueError("Missing or unexpected slot")
        seen.add(intent)
        for name, spec in slots.items():
            slot = item[name]
            if spec["type"] == "integer":
                if type(slot) is not int or not spec["minimum"] <= slot <= spec["maximum"]:
                    raise ValueError("Invalid minutes")
            elif type(slot) is not str or slot not in spec["enum"]:
                raise ValueError("Invalid slot")
    return actions


@dataclass(frozen=True)
class Interpretation:
    actions: list
    source: str
    latency_ms: float = 0
    reason: str = ""


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("’", "'").split())


def check_report_evidence(actions, text):
    """Numeric slots must agree with the stated duration before a draft is shown."""
    if actions == [action(CLARIFY)]:
        return
    # Preserve recognized clauses even when another clause needed inference.
    # Remove only a fully matched transfer clause from the study-duration check.
    evidence = []
    transfer = None
    for piece in clauses(normalized(text)):
        known = clause_action(piece)
        if known and known not in actions:
            raise ValueError("Missing or conflicting intent")
        if known and "remaining_minutes" in known:
            transfer = known
        else:
            evidence.append(piece)
    recovery = next((item for item in actions if item["intent"] == RECOVER), {})
    if "remaining_minutes" in recovery and recovery != transfer:
        raise ValueError("Ungrounded transfer amount")
    minutes = next((item["minutes"] for item in actions if item["intent"] == REPORT), None)
    words = " ".join(evidence)
    if minutes is None:
        if recovery and re.search(r"[0-9]", words):
            raise ValueError("Unassigned numeric duration")
        return
    numbers = re.findall(r"[-+]?[0-9]+(?:[.,][0-9]+)?", words)
    if numbers:
        match = re.search(r"(?<![\w.+-])([0-9]{1,4})\s*(minutes?\b|mins?\b|phút\b|분|hours?\b|giờ\b|시간)", words)
        if match and numbers == [match[1]]:
            multiplier = 60 if match[2] in {"hour", "hours", "giờ", "시간"} else 1
            if int(match[1]) * multiplier == minutes:
                return
        raise ValueError("Unclear or conflicting numeric duration")
    # Only offer a word-number draft when a quantity accompanies the unit.
    quantity = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|\w+ty)"
                r"\s+(?:minutes?|hours?)\b|(?:một|hai|ba|bốn|năm|lăm|sáu|bảy|tám|chín|mười|mươi)\s+(?:phút|giờ)\b|"
                r"[일이삼사오육칠팔구십백한두세네열스무]\s*(?:분|시간)|half an hour")
    if not re.search(quantity, words):
        raise ValueError("Ask for an explicit duration")


# Whole-message replies to the single pending draft or plan change. These are the user's literal
# words, never model output. A plain "yes" confirms a draft but does not approve a plan change.
YES = {"được chứ", "được", "yes", "yep", "yeah", "yup", "sure", "ok", "okay", "correct", "that's right",
       "looks good", "sounds good", "right", "네", "예", "응", "좋아요", "맞아요", "có", "đúng", "đúng rồi", "ừ"}
NO = {"no", "nope", "nah", "wrong", "not quite", "아니요", "아니", "아니에요", "không", "sai", "không phải"}
BARE = {
    APPROVE: {"approve", "approve it", "i approve", "đồng ý", "phê duyệt", "승인", "승인할게요", "승인해요"},
    REJECT: {"reject", "reject it", "i reject", "từ chối", "거절", "거절할게요"},
    "CONFIRM_DRAFT": {"confirm", "confirm it", "save it", "xác nhận", "확인"},
    "DISCARD_DRAFT": {"discard", "discard it", "cancel", "hủy", "취소"},
}


DROP = re.compile(r"(?:drop|remove|delete|skip|bỏ|xóa|빼기|빼|삭제)\s+(#?\d{1,2}(?:\s*(?:,|and|&|\s)\s*#?\d{1,2})*)")


def control(text):
    """Full-message decisions cannot be inferred or mixed with other actions."""
    reply = text.rstrip(".!? ")
    if match := DROP.fullmatch(reply):
        return action("DROP_IMPORT_ITEMS", items=[int(n) for n in re.findall(r"\d+", match[1])])
    if reply in YES:
        return action("PENDING_DECISION")
    if reply in NO:
        return action("PENDING_NO")
    for kind, words in BARE.items():
        if reply in words:
            return action(kind, id=None)  # resolved to the single pending item for today
    for kind, verbs in (
        (APPROVE, "approve|i approve|đồng ý|phê duyệt|승인"),
        (REJECT, "reject|i reject|từ chối|거절"),
        ("CONFIRM_DRAFT", "confirm|i confirm|xác nhận|확인"),
        ("DISCARD_DRAFT", "discard|hủy|취소"),
    ):
        match = re.fullmatch(rf"(?:{verbs})\s+(?:proposal\s+)?#?([1-9][0-9]{{0,17}})[.!]?", text)
        if match:
            return action(kind, id=int(match[1]))
    match = re.fullmatch(r"(?:answer|trả lời|정답)\s+#?([1-9][0-9]{0,17})\s+([^\W\d_]{1,100})[.!]?", text)
    if match:
        return action("ANSWER_QUIZ", id=int(match[1]), answer=match[2])


DONT_KNOW = {"i don't know", "i dont know", "idk", "no idea", "don't know", "dont know", "not sure",
             "모르겠어요", "몰라요", "모르겠다", "không biết", "chịu"}
ANSWER_PREFIX = (r"^(?:i think |maybe |hmm,? |um,? )?(?:it means |it's |it is |that's |is it |means |"
                 r"the answer is |nghĩa là |là )?(?:a |an |the )?")


# Words that make a short message a request, not an answer ("try quizzing me", "another one please").
NOT_ANSWERS = {"quiz", "quizzing", "test", "testing", "me", "please", "again", "another", "next", "skip",
               "word", "words", "vocab", "vocabulary", "try", "give", "ask", "one", "more", "stop"}


def quiz_answer(text):
    """A short reply such as "school", "it means school?" or "I don't know", or None."""
    words = normalized(text).strip(" .!?\"'“”")
    if words in DONT_KNOW:
        return "(don't know)"
    words = re.sub(ANSWER_PREFIX, "", words).strip()
    if re.fullmatch(r"[^\W\d_]+(?: [^\W\d_]+)?", words) and not set(words.split()) & NOT_ANSWERS:
        return words


_MIN = r"(?P<minutes>[0-9]{1,4})"
BARE_DURATION = rf"{_MIN}\s*(?:minutes?|mins?|phút|분)"
REPORT_PATTERNS = (
    BARE_DURATION,
    rf"(?:(?:today )?i studied (?:korean )?(?:for )?|(?:today's )?total(?: is|:)? ){_MIN} (?:minutes?|mins?)(?: today)?",
    rf"{_MIN} (?:minutes?|mins?) (?:total|today)",
    rf"(?:hôm nay )?(?:tôi |mình |em )?(?:đã )?học (?:tiếng hàn )?(?:được |trong ){_MIN} phút(?: hôm nay)?",
    rf"(?:hôm nay )?(?:tôi |mình |em )?(?:đã )?học (?:tiếng hàn )?{_MIN} phút(?: hôm nay)?",
    rf"(?:오늘\s*)?(?:한국어(?:를)?\s*)?{_MIN}\s*분(?:\s*동안)?\s*(?:한국어(?:를)?\s*)?공부(?:했어요|했습니다|했다|했어|했고|했는데|했지만)",
)
PHRASES = (
    (r"i(?:'m| am| feel) (?:too |really )?tired|(?:tôi |mình |em )?(?:hơi |rất )?mệt|(?:너무\s*)?피곤(?:해요|합니다|해서|하다|해|하고|한데)", OBSTACLE, {"reason": "tired"}),
    (r"i(?:'m| am) busy|(?:tôi |mình )?bận|바빠요", OBSTACLE, {"reason": "busy"}),
    (r"(?:i )?(?:(?:want|need|would like) to |will )?(?:finish|continue)(?: (?:it|the rest))? tomorrow|"
     r"(?:tôi |mình |em )?(?:muốn |sẽ )?(?:hoàn thành|học tiếp|làm nốt|học nốt)(?: vào)? ngày mai|"
     r"내일\s*(?:나머지(?:를)?\s*)?(?:마저\s*|이어서\s*)?(?:끝내|공부하|하|마무리하)고\s*싶(?:어요|습니다|어|다)", RECOVER, {"when": "tomorrow"}),
    (r"(?:can i (?:have|try) |i (?:want|need) )?a smaller (?:target|goal)(?: today)?|"
     r"giảm mục tiêu hôm nay|오늘 목표를 줄이고 싶어요", RECOVER, {"when": "today"}),
    (r"(?:how (?:am i doing|much did i study)|(?:show |what(?:'s| is) )?(?:today's |my )?progress)(?: today)?|"
     r"tiến độ(?: hôm nay)?(?: của tôi)?|오늘(?:의)? 진도(?:는| 어때| 알려줘)?", QUERY, {"day": "today"}),
    (r"how much did i study yesterday|yesterday's progress|tiến độ hôm qua|어제 진도(?:는)?", QUERY, {"day": "yesterday"}),
    (r"quiz me|test my (?:vocabulary|learning)|kiểm tra từ vựng|퀴즈(?: 내줘| 주세요)?", QUIZ, {}),
    (r"hello|hi|thanks|thank you|xin chào|cảm ơn|안녕(?:하세요)?|고마워|감사합니다", CHAT, {}),
)

MOVE_PATTERNS = (
    rf"(?:(?:tôi|mình|em) )?(?:muốn )?chuyển {_MIN} phút(?: còn lại)? sang (?:ngày )?mai",
    rf"(?:(?:i )?(?:want|need) to )?(?:move|carry over) (?:the )?(?:remaining )?{_MIN} (?:minutes?|mins?) (?:over )?to tomorrow",
    rf"(?:남은\s*)?{_MIN}\s*분(?:을|은)?\s*내일(?:로)?\s*(?:미루|넘기|옮기)고\s*싶(?:어요|습니다|어)",
)


def clauses(text):
    parts = re.split(r"\s*[,;]\s*|\.\s+|\s+(?:but|and|nhưng|và|nên|그래서)\s+|(?<=했고)\s+|"
                     r"(?<=했는데)\s+|(?<=했지만)\s+|(?<=해서)\s+|(?<=피곤하고)\s+", text.strip(".!? "))
    return [re.sub(r"^(?:but|and|nhưng|và|nên)\s+", "", piece) for piece in parts]


def clause_action(piece):
    for pattern in REPORT_PATTERNS:
        if match := re.fullmatch(pattern, piece):
            return action(REPORT, minutes=int(match["minutes"]))
    for pattern in MOVE_PATTERNS:
        if match := re.fullmatch(pattern, piece):
            return action(RECOVER, when="tomorrow", remaining_minutes=int(match["minutes"]))
    for pattern, intent, slots in PHRASES:
        if re.fullmatch(pattern, piece):
            return action(intent, **slots)


def rule_actions(text):
    text = normalized(text)
    decision = control(text)
    if decision:
        return [decision]
    if re.match(r"(?:answer|trả lời|정답|approve|reject|confirm|discard|승인|거절|확인|취소|"
                r"phê duyệt|đồng ý|từ chối|xác nhận|hủy)\s+#?[0-9]", text):
        return [action(CLARIFY)]
    # Question forms never become activity reports or approvals.
    question = "?" in text
    actions = []
    for piece in clauses(text):
        item = clause_action(piece)
        if item is None:
            return None
        actions.append(item)
    if question and any(a["intent"] not in {QUERY, RECOVER, QUIZ} for a in actions):
        return [action(CLARIFY)]
    try:
        return validate({"actions": actions})
    except ValueError:
        log.info("DM stage=rules outcome=clarify reason=invalid_slots")
        return [action(CLARIFY)]


async def interpret(text, adapter):
    if not isinstance(text, str) or len(text) > 2000 or not text.strip():
        return Interpretation([action(CLARIFY)], "input_limit")
    parsed = rule_actions(text)
    if parsed:
        log.info("DM stage=rules outcome=matched intents=%s", ",".join(a["intent"] for a in parsed))
        return Interpretation(parsed, "rule")
    log.info("DM stage=rules outcome=miss")
    return await adapter.interpret(text)
