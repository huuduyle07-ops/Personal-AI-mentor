"""Confusion pairs: which deck words a learner mixes up, found from quiz history. Read-only (SELECTs only).

Three kinds of evidence, strongest first:
  meaning     a wrong answer for word X was the English meaning of another deck word Y
              (needs the typed answer, which is stored for wrong flashcard answers from now on)
  production  in an active (English -> Hangul) quiz for X, the learner typed another deck word Y
  look-alike  X was missed at least LOOKALIKE_MISSES times, and a word the learner has seen, Y, is
              spelled similarly: jamo similarity >= LOOKALIKE_SIMILARITY

Jamo similarity is 1 - (edit distance / longer length) over the Unicode NFD decomposition, which splits
each Hangul syllable into its letters (학교 -> ㅎㅏㄱㄱㅛ), so 나무 / 나라 share half their letters.
"""
import json
from contextlib import nullcontext
import re
import unicodedata

from verification import user_deck

LOOKALIKE_SIMILARITY = 0.5
LOOKALIKE_MISSES = 2
WRONG = "review_required"
PUNCTUATION = ".,!?~。、…'\"“”‘’"


def jamo(text):
    """Hangul letters (and any other characters) of a word, with syllables decomposed."""
    return unicodedata.normalize("NFD", text)


def similarity(a, b):
    """1 - normalized Levenshtein distance between the jamo sequences of a and b (1.0 = identical)."""
    a, b = jamo(a), jamo(b)
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return 1 - previous[-1] / max(len(a), len(b))


def normalize_hangul(text):
    """NFC, no whitespace and no trailing punctuation: ' 학 교 . ' -> '학교'."""
    text = unicodedata.normalize("NFKC", text).strip().rstrip(PUNCTUATION + " ")
    return unicodedata.normalize("NFC", re.sub(r"\s+", "", text))


def _answer_lookup(deck):
    meanings = {}
    for question_id, question in deck.items():
        for answer in question["answers"]:
            meanings.setdefault(answer, set()).add(question_id)
    return meanings


def pairs(store, user_id, limit=3, db=None):
    """Top confusion pairs, most evidence first. Each pair: word/meaning (the one missed), other/other_meaning,
    kind, count (confusions observed, or misses for look-alikes) and similarity.
    `db` lets a caller inside a transaction include its own uncommitted answer."""
    with (nullcontext(db) if db is not None else store.connect()) as db:
        deck = user_deck(db, user_id)
        passive = db.execute("SELECT question_id, state FROM verification_attempts WHERE user_id=? AND state!='pending'",
                             (user_id,)).fetchall()
        given = db.execute("SELECT payload FROM events WHERE user_id=? AND kind='verification_result' "
                           "AND json_extract(payload,'$.state')=? AND json_extract(payload,'$.given') IS NOT NULL",
                           (user_id, WRONG)).fetchall()
        active = db.execute("SELECT question_id, state, given FROM active_attempts WHERE user_id=? AND state!='pending'",
                            (user_id,)).fetchall()
    misses, seen, found = {}, set(), {}

    def add(word_id, other_id, kind, count=1, score=None):
        if word_id == other_id or word_id not in deck or other_id not in deck:
            return
        key = (word_id, other_id)
        item = found.setdefault(key, {"meaning": 0, "production": 0, "look-alike": 0, "similarity": score})
        item[kind] += count
        if score is not None:
            item["similarity"] = score

    for row in passive:
        seen.add(row["question_id"])
        misses[row["question_id"]] = misses.get(row["question_id"], 0) + (row["state"] == WRONG)
    for row in active:
        seen.add(row["question_id"])
        misses[row["question_id"]] = misses.get(row["question_id"], 0) + (row["state"] == WRONG)

    meanings = _answer_lookup(deck)
    for row in given:
        payload = json.loads(row["payload"])
        typed = unicodedata.normalize("NFKC", str(payload["given"])).strip().casefold()
        for other_id in meanings.get(typed, ()):
            add(payload["question_id"], other_id, "meaning")

    words = {normalize_hangul(question["word"]): question_id for question_id, question in deck.items()}
    for row in active:
        if row["state"] == WRONG and row["given"]:
            other_id = words.get(normalize_hangul(row["given"]))
            if other_id is not None:
                add(row["question_id"], other_id, "production")

    for word_id, count in misses.items():
        if count < LOOKALIKE_MISSES or word_id not in deck:
            continue
        scored = [(similarity(deck[word_id]["word"], deck[other_id]["word"]), other_id)
                  for other_id in seen if other_id != word_id and other_id in deck]
        best = max(scored, default=None)
        if best and best[0] >= LOOKALIKE_SIMILARITY:
            add(word_id, best[1], "look-alike", count, best[0])

    result = []
    for (word_id, other_id), item in found.items():
        kind = next(k for k in ("meaning", "production", "look-alike") if item[k])
        score = item["similarity"]
        if score is None:
            score = similarity(deck[word_id]["word"], deck[other_id]["word"])
        result.append({"word": deck[word_id]["word"], "meaning": deck[word_id]["meaning"],
                       "other": deck[other_id]["word"], "other_meaning": deck[other_id]["meaning"],
                       "question_id": word_id, "other_id": other_id, "kind": kind,
                       "count": item["meaning"] + item["production"] + item["look-alike"], "similarity": score,
                       "rank": (item["meaning"] + item["production"], item["look-alike"], score)})
    result.sort(key=lambda pair: pair["rank"], reverse=True)
    # A pair confused both ways is listed once.
    unique, used = [], set()
    for pair in result:
        key = frozenset((pair["question_id"], pair["other_id"]))
        if key not in used:
            used.add(key)
            unique.append(pair)
    return unique[:limit] if limit else unique


def contrast(pair):
    return f"{pair['word']} = {pair['meaning']}, {pair['other']} = {pair['other_meaning']}"


def describe(pair):
    if pair["kind"] == "meaning":
        why = f"you answered {pair['word']} with the meaning of {pair['other']}"
    elif pair["kind"] == "production":
        why = f"you typed {pair['other']} when asked for {pair['word']}"
    else:
        why = f"{pair['word']} was missed {pair['count']} times and looks {pair['similarity']:.0%} like {pair['other']}"
    times = f" ({pair['count']}×)" if pair["kind"] != "look-alike" and pair["count"] > 1 else ""
    return f"{contrast(pair)}: {why}{times}"


def format_pairs(found):
    if not found:
        return ""
    return "\n".join(["Often mixed up:"] + [f"• {describe(pair)}" for pair in found])
