"""Local terminal chat: the same conversation logic and database as the Discord bot, without Discord.

Run with `uv run python chat.py`. Plain messages go through the DM conversation handler;
lines starting with `/` mirror the Discord slash commands. Type /help for the list.
"""
import argparse
import asyncio
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from config import Config
from conversation import Conversation
from core import GoalStore, daily_review
from dm_replies import AdjustmentPrompt
from llm_adapter import OllamaAdapter
from memory import Memory
from recovery import Recovery
from router import AuditedRouter, RoutingInput
from study_notes import NoteImporter
from time_utils import now_kst
import checkin_policy
import confusion
import focus
from verification import Verification, explain, format_recall_report

LOCAL_USER_ID = 1
# Local message IDs live in a high range that Discord snowflakes won't reach for decades,
# so local and Discord messages never collide in the shared receipt table.
LOCAL_ID_BASE = 2**62

HELP = """Talk normally, for example:
  today's total is 20 minutes      how am I doing?      quiz me
  I'm tired, finish tomorrow       yes / no             approve / reject
After "quiz me", just type the meaning (e.g. school) or "I don't know". After "/quiz active", type the Hangul (e.g. 학교).
Commands:
  /setgoal <title> <minutes>   /report <minutes>   /summary
  /quiz  /quiz context (a Korean sentence using the word)   /answer <id> <answer>
  /quiz active  (see the English meaning, type the Korean word in Hangul; or say "quiz me in Korean")
  /recover <reason>            /adjust <id> approve|reject|cancel
  /focus  (today's study, words and check-ins at a glance)
  /recall  (due words, predicted recall and why)   /policy  (what the check-in bandit would do now)
  /import <file>  (add words from notes: .txt .md .csv .png .jpg; you confirm before saving)
  /history                     /help               /quit"""


def today():
    return now_kst().date().isoformat()


class LocalChat:
    def __init__(self, config, store, user_id):
        self.store, self.user_id = store, user_id
        self.routing = AuditedRouter(store)
        self.verification, self.recovery, self.memory = Verification(store), Recovery(store), Memory(store)
        self.conversation = Conversation(store, user_id, OllamaAdapter(config), self.recovery,
                                         self.verification, today, NoteImporter(config))
        self._last_id = 0

    def next_message_id(self):
        self._last_id = max(self._last_id + 1, LOCAL_ID_BASE + time.time_ns() // 1000)
        return self._last_id

    async def handle(self, line):
        """Return the reply text for one input line, or None when there is nothing to say."""
        line = line.strip()
        if not line:
            return None
        try:
            if line.lower().startswith("/import"):
                path = line[len("/import"):].strip().strip("'\"")
                return await self.import_file(path) if path else "Usage: /import <path to .txt, .md, .csv, .png or .jpg>"
            if line.lower().split()[:2] in (["/quiz", "active"], ["/quiz", "spelling"]):
                self.routing.route(self.user_id, RoutingInput(command="quiz"))
                return (await self.conversation.active_quiz_reply(self.user_id, today(), "en"))[1]
            if line.lower().split()[:2] in (["/quiz", "context"], ["/quiz", "sentence"]):
                return (await self.conversation.quiz_reply(self.user_id, today(), "en", sentence=True))[1]
            if line.startswith("/"):
                return await asyncio.to_thread(self.command, line)
            reply = await self.conversation.receive(self.user_id, self.next_message_id(), line, today())
            return reply.text if isinstance(reply, AdjustmentPrompt) else reply
        except (ValueError, PermissionError) as error:
            return str(error)

    async def import_file(self, path):
        """Extract vocabulary from a local notes file or photo; nothing is saved until you reply yes."""
        source = Path(path).expanduser()
        if not source.is_file():
            return f"No file at {source}."
        data = await asyncio.to_thread(source.read_bytes)
        return await self.conversation.import_notes(self.user_id, self.next_message_id(), source.name, data, today())

    def command(self, line):
        name, _, rest = line[1:].partition(" ")
        name, rest, args = name.lower(), rest.strip(), rest.split()
        if name == "help":
            return HELP
        user, day = self.user_id, today()
        handlers = {
            "setgoal": self._setgoal, "report": self._report, "summary": self._summary, "quiz": self._quiz,
            "answer": self._answer, "recover": self._recover, "adjust": self._adjust, "history": self._history,
            "recall": self._recall, "policy": self._policy, "focus": self._focus,
        }
        if name not in handlers:
            return f"Unknown command /{name}. Type /help for the list."
        minutes = number(args[0]) if name == "report" and args else None
        self.routing.route(user, RoutingInput(command=name, minutes=minutes))
        return handlers[name](user, day, rest, args)

    def _setgoal(self, user, day, rest, args):
        if len(args) < 2:
            raise ValueError("Usage: /setgoal <title> <minutes>")
        title, minutes = rest.rsplit(maxsplit=1)[0], number(args[-1])
        self.store.set_goal(user, title, minutes)
        return f"Goal saved: {title.strip()} — {minutes} min/day"

    def _report(self, user, day, rest, args):
        if len(args) != 1:
            raise ValueError("Usage: /report <minutes>")
        minutes = number(args[0])
        if not self.store.report(user, day, minutes).changed:
            return f"{minutes} minutes is already recorded as today's total, so nothing changed."
        return f"Reported {minutes} minutes. Try /quiz for the optional vocabulary check."

    def _summary(self, user, day, rest, args):
        return daily_review(self.store.summary(user, day))

    def _quiz(self, user, day, rest, args):
        task = self.verification.start(user, day)
        return f"Vocabulary #{task['id']}: {task['prompt']} {explain(task)} Just type your answer."

    def _answer(self, user, day, rest, args):
        if len(args) < 2:
            raise ValueError("Usage: /answer <id> <answer>")
        result = self.verification.answer(user, day, number(args[0]), rest.split(maxsplit=1)[1])
        return f"Learning verification: {result['state']}. {result['message']}"

    def _recover(self, user, day, rest, args):
        if not rest:
            raise ValueError("Usage: /recover <reason>")
        proposal = self.recovery.propose(user, day, rest)
        return f"Proposal #{proposal['id']}: {proposal['explanation']} Reply with /adjust {proposal['id']} approve."

    def _adjust(self, user, day, rest, args):
        if len(args) != 2:
            raise ValueError("Usage: /adjust <id> approve|reject|cancel")
        return f"Proposal #{args[0]}: {self.recovery.resolve(user, day, number(args[0]), args[1].lower())}."

    def _focus(self, user, day, rest, args):
        return focus.format_focus(focus.snapshot(self.store, self.verification, user))

    def _policy(self, user, day, rest, args):
        return checkin_policy.format_preview(checkin_policy.preview(self.store, self.verification, user))

    def _recall(self, user, day, rest, args):
        return format_recall_report(self.verification.recall_report(user),
                                    confusion.format_pairs(confusion.pairs(self.store, user)))

    def _history(self, user, day, rest, args):
        events = self.memory.history(user, limit=10)
        return "\n".join(f"{e['created_at']}: {e['kind']} ({e['source']})" for e in events) or "No history yet."


def number(text):
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"Expected a whole number, got {text!r}") from None


async def repl(chat, import_path=None):
    print("Personal AI Mentor (local chat). Type /help for examples, /quit to exit.")
    if import_path:
        print("bot> " + (await chat.import_file(import_path) or "").replace("\n", "\n     "))
    while True:
        try:
            line = await asyncio.to_thread(input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.strip().lower() in {"/quit", "/exit"}:
            return
        reply = await chat.handle(line)
        if reply:
            print("bot> " + reply.replace("\n", "\n     "))


def main():
    load_dotenv()
    logging.basicConfig(level=logging.ERROR)  # Keep routine diagnostics out of the chat.
    config = Config.from_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=config.database_path,
                        help="SQLite file (default: DATABASE_PATH, shared with the Discord bot)")
    parser.add_argument("--import", dest="import_path", metavar="PATH",
                        help="Extract vocabulary from a notes file or photo, then confirm it in the chat")
    parser.add_argument("--user", type=int, default=config.allowed_user_id or LOCAL_USER_ID,
                        help="User ID to act as (default: ALLOWED_USER_ID, so you see your Discord data)")
    args = parser.parse_args()
    if not 0 < args.user < 2**63:
        parser.error("--user must be a positive integer")
    try:
        asyncio.run(repl(LocalChat(config, GoalStore(args.db), args.user), args.import_path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
