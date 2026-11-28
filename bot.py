"""Discord slash-command integration. Importing this module performs no I/O."""
import asyncio
import hashlib
import json
import logging
import sqlite3
from typing import Literal

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv

from config import Config
from conversation import Conversation
from dm_replies import AdjustmentPrompt
from core import GoalStore, daily_review
from llm_adapter import OllamaAdapter
from memory import Memory, record
from recovery import Recovery
from router import AuditedRouter, Route, RoutingInput
from scheduler import CheckinScheduler, DeliveryError
from study_notes import MAX_IMAGE_BYTES, NoteImporter
from time_utils import KST, now_kst
import checkin_policy
import confusion
import focus
from verification import Verification, explain, format_recall_report

log = logging.getLogger(__name__)

DISCORD_LIMIT = 2000
COMMAND_SYNC_KEY = "command_sync_fingerprint"


def today():
    return now_kst().date().isoformat()


def discord_text(text, limit=DISCORD_LIMIT):
    """Escape markdown and fit Discord's message limit without leaving a dangling escape."""
    escaped = discord.utils.escape_markdown(text)
    if len(escaped) <= limit:
        return escaped
    cut = escaped[:limit - 1]
    if (len(cut) - len(cut.rstrip("\\"))) % 2:
        cut = cut[:-1]
    return cut + "…"


async def authorized(interaction, allowed_user_id):
    if not allowed_user_id or interaction.user.id != allowed_user_id:
        await interaction.response.send_message("This is a private assistant.", ephemeral=True)
        return False
    return True


class PrivateTree(app_commands.CommandTree):
    async def interaction_check(self, interaction):
        return await authorized(interaction, self.client.config.allowed_user_id)


class PersonalBot(discord.Client):
    def __init__(self, config, store):
        super().__init__(intents=discord.Intents(guilds=True, dm_messages=True), max_messages=None,
                         allowed_mentions=discord.AllowedMentions.none())
        self.config, self.store = config, store
        self.tree = PrivateTree(self)
        self.routing = AuditedRouter(store)
        self.verification, self.recovery = Verification(store), Recovery(store)
        self.scheduler = CheckinScheduler(store, config.allowed_user_id, self.send_checkin,
                                          shadow=checkin_policy.ShadowLearner(store, self.verification))
        self.memory, self.inference = Memory(store), OllamaAdapter(config)
        self.conversation = Conversation(store, config.allowed_user_id, self.inference,
                                         self.recovery, self.verification, today, NoteImporter(config))
        self.scheduler_task = None
        register_commands(self)

    async def on_message(self, message):
        if (message.author.bot or message.author.id != self.config.allowed_user_id
                or message.guild is not None or message.channel.type != discord.ChannelType.private
                or message.type not in {discord.MessageType.default, discord.MessageType.reply}
                or not (message.content.strip() or message.attachments)):
            return
        day = message.created_at.astimezone(KST).date().isoformat()
        reply_to = message.reference.message_id if message.type == discord.MessageType.reply and message.reference else None
        try:
            if message.attachments:
                # One file per message: study notes to extract vocabulary from. Nothing is saved until confirmed.
                attachment = message.attachments[0]
                if attachment.size > MAX_IMAGE_BYTES:
                    reply = "That file is too large; send notes under 8 MB."
                else:
                    reply = await self.conversation.import_notes(message.author.id, message.id, attachment.filename,
                                                                 await attachment.read(), day)
            else:
                reply = await self.conversation.receive(message.author.id, message.id, message.content, day,
                                                        reply_to=reply_to)
        except Exception as exc:
            log.warning("DM processing failed (%s)", type(exc).__name__)
            reply = "I couldn't process that request. Check today's progress before trying again."
        if reply:
            try:
                body = reply.text if isinstance(reply, AdjustmentPrompt) else reply
                sent = await message.channel.send(discord_text(body), allowed_mentions=discord.AllowedMentions.none())
                if isinstance(reply, AdjustmentPrompt):
                    # Bind consent to a Discord-issued reply ID only after confirmed delivery.
                    # The numbered approval path also works after an uncertain delivery.
                    await asyncio.to_thread(self._bind_prompt, message.author.id, day, reply.proposal_id, sent.id)
            except discord.HTTPException as exc:
                log.warning("DM reply delivery failed (%s)", type(exc).__name__)
            except sqlite3.Error:
                log.warning("DM approval prompt binding failed; numbered approval is available")

    def _bind_prompt(self, user_id, day, proposal_id, message_id):
        with self.store.connect() as db:
            record(db, user_id, day, "dm_approval_prompt",
                   {"proposal_id": proposal_id, "message_id": message_id}, "system")

    def command_fingerprint(self):
        commands = [command.to_dict(self.tree) for command in self.tree.get_commands()]
        payload = json.dumps({"guild_id": self.config.guild_id, "commands": commands}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def setup_hook(self):
        # Syncing is rate limited by Discord, so only sync when the command definitions changed.
        fingerprint = self.command_fingerprint()
        if await asyncio.to_thread(self.store.get_state, COMMAND_SYNC_KEY) != fingerprint:
            # Global commands work in servers and in bot DMs.
            await self.tree.sync()
            if self.config.guild_id:
                # Global commands already appear in the server. Clear guild copies left by older
                # versions, which would otherwise show every command twice.
                guild = discord.Object(id=self.config.guild_id)
                self.tree.clear_commands(guild=guild)
                await self.tree.sync(guild=guild)
            await asyncio.to_thread(self.store.set_state, COMMAND_SYNC_KEY, fingerprint)
        self.scheduler_task = asyncio.create_task(self.checkin_loop())

    async def send_checkin(self, user_id, message):
        if user_id != self.config.allowed_user_id:
            raise DeliveryError("forbidden")
        try:
            user = self.get_user(user_id) or await self.fetch_user(user_id)
            await user.send(discord_text(message), allowed_mentions=discord.AllowedMentions.none())
        except discord.Forbidden:
            raise DeliveryError("forbidden") from None
        except discord.NotFound:
            raise DeliveryError("not_found") from None
        except discord.DiscordServerError:
            # A server error may happen after Discord stored the message.
            raise DeliveryError("delivery_failed", uncertain=True) from None
        except discord.HTTPException:
            # Discord rejected the request (4xx): nothing was sent, and resending it would fail the same way.
            raise DeliveryError("delivery_failed") from None
        except aiohttp.ClientConnectorError:
            # No connection was made, so nothing was sent and a later retry is safe.
            raise DeliveryError("unavailable", retryable=True) from None

    async def checkin_loop(self):
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                await self.scheduler.tick()
            except Exception:
                log.warning("Check-in tick failed; retrying next tick")
            await asyncio.sleep(30)

    async def close(self):
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        await super().close()


def register_commands(bot):
    store = bot.store
    run = asyncio.to_thread  # SQLite work runs off the event loop.

    async def guard(interaction, command, minutes=None):
        if not await authorized(interaction, bot.config.allowed_user_id):
            return False
        await run(bot.routing.route, interaction.user.id, RoutingInput(command=command, minutes=minutes))
        return True

    async def reply(interaction, message):
        await interaction.response.send_message(discord_text(message), ephemeral=True,
                                                allowed_mentions=discord.AllowedMentions.none())

    @bot.tree.command(name="setgoal", description="Set your daily learning goal")
    async def setgoal(interaction: discord.Interaction, title: str, minutes: app_commands.Range[int, 1, 600]):
        if not await guard(interaction, "setgoal"):
            return
        await run(store.set_goal, interaction.user.id, title, minutes)
        await reply(interaction, f"Goal saved: {title.strip()} — {minutes} min/day")

    @bot.tree.command(name="report", description="Report today's total study minutes")
    async def report(interaction: discord.Interaction, minutes: app_commands.Range[int, 0, 1440]):
        if not await guard(interaction, "report", minutes):
            return
        if not (await run(store.report, interaction.user.id, today(), minutes)).changed:
            await reply(interaction, f"{minutes} minutes is already recorded as today's total, so nothing changed.")
            return
        await reply(interaction, f"Reported {minutes} minutes. Learning outcome is not yet verified. "
                    "Try the optional Korean vocabulary demo with /quiz. It does not verify study duration.")

    @bot.tree.command(name="summary", description="View today's progress and daily review")
    async def summary(interaction: discord.Interaction):
        if await guard(interaction, "summary"):
            await reply(interaction, daily_review(await run(store.summary, interaction.user.id, today())))

    @bot.tree.command(name="checkins", description="Opt in, pause, resume, or view KST check-ins")
    async def checkins(interaction: discord.Interaction,
                       action: Literal["enable", "disable", "pause", "resume", "status"],
                       midday: str | None = None, evening: str | None = None):
        if not await guard(interaction, "checkins"):
            return
        user_id = interaction.user.id

        def apply():
            if action in {"enable", "disable"}:
                bot.scheduler.configure(user_id, enabled=action == "enable", midday=midday or bot.config.midday,
                                        evening=evening or bot.config.evening)
            elif action in {"pause", "resume"}:
                bot.scheduler.pause(user_id, action == "pause")
            return bot.scheduler.status(user_id)

        status = await run(apply)
        settings = status["settings"]
        message = "Check-ins are off. Use /checkins action:enable to opt in."
        if settings:
            message = (f"Check-ins: enabled={bool(settings['enabled'])}, paused={bool(settings['paused'])}. "
                       f"Asia/Seoul: midday {settings['midday']}, evening {settings['evening']}. "
                       "Times already passed are skipped when enabling or resuming.")
        for delivery in status["deliveries"]:
            message += f"\n{delivery['day']} {delivery['slot']}: {delivery['status']} (attempts {delivery['attempts']})"
        await reply(interaction, message)

    @bot.tree.command(name="quiz", description="Quiz a Korean word: flashcard, inside a sentence, or active (type the Hangul)")
    async def quiz(interaction: discord.Interaction, style: Literal["word", "context", "active"] = "word"):
        if not await guard(interaction, "quiz"):
            return
        if style == "active":
            task, text = await bot.conversation.active_quiz_reply(interaction.user.id, today(), "en")
            if task is not None:
                text += " Reply in our DM with the Hangul."
            await reply(interaction, text)
            return
        if style == "context":
            await interaction.response.defer(ephemeral=True, thinking=True)
            task, text = await bot.conversation.quiz_reply(interaction.user.id, today(), "en", sentence=True)
            if task is not None:
                text += f" Reply in our DM, or use /answer attempt:{task['id']}."
            await interaction.followup.send(discord_text(text), ephemeral=True,
                                            allowed_mentions=discord.AllowedMentions.none())
            return
        task = await run(bot.verification.start, interaction.user.id, today())
        await reply(interaction, f"Vocabulary #{task['id']}: {task['prompt']} {explain(task)} "
                    f"Reply in this DM with the meaning, or use /answer attempt:{task['id']}. "
                    "This tests only the question, not study duration or the whole goal.")

    @bot.tree.command(name="answer", description="Answer your pending vocabulary quiz")
    async def answer(interaction: discord.Interaction, attempt: int, answer: str):
        if not await guard(interaction, "answer"):
            return
        result = await run(bot.verification.answer, interaction.user.id, today(), attempt, answer)
        await reply(interaction, f"Learning verification: {result['state']}. {result['message']} "
                    "Study duration remains self-reported.")

    @bot.tree.command(name="recover", description="Describe an obstacle and request a smaller target for today")
    async def recover(interaction: discord.Interaction, reason: str):
        if not await guard(interaction, "recover"):
            return
        proposal = await run(bot.recovery.propose, interaction.user.id, today(), reason)
        await reply(interaction, f"Proposal #{proposal['id']}: {proposal['explanation']} "
                    f"Use /adjust proposal:{proposal['id']} decision:approve, reject, or cancel.")

    @bot.tree.command(name="adjust", description="Approve, reject, or cancel your recovery proposal")
    async def adjust(interaction: discord.Interaction, proposal: int, decision: Literal["approve", "reject", "cancel"]):
        if not await guard(interaction, "adjust"):
            return
        status = await run(bot.recovery.resolve, interaction.user.id, today(), proposal, decision)
        await reply(interaction, f"Proposal #{proposal}: {status}. Use /summary to review today's plan.")

    @bot.tree.command(name="recall", description="Show which Korean words are due, their predicted recall and why")
    async def recall(interaction: discord.Interaction):
        if not await guard(interaction, "recall"):
            return
        report = await run(bot.verification.recall_report, interaction.user.id)
        pairs = await run(confusion.pairs, store, interaction.user.id)
        await reply(interaction, format_recall_report(report, confusion.format_pairs(pairs)))

    @bot.tree.command(name="focus", description="Today's study status, word queue and check-ins at a glance")
    async def focus_command(interaction: discord.Interaction):
        if not await guard(interaction, "focus"):
            return
        snapshot = await run(focus.snapshot, store, bot.verification, interaction.user.id)
        await reply(interaction, focus.format_focus(snapshot))

    @bot.tree.command(name="policy", description="Preview what the learned check-in policy would do now and why")
    async def policy(interaction: discord.Interaction):
        if not await guard(interaction, "policy"):
            return
        report = await run(checkin_policy.preview, store, bot.verification, interaction.user.id)
        await reply(interaction, checkin_policy.format_preview(report))

    @bot.tree.command(name="history", description="View recent event types and evidence sources")
    async def history(interaction: discord.Interaction):
        if not await guard(interaction, "history"):
            return
        events = await run(bot.memory.history, interaction.user.id, limit=10)
        await reply(interaction, "\n".join(f"{e['created_at']}: {e['kind']} ({e['source']})" for e in events) or "No history yet.")

    @bot.tree.command(name="ask", description="Request optional local AI advice; it cannot change your settings")
    async def ask(interaction: discord.Interaction, text: str, non_urgent: bool = False):
        if not await authorized(interaction, bot.config.allowed_user_id):
            return
        decision = await run(bot.routing.route, interaction.user.id, RoutingInput(text=text, non_urgent=non_urgent))
        if decision.route == Route.CLARIFY:
            await reply(interaction, "Please describe the obstacle or question in a little more detail.")
            return
        if decision.route == Route.DEFER:
            await reply(interaction, "No inference was started. Bring this question back when you want to discuss it; "
                        "this MVP does not automatically process deferred advice.")
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.inference.suggest(text)

        def save():
            with store.connect() as db:
                record(db, interaction.user.id, today(), "inference_result",
                       {"fallback": result.fallback, "reason": result.reason, "latency_ms": result.latency_ms},
                       "system" if result.fallback else "model_interpretation")

        await run(save)
        prefix = "Assistant fallback: " if result.fallback else "Local model suggestion (no changes applied): "
        await interaction.followup.send(discord_text(prefix + result.message), ephemeral=True,
                                        allowed_mentions=discord.AllowedMentions.none())

    @bot.tree.error
    async def on_error(interaction, error):
        original = getattr(error, "original", error)
        message = str(original) if isinstance(original, (ValueError, PermissionError)) else "The operation failed. Try again."
        if not isinstance(original, (ValueError, PermissionError)):
            log.warning("Slash command failed (%s)", type(original).__name__)
        if interaction.response.is_done():
            await interaction.followup.send(discord_text(message), ephemeral=True,
                                            allowed_mentions=discord.AllowedMentions.none())
        else:
            await reply(interaction, message)


def create_bot(config=None, store=None):
    config = config or Config.from_env()
    config.require_discord()
    return PersonalBot(config, store or GoalStore(config.database_path))


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    bot = create_bot(config)
    bot.run(config.token, log_handler=None)


if __name__ == "__main__":
    main()
