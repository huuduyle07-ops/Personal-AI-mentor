"""Discord slash-command integration. Importing this module performs no I/O."""
import asyncio
import logging
from typing import Literal

import discord
from discord import app_commands
from dotenv import load_dotenv

from config import Config
from core import GoalStore, daily_review
from llm_adapter import OllamaAdapter
from memory import Memory, record
from recovery import Recovery
from router import AuditedRouter, Route, RoutingInput
from scheduler import CheckinScheduler, DeliveryError
from time_utils import now_kst
from verification import Verification

log = logging.getLogger(__name__)


def today():
    return now_kst().date().isoformat()


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
        super().__init__(intents=discord.Intents(guilds=True), allowed_mentions=discord.AllowedMentions.none())
        self.config, self.store = config, store
        self.tree = PrivateTree(self)
        self.scheduler = CheckinScheduler(store, config.allowed_user_id, self.send_checkin)
        self.routing = AuditedRouter(store)
        self.verification, self.recovery = Verification(store), Recovery(store)
        self.memory, self.inference = Memory(store), OllamaAdapter(config)
        self.scheduler_task = None
        register_commands(self)

    async def setup_hook(self):
        # Global commands make slash interactions available in bot DMs too.
        await self.tree.sync()
        if self.config.guild_id:
            guild = discord.Object(id=self.config.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        self.scheduler_task = asyncio.create_task(self.checkin_loop())

    async def send_checkin(self, user_id, message):
        if user_id != self.config.allowed_user_id:
            raise DeliveryError("forbidden")
        try:
            user = self.get_user(user_id) or await self.fetch_user(user_id)
            await user.send(discord.utils.escape_markdown(message), allowed_mentions=discord.AllowedMentions.none())
        except discord.Forbidden:
            raise DeliveryError("forbidden") from None
        except discord.NotFound:
            raise DeliveryError("not_found") from None
        except discord.HTTPException:
            raise DeliveryError("delivery_failed", uncertain=True) from None

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

    async def guard(interaction, command, minutes=None):
        if not await authorized(interaction, bot.config.allowed_user_id):
            return False
        bot.routing.route(interaction.user.id, RoutingInput(command=command, minutes=minutes))
        return True

    async def reply(interaction, message):
        await interaction.response.send_message(discord.utils.escape_markdown(message), ephemeral=True,
                                                allowed_mentions=discord.AllowedMentions.none())

    @bot.tree.command(name="setgoal", description="Set your daily learning goal")
    async def setgoal(interaction: discord.Interaction, title: str, minutes: app_commands.Range[int, 1, 600]):
        if not await guard(interaction, "setgoal"):
            return
        store.set_goal(interaction.user.id, title, minutes)
        await reply(interaction, f"Goal saved: {title.strip()} — {minutes} min/day")

    @bot.tree.command(name="report", description="Report today's total study minutes")
    async def report(interaction: discord.Interaction, minutes: app_commands.Range[int, 0, 1440]):
        if not await guard(interaction, "report", minutes):
            return
        store.report(interaction.user.id, today(), minutes)
        await reply(interaction, f"Reported {minutes} minutes. Learning outcome is not yet verified. "
                    "Try the optional Korean vocabulary demo with /quiz. It does not verify study duration.")

    @bot.tree.command(name="summary", description="View today's progress and daily review")
    async def summary(interaction: discord.Interaction):
        if await guard(interaction, "summary"):
            await reply(interaction, daily_review(store.summary(interaction.user.id, today())))

    @bot.tree.command(name="checkins", description="Opt in, pause, resume, or view KST check-ins")
    async def checkins(interaction: discord.Interaction,
                       action: Literal["enable", "disable", "pause", "resume", "status"],
                       midday: str | None = None, evening: str | None = None):
        if not await guard(interaction, "checkins"):
            return
        user_id = interaction.user.id
        if action in {"enable", "disable"}:
            bot.scheduler.configure(user_id, enabled=action == "enable", midday=midday or bot.config.midday,
                                    evening=evening or bot.config.evening)
        elif action in {"pause", "resume"}:
            bot.scheduler.pause(user_id, action == "pause")
        status = bot.scheduler.status(user_id)
        settings = status["settings"]
        message = "Check-ins are off. Use /checkins action:enable to opt in."
        if settings:
            message = (f"Check-ins: enabled={bool(settings['enabled'])}, paused={bool(settings['paused'])}. "
                       f"Asia/Seoul: midday {settings['midday']}, evening {settings['evening']}. "
                       "Times already passed are skipped when enabling or resuming.")
        for delivery in status["deliveries"]:
            message += f"\n{delivery['day']} {delivery['slot']}: {delivery['status']} (attempts {delivery['attempts']})"
        await reply(interaction, message)

    @bot.tree.command(name="quiz", description="Start or resume the optional Korean vocabulary demo")
    async def quiz(interaction: discord.Interaction):
        if not await guard(interaction, "quiz"):
            return
        task = bot.verification.start(interaction.user.id, today())
        await reply(interaction, f"Vocabulary demo #{task['id']}: {task['prompt']} "
                    f"Use /answer attempt:{task['id']} answer:<your answer>. "
                    "This tests only the question, not study duration or the whole goal.")

    @bot.tree.command(name="answer", description="Answer your pending vocabulary quiz")
    async def answer(interaction: discord.Interaction, attempt: int, answer: str):
        if not await guard(interaction, "answer"):
            return
        result = bot.verification.answer(interaction.user.id, today(), attempt, answer)
        await reply(interaction, f"Learning verification: {result['state']}. {result['message']} "
                    "Study duration remains self-reported.")

    @bot.tree.command(name="recover", description="Describe an obstacle and request a smaller target for today")
    async def recover(interaction: discord.Interaction, reason: str):
        if not await guard(interaction, "recover"):
            return
        proposal = bot.recovery.propose(interaction.user.id, today(), reason)
        await reply(interaction, f"Proposal #{proposal['id']}: {proposal['explanation']} "
                    f"Use /adjust proposal:{proposal['id']} decision:approve, reject, or cancel.")

    @bot.tree.command(name="adjust", description="Approve, reject, or cancel your recovery proposal")
    async def adjust(interaction: discord.Interaction, proposal: int, decision: Literal["approve", "reject", "cancel"]):
        if not await guard(interaction, "adjust"):
            return
        status = bot.recovery.resolve(interaction.user.id, today(), proposal, decision)
        await reply(interaction, f"Proposal #{proposal}: {status}. Use /summary to review today's plan.")

    @bot.tree.command(name="history", description="View recent event types and evidence sources")
    async def history(interaction: discord.Interaction):
        if not await guard(interaction, "history"):
            return
        events = bot.memory.history(interaction.user.id, limit=10)
        await reply(interaction, "\n".join(f"{e['created_at']}: {e['kind']} ({e['source']})" for e in events) or "No history yet.")

    @bot.tree.command(name="ask", description="Request optional local AI advice; it cannot change your settings")
    async def ask(interaction: discord.Interaction, text: str, non_urgent: bool = False):
        if not await authorized(interaction, bot.config.allowed_user_id):
            return
        decision = bot.routing.route(interaction.user.id, RoutingInput(text=text, non_urgent=non_urgent))
        if decision.route == Route.CLARIFY:
            await reply(interaction, "Please describe the obstacle or question in a little more detail.")
            return
        if decision.route == Route.DEFER:
            await reply(interaction, "No inference was started. Bring this question back when you want to discuss it; "
                        "this MVP does not automatically process deferred advice.")
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.inference.suggest(text)
        with store.connect() as db:
            record(db, interaction.user.id, today(), "inference_result",
                   {"fallback": result.fallback, "reason": result.reason, "latency_ms": result.latency_ms},
                   "system" if result.fallback else "model_interpretation")
        prefix = "Assistant fallback: " if result.fallback else "Local model suggestion (no changes applied): "
        await interaction.followup.send(prefix + discord.utils.escape_markdown(result.message), ephemeral=True,
                                        allowed_mentions=discord.AllowedMentions.none())

    @bot.tree.error
    async def on_error(interaction, error):
        original = getattr(error, "original", error)
        message = str(original) if isinstance(original, (ValueError, PermissionError)) else "The operation failed. Try again."
        if not isinstance(original, (ValueError, PermissionError)):
            log.warning("Slash command failed (%s)", type(original).__name__)
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
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
