# Personal Accountability AI

[English](README.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md)

A working, local-processing-first Discord assistant for the 2026 SOL-AIM competition (deliverables due December 2, 2026). It tracks one recurring daily goal for one configured Discord user, sends opt-in KST check-ins, distinguishes reported minutes from quiz results, and requests approval for recovery adjustments.

Personal state is stored in SQLite. Discord messages travel through Discord; the system is not fully offline. Local Ollama advice is optional and disabled by default. The original `/setgoal`, `/report`, and `/summary` commands and existing database rows are retained.

## Install and run

Requires Python 3.11.9+ and [uv](https://docs.astral.sh/uv/). In PowerShell, from the repository root:

```powershell
uv sync --locked
Copy-Item .env.example .env
# Edit .env locally, then:
uv run python bot.py
```

Do not overwrite an existing `.env`. The bot validates configuration before connecting. Importing `bot` does not create a database, read `.env`, or connect to Discord. Run one bot process per database.

If the default uv cache is restricted, use a workspace-local cache:

```powershell
uv --cache-dir .uv-cache sync --locked
uv --cache-dir .uv-cache run python bot.py
```

## Configuration

| Variable | Required | Default / meaning |
|---|---|---|
| `DISCORD_TOKEN` | Yes | Bot token; keep secret |
| `ALLOWED_USER_ID` | Yes | Positive numeric Discord user ID |
| `DISCORD_GUILD_ID` | No | `0`; optional development server ID |
| `DATABASE_PATH` | No | `data/accountability.db`, relative to working directory |
| `MIDDAY_CHECKIN` | No | `12:00`, Asia/Seoul, 24-hour HH:MM |
| `EVENING_REVIEW` | No | `21:00`, Asia/Seoul |
| `OLLAMA_ENABLED` | No | `false`; must be `true` or `false` |
| `OLLAMA_URL` | No | `http://127.0.0.1:11434`; local HTTP only |
| `OLLAMA_MODEL` | No | `qwen3:0.6b`; install the chosen local model separately |
| `OLLAMA_TIMEOUT` | No | `10` seconds; allowed range 0.1 to 60 |

Legacy `DISCORD_BOT_TOKEN` and `GUILD_ID` aliases are accepted when the canonical variable is absent. Environment schedule defaults apply when enabling check-ins; persisted user choices remain authoritative after restart. Modern KST uses the `Asia/Seoul` zone, with UTC+09:00 fallback on Windows systems without a timezone database.

## Discord setup

1. Create an application and bot in the Discord Developer Portal. Copy its bot token into `.env` locally.
2. Enable Developer Mode in Discord and copy your user ID into `ALLOWED_USER_ID`. Optionally copy a development server ID into `DISCORD_GUILD_ID`.
3. Install the bot into your server with the `bot` and `applications.commands` scopes. Allow application commands and bot DMs; an administrator permission is not required.
4. Start the bot. It synchronizes global commands for server and DM interactions, plus optional development-guild commands. Global command availability can take time to propagate.
5. Every command checks the configured user ID. Enable check-ins explicitly. If Discord blocks DMs, `/checkins action:status` shows `blocked`; permit DMs before the next scheduled slot.

Open a one-to-one DM with PM and type “how am I doing?” or “today's total is 20 minutes”. Only the configured user's private text messages reach the interpreter. Slash replies remain ephemeral. DM conversations use the direct-messages gateway intent. Discord supplies text in DMs with the app without the privileged Message Content intent. [Discord gateway documentation](https://docs.discord.com/developers/events/gateway#message-content-intent)

For the three-language SOL-AIM demo, draft confirmations, one-off moves to tomorrow and offline operation, see the [manual DM live-test guide](docs/DISCORD_DM_DEMO.md). The [synthetic evaluation](experiments/conversation/README.md) includes real local Ollama and unavailable-backend runs.

## Commands and demonstration

Use the following with a synthetic Korean learning goal. Slash-command option names are shown as Discord displays them.

1. `/setgoal title:Korean minutes:30`
2. `/checkins action:enable midday:12:00 evening:21:00` (KST). For a quick demo, choose a midday time a few minutes ahead and an evening time after it. Keep the bot running and wait for the DM.
3. `/report minutes:30` records today's **total**, replacing the current total while retaining each previous report event. Reports are not added together.
4. `/quiz` offers a predefined Korean vocabulary question. Follow its `/answer attempt:<id> answer:<answer>` instruction. The demo asks the English meaning of 물: `water`.
5. Try a wrong answer first to see `review_required`, then `/quiz` and the correct answer to see `passed_after_support`. A correct first answer gives `verified`. The state applies only to the tested vocabulary and current report revision; it never independently proves 30 minutes of study. A new report begins with `reported` verification state.
6. `/recover reason:Unexpected work shift` proposes halving today's target. It makes no target change yet.
7. `/adjust proposal:<id> decision:approve` applies that proposal for today. `reject` and `cancel` preserve the target. The original 30-minute recurring goal and original daily target remain recorded. Tomorrow returns to the recurring target.
8. `/summary` shows original and effective target, reported minutes, quiz result, and obstacles. The evening DM uses the same review.
9. Stop and restart the bot, then use `/summary`, `/history`, and `/checkins action:status`. State survives and an already claimed daily check-in slot is not resent.
10. `/checkins action:pause`, `resume`, or `disable` controls reminders. `/ask text:<question>` requests optional local advice. `/ask ... non_urgent:true` records a DEFER routing decision without calling the model; bring the question back manually when ready.

## Architecture and persistence

| Module | Responsibility |
|---|---|
| `core.py` | Existing GoalStore API, daily plan snapshots, report revisions, daily review |
| `memory.py` | Transactional SQLite migrations, append-only events, filtered history |
| `bot.py` | Authorized slash commands, private replies, DM adapter and background loop |
| `dm_intents.py`, `conversation.py` | Whole-message parsing, structured proposal validation, SQLite-backed confirmations and idempotent dispatch |
| `scheduler.py` | Opt-in configuration, persistent daily slots, restart recovery and delivery accounting |
| `verification.py` | Predefined vocabulary questions and immutable attempt history |
| `recovery.py` | Deterministic daily proposals and atomic approval/rejection/cancellation |
| `router.py` | Replaceable Router protocol, RULE/CLARIFY/LLM/DEFER rules and decision audit |
| `llm_adapter.py` | Bounded local HTTP inference, strict JSON advice validation and fallback |
| `config.py`, `time_utils.py` | Validated environment settings, dates, minutes and KST times |
| `experiments/router/` | Isolated synthetic dataset, trained baseline and evaluation pipeline |

SQLite `PRAGMA user_version` migrations run transactionally and add tables without deleting existing records. Original `goals` and `reports` tables remain compatible. The migration backfills report revisions and day snapshots from legacy rows. Legacy report timestamps are marked unknown; past goal changes that were never stored cannot be reconstructed, so migrated targets use the existing goal value. Back up the database with the bot stopped before upgrading; keep the `.db` file private.

`summary()['status']` retains its original activity meaning (`not_reported` or `reported`); `verification_state` separately exposes all five verification states. Quiz attempts remain in the database after newer reports or attempts. Recovery approvals reject stale proposals after the plan changes and require an explicit same-day user interaction. `Memory(store).history(user_id, day=None, kind=None, limit=20)` returns relevant events, newest first, with evidence source and timestamp.

## Scheduler reliability contract

- Polls every 30 seconds after Discord is ready. No opt-in means no scheduled DM.
- Unique `(user_id, KST day, slot)` claims prevent duplicate daily slots, including concurrent ticks and restarts.
- Restart catches up today's eligible slot within six hours of its scheduled time. Older days and slots before opt-in, reconfiguration, or resume are skipped. There is no reminder backlog.
- Claims are committed before Discord delivery. A process crash can leave delivery uncertain; stale claims become `unknown` after two minutes and are not automatically resent. This favors avoiding duplicate DMs over guaranteeing every DM arrives. Exactly-once delivery across Discord and SQLite is not guaranteed.
- A definite, retryable transport failure has at most three attempts, spaced at least five minutes apart. Forbidden or missing recipients become `blocked`. Discord HTTP errors, timeouts and other uncertain outcomes become `unknown` without automatic retry.
- Pause/disable prevents new claims. A send already in progress may finish. Changing times does not reopen an already claimed slot that day. Re-enable/resume applies to future scheduled times.
- Missed replies and failed deliveries never classify the user as procrastinating, dishonest, or failing a goal.

## Optional local Ollama advice

Install Ollama and the chosen local model independently, set `OLLAMA_ENABLED=true`, then use `/ask`. The adapter uses [`POST /api/generate`](https://docs.ollama.com/api/generate) with streaming disabled and an explicit JSON schema. Requests have a total timeout, response size bound, disabled redirects and disabled environment proxies. Only loopback endpoints are accepted.

Output must contain exactly an allowlisted action (`SUGGEST` or `ASK_FOLLOWUP`) and a short message. Advice is displayed as a model suggestion, with no executor or ability to change goals, schedules, privacy settings, files, or commands. JSON validation does not guarantee factual or suitable advice. Failure, timeout, invalid JSON, or a disabled backend produces a deterministic follow-up. Basic tracking and scheduling never require Ollama. Advice text and raw prompts are not retained; invocation outcomes and latency are.

## Tests and development

```powershell
uv run python -m pytest -q
uv run python check.py
uv run python -m compileall -q bot.py core.py config.py scheduler.py router.py verification.py recovery.py memory.py llm_adapter.py time_utils.py
# Include the model round-trip/evaluation test:
uv run --extra research python -m pytest -q
```

Tests use temporary synthetic SQLite databases and mock Discord/Ollama. No token, live bot, local model, or external service is required. Without the research extra, the model evaluation test skips. Existing three core tests remain included. Coverage includes migration, persistence, invalid inputs, all command authorization checks, scheduling retries/deduplication/restart/pause, quiz transitions, stale proposal protection, and the end-to-end command workflow.

## Offline model experiment

```powershell
uv sync --locked --extra research
uv run --extra research python -m experiments.router.run --seed 42
# Optional output directory and dataset size:
uv run --extra research python -m experiments.router.run --seed 42 --samples-per-family 20 --output experiments/router/artifacts
```

The run needs no network after dependencies are installed. It writes synthetic train/validation/test JSONL, `baseline.joblib`, and `metrics.json` containing per-class precision/recall/F1, confusion matrices, median/p95 single-input latency, artifact size, training time, and limited Python allocation measurements. Only load your own trusted joblib artifacts.

The default split is 240/80/160 examples from disjoint scenario-template families. Hyperparameter selection uses validation data only. Both routers receive the same text and explicit metadata; family identifiers and labels are excluded from model features. See [experiment protocol and results](experiments/router/README.md). The measured baseline is not installed into the running bot.

## Completed, experimental, and future work

Completed MVP: original commands, durable reports, KST opt-in DM check-ins, pause/resume, daily reviews, vocabulary verification, approval-controlled daily recovery, history, deterministic routing, and optional local advice with fallback.

Completed research infrastructure: synthetic generation, family-separated splits, TF-IDF/logistic-regression training, artifact round-trip, classification metrics, confusion matrices and rule comparison. Synthetic results evaluate this routing dataset only, not accountability effectiveness with people.

Future research: more varied reviewed routing scenarios and metadata features; temporal simulator and sequence contracts before LSTM or Mamba comparisons; only then consider distillation or critics with justified targets. No LSTM, Mamba, critic or distillation model has been trained or evaluated. Embeddings, vector search, cloud inference, iOS monitoring, generic priority/deferred task queues, and public deployment are outside this MVP. DEFER currently recommends manual postponement, with no deferred inference worker.

Privacy limits: SQLite is unencrypted and personal data has no automatic retention/deletion UI. Local filesystem access can read it. Discord holds externally transmitted messages. One-user authorization is enforced in the integration layer; the domain library is a trusted local API. No private conversations are automatically collected for training. Logs exclude message bodies, private records and tokens. Secrets, databases, caches and research artifacts are ignored by Git. This assistant does not diagnose health conditions or infer intent from response timing.

The prior architectural proposal is preserved in [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md); it describes future ideas, not implemented capabilities. Implementation evidence is recorded in [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).
