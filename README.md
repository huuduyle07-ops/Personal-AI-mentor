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
| `DISCORD_GUILD_ID` | No | `0`; optional development server whose duplicate server-specific commands are cleared |
| `DATABASE_PATH` | No | `data/accountability.db`, relative to working directory |
| `MIDDAY_CHECKIN` | No | `12:00`, Asia/Seoul, 24-hour HH:MM |
| `EVENING_REVIEW` | No | `21:00`, Asia/Seoul |
| `OLLAMA_ENABLED` | No | `false`; must be `true` or `false` |
| `OLLAMA_URL` | No | `http://127.0.0.1:11434`; local HTTP only |
| `OLLAMA_MODEL` | No | `qwen3:0.6b`; install the chosen local model separately |
| `OLLAMA_TIMEOUT` | No | `10` seconds; allowed range 0.1 to 60 |

Legacy `DISCORD_BOT_TOKEN` and `GUILD_ID` aliases are accepted when the canonical variable is absent or blank. Blank values, as in a fresh copy of `.env.example`, use the defaults. Environment schedule defaults apply when enabling check-ins; persisted user choices remain authoritative after restart. Modern KST uses the `Asia/Seoul` zone, with UTC+09:00 fallback on Windows systems without a timezone database.

## Local chat (no Discord)

Try the assistant in your terminal with the same conversation logic and database as the Discord bot. No Discord token is needed:

```bash
uv run python chat.py
```

Plain messages work like DMs (“today's total is 20 minutes”, “how am I doing?”, “quiz me”), and `/setgoal`, `/report`, `/summary`, `/quiz`, `/answer`, `/recover`, `/adjust` and `/history` mirror the slash commands. Type `/help` for examples and `/quit` to leave. By default it acts as `ALLOWED_USER_ID` on `DATABASE_PATH`, so you see the same goals and reports as in Discord. Use `--db data/scratch.db` for a separate practice database, or `--user <id>` to act as another local user. Check-ins are only delivered through Discord.

## Discord setup

1. Create an application and bot in the Discord Developer Portal. Copy its bot token into `.env` locally.
2. Enable Developer Mode in Discord and copy your user ID into `ALLOWED_USER_ID`. Optionally copy a development server ID into `DISCORD_GUILD_ID`.
3. Install the bot into your server with the `bot` and `applications.commands` scopes. Allow application commands and bot DMs; an administrator permission is not required.
4. Start the bot. It synchronizes global commands for server and DM interactions, only when the command definitions changed since the last sync. If `DISCORD_GUILD_ID` is set, server-specific copies of the commands are cleared there so each command appears once. Global command availability can take time to propagate.
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
- Restart catches up an eligible slot within six hours of its scheduled time, even past midnight (a 23:30 evening slot can still go out at 01:00 and counts for the day it was scheduled). Older slots and slots before opt-in, reconfiguration, or resume are skipped. There is no reminder backlog.
- Claims are committed before Discord delivery. A process crash can leave delivery uncertain; stale claims become `unknown` after two minutes and are not automatically resent. This favors avoiding duplicate DMs over guaranteeing every DM arrives. Exactly-once delivery across Discord and SQLite is not guaranteed.
- A failure to connect to Discord (nothing was sent) is retried, with at most three attempts spaced at least five minutes apart. Forbidden or missing recipients and other rejected requests (4xx) become `blocked`. Discord server errors, timeouts and other uncertain outcomes become `unknown` without automatic retry. Messages longer than Discord's 2,000-character limit are shortened before sending.
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

## Machine learning features and benchmarks

- **Forgetting-curve vocabulary quiz.** `/quiz` and “quiz me” draw from a 40-word Korean deck ([vocab/korean_basic.csv](vocab/korean_basic.csv)). A logistic recall model trained on 1 million Duolingo review records predicts how likely you are to remember each word you've seen; the quiz reviews the weakest word once its predicted recall drops below 90%, otherwise it introduces a new one. The reply says why each word was chosen. Training, baselines (half-life regression, Leitner, Pimsleur, constant, and the course algorithms KNN, SVM, decision tree and random forest), held-out-learner evaluation and why half-life regression was rejected: [experiments/forgetting_curve/REPORT.md](experiments/forgetting_curve/REPORT.md). The exported weights live in [models/recall_model.json](models/recall_model.json) (CC BY-NC 4.0 derived data, noncommercial use).
- **Personalized recall.** After 5 genuine reviews, the recall model is fitted to your own answers with ridge shrinkage toward the Duolingo prior (`personalize.py`), and `/recall` shows your due words, predicted recall, half-life and why each is scheduled. Population vs personalized comparison (logistic and HLR, synthetic multi-learner benchmark plus your own history): [experiments/personalization/REPORT.md](experiments/personalization/REPORT.md).
- **Import vocabulary from your notes.** Send a `.txt`, `.md` or `.csv` file, or a `.png`/`.jpg` photo, in the bot DM (or run `uv run python chat.py --import notes.md`, or `/import <path>` in the chat). The bot lists the Korean–English pairs it found; reply `yes` to add them, `drop 2 5` to remove some first, or `no` to cancel. Nothing is saved before you confirm. Imported words become the next new words in `/quiz` and `/recall`. Text files use fixed parsing rules; photos need a local vision model (`ollama pull qwen2.5vl:3b`, set with `OLLAMA_VISION_MODEL`). Extraction precision/recall: [experiments/notes_extraction/README.md](experiments/notes_extraction/README.md).
- **Sentence quizzes.** Say “quiz me with a sentence” (or 문장으로 퀴즈 내줘, `/quiz style:context` in Discord, `/quiz context` in the terminal chat). The recall scheduler picks the word as usual (read-only preview), `qwen3:1.7b` writes a short Korean sentence using it, and you answer the word's meaning, graded and logged exactly like a flashcard (`quiz_offered` records `style` and the model). The output must be Hangul-only, contain the word or its stem, and come with an English translation that contains the word's meaning; otherwise you get the plain flashcard. The sentence and translation are never stored; the translation is shown after you answer. About 1 in 12 test sentences was still unnatural Korean, so the prompt notes that examples are AI-written.
- **Web dashboard.** A local Streamlit page with a *Chat* tab (talk to the bot exactly as in the DM or terminal chat: same conversation code, drafts, confirmations and database, plus shortcut buttons; attach a notes photo or text file with the + button to import its words, confirmed with “yes”; or press the mic and speak. Speech is transcribed on your computer by Whisper through [`speech.py`](speech.py) (faster-whisper, CPU int8, any of English/Korean/Vietnamese). The model, `small` by default (about 480 MB, set with `WHISPER_MODEL`), is downloaded once when you press “Download the speech model” and runs offline afterwards. A “🎤 I'm speaking” switch picks Auto, English, 한국어 or Tiếng Việt. In Auto, clearly English speech is transcribed directly; anything less certain is transcribed as English and as Korean (and Vietnamese when likely), and the transcript Whisper is most confident in wins, skipping ones in the wrong script or stuck in a repetition loop. With macOS test voices, all English and Korean sentences and all active-quiz answers were recognised correctly in 2–5 s; a single Korean word said outside a quiz can still be misheard, so pick 한국어 for those) and four more tabs: *Overview & Focus* (today's minutes, streak, due words, personalization progress, shadow-mode windows, last 7 days), *Memory & Confusion Pairs* (predicted recall per word, due/resting status, population vs personalized weights, mix-ups), *Check-in Policy* (the 10 context features, predicted reward and exploration bonus for each action, shadow-mode history) and *Practice & Note Import* (flashcard and active quizzes, plus extracting words from a notes file or photo). The first three tabs only read; quizzes write exactly like the chat, and note words are saved only when you press “Add to my deck”. The data layer is [`dashboard_data.py`](dashboard_data.py), tested in `tests/test_dashboard.py`.

  ```bash
  uv run --extra dashboard streamlit run dashboard.py
  ```

  It opens at http://localhost:8501 and only listens on localhost, with Streamlit usage stats off (`.streamlit/config.toml`). It uses `DATABASE_PATH` and `ALLOWED_USER_ID` from `.env`. To point it elsewhere, set `DASHBOARD_DB` / `DASHBOARD_USER`, or pass arguments after `--`: `uv run --extra dashboard streamlit run dashboard.py -- --db /tmp/test.db --user 1`. Streamlit is an optional extra, so `uv run python -m pytest` works without it (the one app-rendering test is skipped).
- **Active production quizzes.** Say “quiz me in Korean” or “spelling quiz” (or 한글로 퀴즈, `/quiz style:active` in Discord, `/quiz active` in the terminal chat). You see the English meaning and type the Korean word in Hangul; spaces, NFD/NFC differences and a final `.`/`!`/`?` are ignored. Only words you've already met are asked: due words first, then the lowest predicted recall, skipping anything practiced actively in the last hour. Producing a word is a different task from recognising it, so these answers live in their own table (`active_attempts`, `mode='active'`) and events (`active_quiz_offered` / `active_quiz_result`, `style: "active"`). They never enter `verification_attempts`, so the flashcard scheduler, the daily verification state and `personalize.py`'s recall events are unaffected (tested, and `personalize.py` is hash-pinned).
- **Confusion pairs.** [`confusion.py`](confusion.py) finds word pairs you mix up, read-only: a wrong flashcard answer that is another deck word's meaning (wrong answers' text is stored from now on for this), a different deck word typed in an active quiz, or a word missed twice that looks like a word you've seen (jamo similarity ≥ 50%, e.g. 학교/학생). The top pairs appear in `/recall` and, when any exist, in `/focus`; a missed active quiz on a word in a pair adds a side-by-side note such as “나무 = tree, 나라 = country”.
- **Streak.** Progress answers (“how am I doing?”, “how's my streak?”) and `/summary` include how many days in a row you've met your goal.
- **Natural conversation.** Small talk, and unclear messages without numbers, get a short natural reply from `qwen3:1.7b` instead of a fixed template. The reply uses your last 6 messages (kept in memory only) and read-only facts: today's minutes and goal, your streak, and how many words are due (`chat_context.py`). It's told not to push Korean practice unless you ask, and to answer in the language of your message. The reply is text only: study logs, goal changes and quizzes still go through the intent router and confirmation, a reply that claims to have logged or changed something is replaced by the template, and with the AI off you get the old template. Chit-chat writes nothing beyond the replay receipt and interpretation audit every message gets (`tests/test_chat_conversation.py`). Intent classification is unchanged: the held-out benchmarks give identical results.
- **Today's focus.** `/focus` (Discord and terminal), or ask “what should I focus on today?”, for one read-only summary: today's minutes, goal and streak; due vs resting words and the next-up word; whether the recall model is still the population model or personalized; check-in and shadow-mode status; and one low-pressure suggestion. It only reads (`focus.py`) and never creates today's plan row.
- **Model size and CPU-only comparison.** On the frozen v2 set, `qwen3:1.7b` (the default) reaches 0.778 intent accuracy with or without a GPU (587 ms median per message on a laptop CPU, 1.4 GB on disk); `qwen3:0.6b` reaches 0.47–0.49 in 522 MB; tinyllama scores 0.156, below the rules alone. All models run CPU-only with the same accuracy: [experiments/model_size/REPORT.md](experiments/model_size/REPORT.md).
- **Check-in timing bandit (preview only).** `checkin_policy.py` is a disjoint LinUCB contextual bandit over `stay_silent`, `nudge_study` and `nudge_quiz`, using read-only context (time of day, weekend, goal progress, streak, due words, recent tired/busy report) and parameters stored in the `bandit_state` table. `/policy` shows the context, each action's predicted reward and exploration bonus, and what it would choose. It sends nothing; the fixed check-ins are unchanged. In shadow mode it learns from real outcomes while check-ins are enabled: each delivered check-in and an 18:00 no-message point are logged with their context, and scored three hours later by whether you logged study or answered a quiz. In simulation it beats the fixed schedule by a wide margin but not “never nudge” unless unnecessary nudges are cheap: [experiments/checkin_bandit/REPORT.md](experiments/checkin_bandit/REPORT.md).
- **Chat-model accuracy benchmark.** A frozen set of 60 held-out messages that only the AI can interpret, scored for intent accuracy, per-intent F1, slot precision/recall, bootstrap intervals and latency across qwen3:1.7b, qwen3:0.6b and AI-off: [experiments/conversation/HELDOUT.md](experiments/conversation/HELDOUT.md).

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
