# Implementation progress

## 2026-11-22 (KST)

- Inspected the existing bot, SQLite schema, three tests, repository instructions and uncommitted work. Preserved the existing goal/report/summary API, working tree changes, real database and secret configuration. Archived the earlier architecture proposal in RESEARCH_ROADMAP.md.
- Baseline: the existing virtual environment and `uv --cache-dir .uv-cache run --offline --no-sync python -m pytest -q` both passed all 3 original tests.
- Added two transactional user_version schema migrations, report revisions, daily plan snapshots, append-only typed-source event history, persistent scheduling state, quiz attempts and recovery proposals. No existing records were deleted; migration tests start from the original schema.
- Added KST opt-in midday/evening check-ins, pause/resume/disable, same-day restart catch-up, unique slot claims, bounded definite-failure retries, blocked-DM handling and conservative uncertain-delivery recovery. Missing responses make no behavioral inference.
- Added separate activity/verification states, a Korean vocabulary quiz with support history, and explicit approval/rejection/cancellation of daily recovery proposals. Original recurring goals remain unchanged by recovery.
- Reworked Discord startup into a configurable factory without import-time side effects. Added private slash commands, global DM command synchronization, no message-content intent, user authorization on every command, and mention-safe replies.
- Added a replaceable deterministic routing interface and decision audit. Added optional, disabled-by-default loopback Ollama advice with strict schema validation, size/time limits and deterministic fallback. Model output has no executor or application write access.
- Replaced the existing check.py raw-row dump with a read-only schema/count health check so the development utility does not print private goal/report contents.
- Added a separate research extra and synthetic family-held-out dataset, TF-IDF/logistic-regression model selection, artifact save/load, comparative classification/latency metrics and documented negative baseline result. The production policy remains deterministic.
- Validation after core changes: original 3 tests passed. Expanded validation: 75 passed with one test-helper failure, corrected for deferred Discord replies. Subsequent full runs passed all 78 tests, including model round-trip and a full scheduled-check-in / report / quiz / approved recovery / evening review / restart scenario. One upstream discord.py audioop deprecation warning remains on Python 3.11.
- Final compileall check passed for application modules, experiments and tests. `uv --locked` accepted the updated lockfile. Tests mock external services; no live Discord login, live DM delivery, or live Ollama call was attempted in this implementation run.
- Final suite after the read-only health-check privacy regression: **79 passed, 1 upstream audioop deprecation warning**, using `uv --cache-dir .uv-cache run --locked --offline --extra research python -m pytest -q`.
- Environment: initial uv cache permission and sandbox DNS failures were resolved using a workspace cache and authorized dependency downloads. Windows sandbox blocks asyncio socketpair initialization; full asynchronous tests ran outside the sandbox with mocked services.
- Experiment: `uv --cache-dir .uv-cache run --offline --extra research python -m experiments.router.run` completed. Seed 42, 240/80/160 family-separated examples: RuleRouter accuracy/F1 1.0000/1.0000; baseline 0.6250/0.5595. Baseline test LLM recall 0.0; no improvement claim. Full report: experiments/router/results_seed42.json.

## Remaining limits

Live Discord/Ollama integration still needs the documented manual walkthrough. SQLite is unencrypted. Discord is external. Crash-uncertain DMs are not resent, so delivery is not guaranteed exactly once. Check-in catch-up is limited to six hours on the current KST day. Quiz results test only the predefined vocabulary and never verify duration. DEFER is a recommendation with manual resubmission, not a generic deferred worker. One goal and one allowed user are supported. Legacy timestamps and unstored past goal revisions cannot be reconstructed. LSTM, Mamba, distillation, critics, iOS event ingestion and public deployment remain future work.
