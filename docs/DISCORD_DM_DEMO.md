# Try a conversation with PM

Use one bot process, your configured Discord account and a **fresh demo database**. A 30-minute recurring goal makes the proposed change easy to verify. From a new PowerShell window in the project, use the normal Discord configuration from the README and select a new file name for each demonstration.

```powershell
$env:DATABASE_PATH = 'data/sol-aim-demo-en.db'
$env:OLLAMA_ENABLED = 'false'
uv run --locked python bot.py
```

## A report, an obstacle and a change of plan

1. In the bot's DM, create the demo goal with `/setgoal title:Korean minutes:30`.
2. Send **I studied Korean for 20 minutes today, but I'm tired and want to finish tomorrow.**
3. Read PM's draft. It should identify a 20-minute daily total, tiredness and a proposed move to tomorrow. Send “how am I doing?” before confirming. The target should still be 30 and the total unreported.
4. Reply `confirm #N` using the **draft number shown**. PM records the 20-minute self-report and obstacle. Read the separate proposal. With a fresh 30-minute goal, it offers to move 10 minutes to tomorrow, bringing tomorrow's target to 40. Use `/summary` now to see today's target of 30 and the pending proposal.
5. Reply `approve #P` using the **proposal number shown**. Ask “how am I doing?” and run `/history`. Today's effective target should be 20, the original target 30, and the minutes self-reported. Both the original goal event and the subsequent decision remain in history. The following KST day uses the 40-minute snapshot. Later days use the recurring 30-minute goal.
6. On another fresh demo, use `discard #N` at step 4 or `reject #P` at step 5. Discarding leaves the report unrecorded. Rejecting keeps the confirmed report and both original targets. A plain “yes” cannot approve a plan. Slash fallback for a proposal is `/adjust proposal:P decision:approve` or `reject`.

Repeat steps 1–5 with a new demo database for each language. PM uses the language of the incoming message for the summary and confirmation instructions.

| Language | Send | Confirm draft | Approve or reject |
|---|---|---|---|
| Vietnamese | Hôm nay tôi học tiếng Hàn được 20 phút, nhưng hơi mệt nên muốn chuyển 10 phút còn lại sang ngày mai. | `xác nhận #N` | `đồng ý #P` / `từ chối #P` |
| Korean | 오늘 한국어를 20분 공부했는데 피곤해서 내일 마저 하고 싶어요. | `확인 #N` | `승인 #P` / `거절 #P` |

## Verify the reported Vietnamese failure in Discord

1. Restart the bot from the updated checkout with a new demo database. With a 30-minute Korean goal and no pending proposal, send **được chứ**. PM should tell you in Vietnamese that nothing is pending.
2. Send the exact Vietnamese message in the table. Look for **20/30 phút**, **mệt**, **10 phút còn lại**, and tomorrow's KST date. The console should show `stage=rules outcome=matched intents=REPORT_PROGRESS,REPORT_OBSTACLE,REQUEST_RECOVERY` followed by `stage=dispatch outcome=draft`. This path works with Ollama disabled.
3. Send the same message again before confirming. Both previews should show the same draft number. Check `/summary` for an unreported total and a target of 30. Send `xác nhận #N` once. The summary should now show 20 self-reported minutes, a target of 30, and one pending proposal.
4. A standalone **được chứ** now shows the numbered approval instruction. To approve conversationally, use Discord's **Reply** action on PM's actual proposal message and send **được chứ**. PM should acknowledge the approval in Vietnamese. Check `/summary` for a target of 20, an original goal of 30 and the self-report. The approved proposal carries 10 minutes to tomorrow's 30-minute goal. Replying to the same proposal again should leave both targets unchanged.
5. On a fresh demo, change “10 phút còn lại” to “7 phút còn lại”. PM should ask about the discrepancy before recording anything. For a midnight check, create the proposal just before 00:00 Asia/Seoul and try approving it after midnight. PM should reject the old proposal for the new day.
6. To exercise inference, use “Chiều nay tôi ôn tiếng Hàn 20 phút, nhưng hơi mệt nên muốn chuyển 10 phút còn lại sang ngày mai.” The current grammar delegates this paraphrase. With inference enabled, look for `stage=rules outcome=miss` and `stage=ollama outcome=invoked`. A complete validated result produces a draft. An offline endpoint, a timeout, or an incomplete response should produce a targeted Vietnamese question and no draft. The console distinguishes `disabled`, `backend_unavailable`, `timeout`, `invalid_json`, `invalid_schema`, and `invalid_evidence`.

### Why the original test passed

The earlier Vietnamese fixture ended with “muốn học tiếp vào ngày mai”, which the rules recognized. In the production wording, the rules recognized the report and fatigue individually, then abandoned the whole match at “muốn chuyển 10 phút còn lại sang ngày mai”. The old model schema only represented the destination. Its evidence check required the entire message to contain exactly one number for a numeric study report, so even a correct 20-minute interpretation conflicted with the second number. Unresolved requests and standalone assent both led to the same English help text.

The checkout inspected for this fix had inference disabled, model `qwen3:0.6b` and a 10-second timeout. The reported Discord reply does not identify the deployed process's settings or distinguish a skipped call from a failed call. New routing diagnostics include stages, intent names and bounded reason codes. Logs exclude the text, inferred minute values, credentials and raw model output.

### Verification results

`python -m pytest -q -rs` completed with **134 passed, 2 skipped, 1 warning**. The decision-model and knowledge-tracing benchmark suites require their isolated environments. The warning comes from discord.py importing Python's deprecated `audioop` module. The DM regressions use the actual handler, simulated Discord delivery, and temporary SQLite databases.

In a separate loopback smoke test using installed `qwen3.5:4b`, the exact reported DM took the rule path with all three intents. The model correctly interpreted a separate Vietnamese 15-minute report. For the “Chiều nay” paraphrase above, it returned only the transfer and omitted the report and obstacle. Evidence validation rejected the incomplete proposal. The enabled-backend integration test also exercises a complete model response through draft confirmation. A new Discord gateway exchange remains a manual check.

## Other live checks

- **Simple report and correction.** On a fresh goal, send “today's total is 12 minutes”. Then send “I studied for 5 minutes”. PM asks whether 5 is the new daily total and shows the previous 12. A bare “20 minutes” also prompts for confirmation.
- **Vocabulary.** Say “quiz me” after reporting. For the water question, reply `answer #N water` with the attempt number. Ask for progress to compare the quiz result with self-reported time. Slash commands also support quiz answers.
- **History.** Try “how much did I study yesterday?” A fresh database has no saved report for yesterday. For additional dated history, use `/history`.
- **Restart.** Restart after a draft or a proposal, then confirm it on the same KST day. A newer report invalidates an older report draft or carryover proposal. Reusing a resolved number cannot apply its effects again. The integration test also replays the identical Discord event ID, which typing a second DM cannot reproduce.
- **Scheduling.** Opt in with `/checkins` and choose a midday slot a few minutes ahead in KST. Reply to the check-in with today's total. Restart after delivery and verify the slot appears once in `/checkins action:status`. Pause before the evening slot and check that no evening DM arrives. Resume and disable using the existing command.
- **Authorization.** A message in a guild channel or a DM from a different account should receive no conversational reply. A bot-authored message is ignored.
- **Ollama.** Stop the demo bot, enable the loopback adapter and select an installed model using `OLLAMA_ENABLED`, `OLLAMA_URL` and `OLLAMA_MODEL`. Restart and try “I put in twenty minutes of Korean practice today.” Review any proposed total. Repeat with the local endpoint unavailable. The expected fallback is a clarification. Structured reports and the three acceptance messages still work.

## What PM retains

Each incoming text is limited to 2,000 characters. Only the current authorized message can go to local inference. Date-scoped answers come from SQLite. The dispatcher accepts at most four validated intents. For Vietnamese reply-based approval, a system event binds the delivered Discord prompt ID to its proposal. The same recovery service checks ownership, KST day, status, report and plan revisions for both numbered and reply-based approvals. The app stores one structured draft per user alongside domain events and message IDs for deduplication. DM bodies and model prompts are excluded from app storage, training and exports.

Startup installs three additive SQLite tables alongside the existing v2 schema. `dm_receipts` holds processed event IDs, `dm_drafts` holds the current interpretation awaiting confirmation, and `recovery_moves` binds a proposed carryover to the report and both daily plans.

Database effects and the incoming event receipt commit together. Discord sends the reply after commit. If delivery fails, check progress or `/summary` to inspect what was saved. Check-in opt-in, pause, retry and deduplication still belong to the scheduler. A move to tomorrow adjusts daily study targets only. Use `/checkins` for reminder settings and `/setgoal` for recurring goals.
