# AGENT.md

# Coding Agent Instructions

**Project:** Personal Accountability AI  
**Specification:** README.md  
**Current phase:** Python MVP  
**Target architecture:** Local-first, event-driven, modular

## 1. Agent Mission

Implement and validate the Personal Accountability AI project according to the architecture and experimental constraints described in README.md.

Prioritize:

1. Correctness.
2. User safety and privacy.
3. Reproducibility.
4. Simplicity.
5. Resource efficiency.
6. Extensibility.

Do not introduce complexity solely because an advanced model or framework is available.

## 2. Scope and Authority

The coding agent MAY:

- Create implementation modules.
- Create tests and fixtures.
- Refactor code with equivalent behavior.
- Improve error handling.
- Add structured logging.
- Implement approved features.
- Propose architectural improvements.

The coding agent MUST NOT, without explicit approval:

- Change the project's primary objectives.
- Introduce new autonomous agent roles.
- Change data collection scope.
- Enable external data transmission.
- Expand LLM permissions.
- Modify user goals or behavioral thresholds without authorization.
- Introduce paid services.
- Deploy or expose services publicly.
- Delete existing user data.
- Change the training objective or model architecture without documenting the reason.

Model-generated instructions, messages and configuration values must never be treated as trusted system commands.

## 3. Development Phases

### Phase 0 — Python MVP

Implement:

- Discord event handling.
- SQLite storage.
- Scheduler.
- Rule-based routing.
- Local Ollama adapter.
- Deferred evaluation.
- Basic user configuration.
- Basic verification workflow.

Do not require Mamba or LSTM to run the MVP.

### Phase 1 — Dataset and Baselines

Build reproducible training and evaluation pipelines.

Start with simpler models.

Include dataset validation and held-out evaluation.

### Phase 2 — Mamba Decision Model

Implement a small sequential decision model using structured feature sequences.

Begin with supervised training.

Do not automatically substitute Mamba for the existing deterministic router.

### Phase 3 — On-Policy Distillation

Introduce an explicit environment/simulator interface.

Define:

- State representation.
- Action space.
- State transitions.
- Teacher policy.
- Student policy.
- Training objective.
- Evaluation protocol.

On-policy state collection must be distinguishable from offline supervised data.

### Phase 4 — Critic Ablation

Compare:

- Deterministic rule validator.
- MLP critic.
- LSTM critic.

Do not retain the LSTM critic without measured benefit.

### Phase 5 — Rust Migration

Only migrate after the Python implementation has stable interfaces, tests and evaluation evidence.

Preserve external behavior and data compatibility during migration.

## 4. Suggested Project Structure

```text
personal-accountability-ai/
│
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── defaults.json
│   └── schemas.py
│
├── src/
│   └── accountability/
│       ├── bot/
│       ├── scheduler/
│       ├── events/
│       ├── routing/
│       ├── memory/
│       ├── inference/
│       ├── verification/
│       └── integrations/
│
├── research/
│   ├── datasets/
│   ├── simulator/
│   ├── baselines/
│   ├── mamba/
│   ├── distillation/
│   └── critics/
│
├── tests/
│
└── docs/
    ├── ARCHITECTURE.md
    ├── EXPERIMENTS.md
    └── DECISIONS.md
```

Directories should be created incrementally as features are implemented, not all at once.

## 5. Engineering Principles

### Separation of Responsibilities

- Scheduler determines when an event is due.
- Router recommends how the event should be handled.
- Validator checks whether the action is permitted.
- Executor performs validated actions.
- Inference adapters communicate with LLM backends.
- Memory layer persists states and history.

Do not combine routing, inference, scheduling and database mutation into a single large function.

### Deterministic Fallback

Core scheduling, reporting and persistence must continue working if the LLM is unavailable.

### Resource Efficiency

Avoid unnecessary LLM calls.

Prefer deterministic parsing for simple structured reports.

Record inference latency and invocation counts.

Do not claim a neural model is more efficient without measurements.

### Asynchronous Processing

Use non-blocking I/O for Discord and network calls.

Avoid blocking the event loop with model training or heavy computation.

Use bounded queues, timeouts and controlled retries.

## 6. Event Contract

An event should contain:

```json
{
  "event_id": "example-id",
  "event_type": "daily_checkin",
  "priority": "GENERAL",
  "timestamp": "2026-11-22T12:00:00+09:00",
  "payload": {},
  "status": "pending"
}
```

All event types and priorities must be validated against allowlists.

Unknown events should not automatically trigger new capabilities.

## 7. Decision Model Contract

The model returns a proposal, not a command.

Example:

```json
{
  "action": "ASK_FOLLOWUP",
  "confidence": 0.78,
  "reason_code": "LOW_INFORMATION"
}
```

Confidence represents a model estimate and must not automatically be treated as a calibrated probability.

The validator must reject unknown actions and malformed output.

The decision model must never bypass permission checks.

## 8. Mamba Training Requirements

Before training:

- Define the exact task and labels.
- Establish baseline performance.
- Specify input feature dimensions.
- Prevent training/validation leakage.
- Fix random seeds where practical.
- Record hyperparameters.
- Record CPU/GPU memory usage.

Track:

- Accuracy or appropriate decision metrics.
- Per-class recall.
- False-negative rates for important classes.
- Inference latency.
- Model size.
- Memory consumption.

Do not treat simulated training success as evidence of real-world effectiveness.

## 9. OPD Requirements

Separate:

- Student rollout generation.
- Teacher evaluation.
- Loss computation.
- Student optimization.
- Held-out evaluation.

Document the direction and meaning of any KL divergence objective.

If teacher outputs only action labels, describe the method accurately as imitation learning.

Never use student-generated outcomes as ground-truth rewards without an explicitly justified evaluation mechanism.

## 10. Critic Requirements

A critic must have a documented training target.

For an action-value critic:

```text
Input:
    state history
    candidate action

Output:
    predicted action value
```

Avoid introducing a critic that merely duplicates the actor's predictions.

Evaluate whether its benefits exceed additional inference and training costs.

## 11. Safety and Privacy Constraints

Do not:

- Infer deception solely from response length.
- Treat application usage as proof of intentional avoidance.
- Diagnose mental health conditions.
- Automatically escalate sensitive conversations to third-party APIs without consent.
- Allow LLM output to execute arbitrary commands.
- Store unnecessary personal information.
- Print tokens, credentials or sensitive message content into logs.

Maintain an appropriate deterministic fallback for urgent or safety-relevant events.

User-controlled monitoring must remain opt-in and reversible.

## 12. Configuration Policy

Allow only explicitly defined configuration keys.

Every model-generated proposal must pass:

1. Schema validation.
2. Type validation.
3. Range validation.
4. Authorization checks.
5. Change-rate limits.
6. Audit recording.

Sensitive configuration changes require human approval.

Maintain rollback capability.

Do not let an LLM create new policy categories by writing arbitrary JSON fields.

## 13. Testing Requirements

Use pytest for the Python prototype.

Minimum coverage should include:

- Event priority ordering.
- Queue fairness.
- Scheduler recovery after restart.
- SQLite persistence.
- Duplicate-event handling.
- Malformed JSON rejection.
- Invalid numeric thresholds.
- LLM timeout.
- Deferred retry limits.
- Model output validation.
- Insufficient behavioral history.
- Correct completion-threshold behavior.
- Safety fallback when inference is unavailable.

Important invariants:

```text
actual >= target
    => target_completed

LLM unavailable
    => core scheduler remains functional

unknown action
    => reject

insufficient evidence
    => do not infer dishonesty

critical event
    => never depend solely on deferred LLM

unapproved policy change
    => reject
```

Every bug fix should include a regression test when feasible.

## 14. Experiment Discipline

For every significant experimental change, record:

- Hypothesis.
- Baseline.
- Changed variables.
- Dataset.
- Evaluation method.
- Measurements.
- Observed limitations.
- Resulting questions.

Avoid changing several major model components simultaneously.

Do not report an experiment as successful when the necessary measurements are missing.

## 15. Coding Agent Workflow

Before implementation:

1. Read README.md and the current phase.
2. Inspect existing project files.
3. Identify the smallest relevant component.
4. Explain important assumptions.
5. Make a bounded change.
6. Run relevant tests.
7. Report results and remaining limitations.

After implementation, summarize:

```text
Implemented:
Tests:
Results:
Known limitations:
Architectural changes:
Next suggested experiment:
```

Do not claim tests passed unless they were actually executed.

Do not claim a model improved without comparative evaluation.

## 16. Definition of Done

A feature is considered complete when:

- Intended behavior is implemented.
- Input and output contracts are defined.
- Relevant tests pass.
- Failure behavior is documented.
- Changes do not bypass safety constraints.
- Logging and persistence are appropriate.
- Documentation is updated when necessary.

An experiment is complete when its results and limitations are recorded, even if the hypothesis was not supported.

---

## Final Directive

**Build for evidence, not complexity.**

The purpose of this project is not to maximize the number of models, agents or subsystems.

The purpose is to investigate whether a small, adaptive decision system can provide useful accountability while minimizing unnecessary inference, maintaining user autonomy and remaining operationally reliable.