# Personal Accountability AI

[Tiếng Việt](README.vi.md) | [English](README.md)

![Python Versions](https://img.shields.io/badge/Python-3.11-blue)

**An adaptive, local-first personal assistant for progress tracking, behavioral decision-making, and continuous learning.**

**Version:** 0.2 — Architecture Draft  
**Status:** Research & Planning  
**Initial Experiment:** 7 days  
**Development Strategy:** Python Prototype → Model Research → Rust Runtime

---

## 1. Project Overview

Personal Accountability AI is a proactive assistant designed to monitor daily progress, verify reported accomplishments, identify behavioral changes, and adapt its interactions to the user's circumstances.

Unlike conventional reminder bots, the system aims to combine:

- **Accountability:** Track commitments and daily objectives.
- **Adaptation:** Adjust interactions based on user behavior and circumstances.
- **Verification:** Assess task completion and learning outcomes.
- **Persistent Memory:** Maintain historical context, progress, and knowledge states.
- **Efficient Intelligence:** Minimize unnecessary large language model (LLM) inference.
- **Continuous Experimentation:** Improve system design through measurable experiments.

The initial implementation is intended for personal experimentation rather than commercial deployment or third-party monitoring.

### Core Philosophy

> Observe → Understand → Decide → Verify → Adapt → Evaluate

A failed experiment does not necessarily mean a failed project. Every experiment should provide evidence, reveal limitations, or generate useful research questions.

## 2. Project Objectives

The system aims to investigate whether an adaptive assistant can:

1. Proactively request daily progress reports.
2. Distinguish goal completion from behavioral anomalies.
3. Evaluate self-reported information without automatically assuming dishonesty.
4. Detect potential knowledge retention problems.
5. Adapt schedules to circumstances such as insufficient sleep, late work shifts, or unexpected obligations.
6. Reduce inference costs through adaptive decision routing.
7. Escalate complex or ambiguous situations to an appropriate LLM.
8. Learn from previous interactions without compromising user autonomy.

## 3. Proposed System Architecture

```text
Discord / iOS Shortcuts
          |
          v
    Event Ingestion
          |
          v
    Priority Queue
  GENERAL / SYSTEM / URGENT
          |
          v
    Feature Extractor
          |
          v
    Decision Router
          |
    +-----+-------------+
    |                   |
    v                   v
Rule Engine       Decision Model
    |              (Experimental
    |                Tiny Mamba)
    |                   |
    |             Action Validator
    |                   |
    +---------+---------+
              |
              v
       Action Executor
              |
    +---------+-----------+
    |         |           |
    v         v           v
Templates  Local LLM  Deferred Queue
              |
        Optional Cloud LLM
              |
              v
       SQLite Memory
              |
              v
      Feedback & Metrics
```

The architecture separates deterministic system operations from machine-learning-based decision-making.

**Core requirement:** Safety rules, authorization checks, and execution constraints must remain independent of the learned decision policy.

### 3.1 Event Priority System

The application uses three logical priority levels inspired by operating-system scheduling concepts.

| Priority | Category | Examples |
|---|---|---|
| P1 | GENERAL | Daily reports, routine questions |
| P2 | SYSTEM | Verification, retries, scheduled reviews |
| P3 | URGENT | Events requiring immediate attention |

Event priority and LLM routing decisions are separate concerns.

URGENT events must have deterministic fallback behavior when inference services are unavailable.

The scheduler must implement fairness mechanisms to prevent indefinite starvation of lower-priority events.

The priority system operates at the application level and does not require FreeRTOS.

## 4. Adaptive Decision Model

### 4.1 Why Mamba?

Mamba is a selective state space architecture designed for sequence modeling.

This project investigates whether a small Mamba-based network can function as a **sequential decision policy**, rather than a general-purpose conversational model.

The model is intended to learn relationships between:

- Historical user behavior.
- Current events and reported activities.
- Task completion patterns.
- Message characteristics.
- Previous assistant actions.
- Outcomes of earlier decisions.

Its output is a proposed action, not an executable command.

### 4.2 Decision Action Space

The initial action space is:

```python
ACTIONS = [
    "RULE_RESPONSE",
    "ASK_FOLLOWUP",
    "LOCAL_LLM",
    "DEFER",
    "REQUEST_REVIEW"
]
```

Cloud inference is controlled by a separate authorization and safety policy.

The decision model must not independently execute commands, modify its action space, or override system constraints.

### 4.3 Input Representation

Early experiments will use structured behavioral features rather than raw natural-language tokens.

Candidate features include:

```text
message_length
message_type
task_completion_ratio
sleep_duration
energy_level
response_delay
recent_completion_trend
previous_followup_count
verification_status
time_since_last_checkin
```

Missing values must be explicitly represented.

Self-reported observations, model-generated inferences, and verified information must remain distinguishable.

### 4.4 Behavioral Baselines

The system maintains historical baselines for comparable activities.

For example, with a daily learning target of 30 minutes:

| Reported Duration | Interpretation |
|---|---|
| 5 minutes | Below target |
| 30 minutes | Target completed |
| 90 minutes | Target completed with a positive deviation |

An unusually high result must not automatically increase future targets.

A behavioral anomaly is an observation requiring interpretation, not evidence of deception.

During the cold-start period, insufficient historical data should produce an uncertain baseline rather than a confident judgment.

## 5. Model Training Strategy

### Phase A — Supervised Baselines

Create a dataset containing behavioral event sequences and annotated routing decisions.

Compare:

- Rule-based routing.
- Logistic Regression.
- Decision Tree.
- Small MLP.
- GRU.
- Tiny Mamba.

Synthetic data may be used for engineering tests and algorithm development, but synthetic evaluation results must not be presented as evidence of effectiveness with real users.

### Phase B — On-Policy Distillation

Investigate training a Mamba student policy using guidance from a stronger teacher policy.

Proposed training workflow:

1. The student observes a simulated state.
2. The student selects an action.
3. The environment produces the next state.
4. The teacher evaluates the student-encountered state and provides a target action distribution.
5. The student updates its policy using an appropriate distillation objective.
6. The process repeats, followed by held-out evaluation.

A candidate objective is:

\[
\mathcal{L}_{OPD}
=
\mathbb{E}_{s\sim d^{\pi_\theta}}
\left[
D_{KL}
\left(
\pi_T(\cdot \mid s)
\parallel
\pi_\theta(\cdot \mid s)
\right)
\right]
\]

Where:

- \(\pi_\theta\) is the student policy.
- \(\pi_T\) is the teacher policy.
- \(d^{\pi_\theta}\) is the state distribution induced by student rollouts.

The on-policy component requires collecting states encountered under the student's current policy.

If the teacher produces only discrete action labels, the experiment should be described as on-policy imitation learning rather than distributional policy distillation.

Teacher judgments must not automatically be treated as ground truth.

### Phase C — Optional Critic Evaluation

Investigate whether an independent critic improves decision quality.

Proposed roles:

- **Mamba Actor:** Proposes actions.
- **MLP or LSTM Critic:** Estimates action quality.
- **Deterministic Validator:** Enforces system constraints.

A possible action-value function is:

\[
Q_\phi(s_t,a_t)
\]

The critic must be trained using meaningful feedback, observed outcomes, or validated targets.

Three configurations will be evaluated:

| Experiment | Architecture |
|---|---|
| A | Mamba + Rule Validator |
| B | Mamba + MLP Critic |
| C | Mamba + LSTM Critic |

The LSTM critic is an experimental component, not a mandatory dependency.

It should only be retained if measurable improvements justify its additional complexity and computational cost.

## 6. Daily Accountability Workflow

### 6.1 Midday Check-in

Default schedule: **12:00 PM**, user-configurable.

The assistant requests a progress report and may:

- Update daily task completion.
- Verify reported accomplishments.
- Identify obstacles.
- Review previously learned material.
- Suggest adjustments for the remainder of the day.

### 6.2 Evening Review

The assistant summarizes:

- Completed objectives.
- Unfinished objectives.
- Identified obstacles.
- Knowledge requiring additional review.
- Potential adjustments for the following day.

Related review activities should be grouped into the same interaction when possible to reduce notification fatigue and unnecessary inference calls.

### 6.3 Recovery Mode

The assistant may adapt its schedule when the user reports:

- Insufficient sleep.
- Late work shifts.
- Illness.
- Unexpected responsibilities.
- Unusually low energy.

Possible adjustments include reduced workloads, delayed check-ins, additional rest, or smaller alternative objectives.

The system should not wake a user solely to verify whether they are sleeping.

## 7. Progress Verification and Learning Memory

The system distinguishes between reported activity and verified learning outcomes.

Initial verification states:

| State | Meaning |
|---|---|
| `reported` | Activity reported by the user |
| `verified` | Successfully completed a verification task |
| `passed_after_support` | Passed after additional instruction |
| `review_required` | Requires additional practice |
| `unknown` | Insufficient evidence |

Verification history must be preserved.

Previously verified knowledge may return to `review_required` if subsequent observations suggest that it has been forgotten.

Future teaching harnesses may access relevant shared memory and return structured learning results.

Specialized modules must not overwrite unrelated user information.

## 8. Deferred LLM Evaluation

When a local LLM is unavailable or an event does not require immediate inference, the system may queue the evaluation for a later session.

Example:

```json
{
  "event_type": "behavior_review",
  "priority": "SYSTEM",
  "status": "pending",
  "retry_slot": "evening",
  "attempts": 0
}
```

Deferred tasks require:

- Bounded retries.
- Deduplication.
- Expiration policies.
- Failure logging.
- Context preservation.
- Cancellation when the event is no longer relevant.

Safety-critical responses must never depend exclusively on deferred inference.

## 9. iOS Shortcuts Integration

Optional Shortcuts automations may report application-open or application-close events.

Initial examples include user-selected applications such as TikTok and YouTube.

The system must:

- Require explicit user configuration and permission.
- Collect only necessary information.
- Authenticate incoming events.
- Allow monitoring to be paused or disabled.
- Avoid treating application activity as proof of intentional message avoidance.

These events are incomplete behavioral signals, not comprehensive device usage records.

## 10. Safety, Privacy, and Authorization

### 10.1 Safety Escalation

Ambiguous expressions of fatigue or distress must not automatically be classified as emergencies.

The assistant should support clarification, appropriate immediate responses, and human-support escalation when necessary.

Cloud LLM usage requires prior informed user consent and should transmit only context relevant to the request.

The system must not represent an LLM as capable of independently guaranteeing user safety.

### 10.2 Configuration Security

LLM-generated configuration proposals must pass:

1. Strict schema validation.
2. Allowlisted configuration fields.
3. Data type validation.
4. Allowed value ranges.
5. Change limits and cooldowns.
6. Authorization checks.
7. Audit logging and rollback support.

Sensitive configuration changes require explicit human approval.

Valid JSON alone is not sufficient protection against prompt injection.

All user messages and model-generated outputs must be treated as untrusted inputs.

## 11. Technology Stack

### Python Prototype

| Component | Technology |
|---|---|
| Language | Python |
| Discord integration | discord.py |
| Database | SQLite |
| Scheduling | APScheduler or asyncio |
| Local inference | Ollama |
| Machine learning | PyTorch |
| Experimental model | Mamba |
| Configuration | JSON / Pydantic |
| Testing | pytest |

### Future Rust Runtime

After the Python prototype and its interfaces are validated:

| Component | Technology |
|---|---|
| Main language | Rust |
| Async runtime | Tokio |
| Discord integration | Serenity |
| Serialization | Serde |
| Database | SQLite |
| HTTP client | Reqwest |

Model training may remain in Python.

Inference may initially operate through a separate Python service, avoiding the need to immediately reimplement the trained model in Rust.

## 12. Experimental Roadmap

### Experiment 0 — Seven-Day MVP

The first experiment focuses on validating the accountability workflow.

Minimum scope:

- Discord check-ins.
- SQLite persistence.
- Daily goal tracking.
- Basic progress verification.
- Rule-first routing.
- Optional local LLM calls.
- Deferred evaluation.
- Basic iOS Shortcuts event ingestion.

Mamba training, On-Policy Distillation, and LSTM Critic evaluation are explicitly outside the minimum seven-day MVP scope.

### Evaluation Metrics

**Behavioral Metrics**

- Check-in response rate.
- Daily goal completion rate.
- User-perceived usefulness.
- Unnecessary notification frequency.
- Number of useful schedule adjustments.

**Routing Metrics**

- Decision agreement with reviewed reference labels.
- Unnecessary LLM invocation rate.
- Missed escalation rate.
- Inference latency.
- CPU and memory consumption.

**Learning Metrics**

- Verification pass rate.
- Review frequency.
- Evidence of knowledge retention or forgetting.

Seven days of personal observations are insufficient to establish generalizable behavioral improvement.

## 13. Experiment Log

Each experiment should record:

```text
Experiment ID:
Date:
Hypothesis:
Configuration:
Expected Outcome:
Observed Outcome:
Metrics:
Unexpected Behavior:
Limitations:
New Questions:
Next Iteration:
```

Architecture changes should be supported by experimental evidence rather than isolated failures.

## 14. Development Roadmap

- [ ] Build the Python accountability prototype.
- [ ] Implement Discord and SQLite integration.
- [ ] Create the scheduler and event priority system.
- [ ] Integrate iOS Shortcuts.
- [ ] Implement the adaptive rule-based router.
- [ ] Complete the seven-day self-experiment.
- [ ] Design the behavioral dataset and simulator.
- [ ] Establish rule-based and neural baselines.
- [ ] Train a supervised Tiny Mamba decision model.
- [ ] Experiment with On-Policy Distillation.
- [ ] Compare rule, MLP, and LSTM critics.
- [ ] Migrate stable runtime components to Rust.
- [ ] Introduce an optional language-teaching harness.
- [ ] Evaluate persistent VPS deployment.

## 15. Research Questions

1. Can a lightweight sequential decision model reduce unnecessary LLM inference?
2. Does a Mamba-based policy outperform simpler alternatives on behavioral routing tasks?
3. Can On-Policy Distillation improve robustness to states encountered under the student's own policy?
4. Does an independent critic improve decision quality enough to justify its computational overhead?
5. Can adaptive accountability increase engagement without creating notification fatigue?
6. How can behavioral adaptation remain useful without compromising autonomy or privacy?
7. Which components benefit measurably from migration to Rust?

---

## 16. Project Status

This README describes a proposed architecture and experimental roadmap.

The system has not yet been validated through completed experiments.

Model-performance claims, behavioral improvements, and deployment capabilities should not be considered established until the corresponding implementations and evaluations are completed.

### Guiding Principle

**Build the smallest testable system, collect evidence, and increase complexity only when the results justify it.**

The objective is not to maximize the number of models, agents, or subsystems.

The objective is to investigate whether a lightweight, adaptive decision system can deliver useful personal accountability while minimizing inference costs, maintaining user autonomy, and preserving operational reliability.