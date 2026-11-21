"""Replaceable rule-first routing policy. Decisions are recommendations only."""
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Protocol

from memory import record
from time_utils import now_kst, valid_minutes


class Route(str, Enum):
    RULE = "RULE"
    CLARIFY = "CLARIFY"
    LLM = "LLM"
    DEFER = "DEFER"


STRUCTURED = {"setgoal", "report", "summary", "checkins", "quiz", "answer", "recover", "adjust", "history"}


@dataclass(frozen=True)
class RoutingInput:
    text: str = ""
    command: str | None = None
    minutes: int | None = None
    non_urgent: bool = False
    priority: str = "GENERAL"

    def __post_init__(self):
        if self.priority not in {"GENERAL", "SYSTEM", "URGENT"}:
            raise ValueError("Unknown event priority")
        if not isinstance(self.text, str) or len(self.text) > 2000:
            raise ValueError("Input text must be at most 2000 characters")
        if type(self.non_urgent) is not bool:
            raise ValueError("non_urgent must be a boolean")


@dataclass(frozen=True)
class Decision:
    route: Route
    reason: str
    latency_ms: float


class Router(Protocol):
    def route(self, event: RoutingInput) -> Decision: ...


class RuleRouter:
    def route(self, event):
        start = perf_counter()
        if event.priority == "URGENT":
            route, reason = Route.RULE, "urgent_deterministic_response"
        elif event.command in STRUCTURED:
            try:
                if event.command == "report":
                    valid_minutes(event.minutes)
                route, reason = Route.RULE, "structured_command"
            except ValueError:
                route, reason = Route.CLARIFY, "missing_or_invalid_minutes"
        elif event.command is not None:
            route, reason = Route.CLARIFY, "unknown_command"
        elif not event.text.strip() or len(event.text.split()) < 3:
            route, reason = Route.CLARIFY, "insufficient_information"
        elif event.non_urgent:
            route, reason = Route.DEFER, "explicitly_non_urgent"
        else:
            route, reason = Route.LLM, "free_text_requires_interpretation"
        return Decision(route, reason, (perf_counter() - start) * 1000)


class AuditedRouter:
    def __init__(self, store, policy=None):
        self.store, self.policy = store, policy or RuleRouter()

    def route(self, user_id, event):
        decision = self.policy.route(event)
        with self.store.connect() as db:
            record(db, user_id, now_kst().date().isoformat(), "routing_decision",
                   {"route": decision.route.value, "reason": decision.reason,
                    "latency_ms": decision.latency_ms, "command": event.command}, "system")
        return decision
