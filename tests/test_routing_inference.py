import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from config import Config
from llm_adapter import OllamaAdapter
from memory import Memory
from router import AuditedRouter, Route, RoutingInput, RuleRouter


@pytest.mark.parametrize("event,route", [
    (RoutingInput(command="report", minutes=30), Route.RULE),
    (RoutingInput(command="report"), Route.CLARIFY),
    (RoutingInput(command="report", minutes=True), Route.CLARIFY),
    (RoutingInput(command="unknown"), Route.CLARIFY),
    (RoutingInput(text="done"), Route.CLARIFY),
    (RoutingInput(text="I need help planning tonight"), Route.LLM),
    (RoutingInput(text="Review my plan sometime later", non_urgent=True), Route.DEFER),
    (RoutingInput(text="", priority="URGENT", non_urgent=True), Route.RULE),
])
def test_routing(event, route):
    decision = RuleRouter().route(event)
    assert decision.route == route and decision.reason and decision.latency_ms >= 0


def test_routing_audit_without_raw_text(store):
    AuditedRouter(store).route(123, RoutingInput(text="synthetic private obstacle description"))
    event = Memory(store).history(123, kind="routing_decision")[0]
    assert event["payload"]["route"] == "LLM"
    assert "private" not in json.dumps(event)
    with pytest.raises(ValueError):
        RoutingInput(priority="INVENTED")


def test_disabled_inference_never_calls_backend():
    transport = AsyncMock()
    result = asyncio.run(OllamaAdapter(Config(), transport).suggest("Please help me plan"))
    assert result.fallback and result.reason == "disabled"
    transport.assert_not_called()


@pytest.mark.parametrize("output", [
    "not JSON", '[]', '{"action":"RUN_COMMAND","message":"oops"}',
    '{"action":"SUGGEST","message":"ok","target":1}',
    '{"action":"SUGGEST","message":42}', '{"action":"SUGGEST","message":""}',
    '{"action":[],"message":"oops"}',
])
def test_invalid_model_output_falls_back(output):
    transport = AsyncMock(return_value={"response": output})
    result = asyncio.run(OllamaAdapter(Config(ollama_enabled=True), transport).suggest("Help me plan tonight"))
    assert result.fallback


def test_valid_advice_and_unavailable_backend():
    config = Config(ollama_enabled=True)
    transport = AsyncMock(return_value={"response": '{"action":"SUGGEST","message":"Try a smaller first step."}'})
    result = asyncio.run(OllamaAdapter(config, transport).suggest("Help me plan tonight"))
    assert not result.fallback and result.action == "SUGGEST"
    assert transport.call_args.args[0]["stream"] is False
    failing = AsyncMock(side_effect=OSError("synthetic backend unavailable"))
    assert asyncio.run(OllamaAdapter(config, failing).suggest("Help me plan tonight")).fallback


def test_inference_timeout_is_bounded():
    async def stalled(payload):
        await asyncio.sleep(10)

    result = asyncio.run(OllamaAdapter(Config(ollama_enabled=True, ollama_timeout=0.1), stalled).suggest("Help me plan tonight"))
    assert result.fallback and result.reason == "timeout"
    assert result.latency_ms < 1000


@pytest.mark.parametrize("options", [
    {"allowed_user_id": -1}, {"allowed_user_id": True}, {"ollama_timeout": 0},
    {"ollama_timeout": float("nan")}, {"ollama_timeout": float("inf")},
    {"ollama_url": "https://external.example"}, {"ollama_url": "http://localhost@external.example"},
    {"ollama_url": "http://localhost:bad"}, {"midday": "27:00"},
])
def test_invalid_config(options):
    with pytest.raises(ValueError):
        Config(**options)


def test_configuration_aliases_and_secret_safe_errors(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "synthetic-token")
    monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)
    monkeypatch.setenv("GUILD_ID", "456")
    monkeypatch.setenv("ALLOWED_USER_ID", "123")
    assert Config.from_env().guild_id == 456
    assert Config.from_env().token == "synthetic-token"
    monkeypatch.setenv("ALLOWED_USER_ID", "synthetic-secret")
    with pytest.raises(ValueError) as error:
        Config.from_env()
    assert "synthetic-secret" not in str(error.value)
