
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import GoalStore

def test_goal_and_report(tmp_path):
    store = GoalStore(tmp_path / "test.db")
    user = 123
    day = "2026-11-22"

    store.set_goal(user, "Korean", 30)

    initial = store.summary(user, day)
    assert initial["status"] == "not_reported"

    store.report(user, day, 5)
    result = store.summary(user, day)

    assert result["minutes"] == 5
    assert result["status"] == "reported"
    assert result["target_reached"] is False

    store.report(user, day, 30)
    result = store.summary(user, day)

    assert result["minutes"] == 30
    assert result["target_reached"] is True
    assert result["status"] == "reported"


def test_report_requires_goal(tmp_path):
    store = GoalStore(tmp_path / "test.db")

    with pytest.raises(ValueError):
        store.report(123, "2026-11-22", 30)


def test_invalid_goal(tmp_path):
    store = GoalStore(tmp_path / "test.db")

    with pytest.raises(ValueError):
        store.set_goal(123, "Korean", -10)


def test_blank_env_values_use_defaults(monkeypatch):
    from config import Config
    for name in ("DISCORD_TOKEN", "ALLOWED_USER_ID", "DISCORD_GUILD_ID", "OLLAMA_TIMEOUT", "MIDDAY_CHECKIN"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "legacy-token")
    config = Config.from_env()
    assert (config.allowed_user_id, config.guild_id, config.midday, config.token) == (0, 0, "12:00", "legacy-token")
    with pytest.raises(ValueError, match="Set DISCORD_TOKEN and ALLOWED_USER_ID"):
        Config(token="").require_discord()
