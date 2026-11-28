"""Validated configuration; loading does not connect Discord or touch the database."""
import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from time_utils import valid_time


def local_endpoint(url):
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("OLLAMA_URL must be a local HTTP endpoint without credentials")
    try:
        parsed.port
    except ValueError:
        raise ValueError("Invalid Ollama port") from None
    return url.rstrip("/")


def env(name, default, *aliases):
    """Read a variable, treating blank values (as in a fresh copy of .env.example) as unset."""
    for key in (name, *aliases):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


@dataclass(frozen=True)
class Config:
    token: str = ""
    allowed_user_id: int = 0
    guild_id: int = 0
    database_path: str = "data/accountability.db"
    midday: str = "12:00"
    evening: str = "21:00"
    ollama_enabled: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:0.6b"
    ollama_timeout: float = 10.0
    ollama_vision_model: str = "qwen2.5vl:3b"

    def __post_init__(self):
        if type(self.ollama_enabled) is not bool:
            raise ValueError("Ollama enabled must be a boolean")
        for value in (self.allowed_user_id, self.guild_id):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("Discord IDs must be non-negative integers below 2^63")
        valid_time(self.midday)
        valid_time(self.evening)
        local_endpoint(self.ollama_url)
        if not math.isfinite(self.ollama_timeout) or not 0.1 <= self.ollama_timeout <= 60:
            raise ValueError("Ollama timeout must be between 0.1 and 60 seconds")
        for name in (self.ollama_model, self.ollama_vision_model):
            if not name.strip() or len(name) > 100:
                raise ValueError("Invalid Ollama model name")
        if not self.database_path.strip():
            raise ValueError("Database path cannot be empty")

    @classmethod
    def from_env(cls):
        enabled = env("OLLAMA_ENABLED", "false").lower()
        if enabled not in {"true", "false"}:
            raise ValueError("OLLAMA_ENABLED must be true or false")
        try:
            return cls(
                token=env("DISCORD_TOKEN", "", "DISCORD_BOT_TOKEN"),
                allowed_user_id=int(env("ALLOWED_USER_ID", "0")),
                guild_id=int(env("DISCORD_GUILD_ID", "0", "GUILD_ID")),
                database_path=env("DATABASE_PATH", "data/accountability.db"),
                midday=env("MIDDAY_CHECKIN", "12:00"), evening=env("EVENING_REVIEW", "21:00"),
                ollama_enabled=enabled == "true", ollama_url=env("OLLAMA_URL", "http://127.0.0.1:11434"),
                ollama_model=env("OLLAMA_MODEL", "qwen3:0.6b"),
                ollama_timeout=float(env("OLLAMA_TIMEOUT", "10")),
                ollama_vision_model=env("OLLAMA_VISION_MODEL", "qwen2.5vl:3b"),
            )
        except (ValueError, TypeError):
            raise ValueError("Invalid environment configuration; check .env.example") from None

    def require_discord(self):
        if not self.token.strip() or not self.allowed_user_id:
            raise ValueError("Set DISCORD_TOKEN and ALLOWED_USER_ID in .env")
