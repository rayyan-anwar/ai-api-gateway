"""Environment-based configuration, route loading and JSON logging.

Author: Rayyan Anwar
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

KNOWN_PROVIDERS = ("local", "openai_compat", "anthropic")


@dataclass(frozen=True)
class Target:
    provider: str
    model: str


DEFAULT_ROUTES = {
    "local-summarizer": (Target("local", "local/summarizer"),),
    "local-sentiment": (Target("local", "local/sentiment"),),
    "local-keywords": (Target("local", "local/keywords"),),
}


def load_routes(data: dict) -> dict[str, tuple[Target, ...]]:
    routes_in = data.get("routes") if isinstance(data, dict) else None
    if not isinstance(routes_in, list) or not routes_in:
        raise ConfigError("routes config must be an object with a non-empty 'routes' list")
    routes: dict[str, tuple[Target, ...]] = {}
    for r in routes_in:
        alias, targets = (r or {}).get("model"), (r or {}).get("targets")
        if not isinstance(alias, str) or not alias or alias in routes:
            raise ConfigError(f"route alias must be a unique non-empty string: {alias!r}")
        if not isinstance(targets, list) or not targets:
            raise ConfigError(f"route '{alias}' needs a non-empty 'targets' list")
        parsed = []
        for t in targets:
            if not isinstance(t, dict) or t.get("provider") not in KNOWN_PROVIDERS or not t.get("model"):
                raise ConfigError(f"route '{alias}': each target needs provider in {KNOWN_PROVIDERS} and a model")
            parsed.append(Target(t["provider"], str(t["model"])))
        routes[alias] = tuple(parsed)
    return routes


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = ()
    rate_limit_rpm: float = 60.0
    rate_limit_burst: int = 10
    request_timeout_s: float = 30.0
    routes: dict = field(default_factory=lambda: dict(DEFAULT_ROUTES))
    openai_base_url: str = ""
    openai_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""
    anthropic_version: str = "2023-06-01"
    log_level: str = "INFO"


def _num(env, name, default, cast, lo, hi):
    raw = env.get(name)
    if raw in (None, ""):
        return default
    try:
        v = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not lo <= v <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return v


def load_settings(env=None) -> Settings:
    env = os.environ if env is None else env
    keys = tuple(k.strip() for k in env.get("GATEWAY_API_KEYS", "").split(",") if k.strip())
    routes = dict(DEFAULT_ROUTES)
    rf = env.get("GATEWAY_ROUTES_FILE", "")
    if rf:
        try:
            routes = load_routes(json.loads(Path(rf).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read GATEWAY_ROUTES_FILE: {exc}") from exc
    return Settings(
        api_keys=keys,
        rate_limit_rpm=_num(env, "GATEWAY_RATE_LIMIT_RPM", 60.0, float, 0, 1_000_000),
        rate_limit_burst=_num(env, "GATEWAY_RATE_LIMIT_BURST", 10, int, 1, 100_000),
        request_timeout_s=_num(env, "GATEWAY_REQUEST_TIMEOUT_S", 30.0, float, 0.05, 300),
        routes=routes,
        openai_base_url=env.get("OPENAI_COMPAT_BASE_URL", "").rstrip("/"),
        openai_api_key=env.get("OPENAI_COMPAT_API_KEY", ""),
        anthropic_base_url=env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        anthropic_version=env.get("ANTHROPIC_VERSION", "2023-06-01"),
        log_level=env.get("GATEWAY_LOG_LEVEL", "INFO").upper(),
    )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level if level in logging._nameToLevel else "INFO")
