"""Provider adapters: local (built-in), OpenAI-compatible and Anthropic Messages API.

Remote adapters use only the standard library (urllib) inside worker threads, so they add no dependencies.
They are tested against a local fake upstream server, NOT against the real vendor services.

Author: Rayyan Anwar
"""
from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from . import local_nlp
from .errors import ProviderError
from .models import ChatRequest

MAX_RESPONSE_BYTES = 2_000_000


@dataclass
class ProviderResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, (socket.timeout, TimeoutError)) or (
        isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (socket.timeout, TimeoutError)))


def http_json(method: str, url: str, headers: dict, payload: dict | None, timeout: float) -> dict:
    """Blocking JSON request. Never includes upstream bodies or credentials in raised errors."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ProviderError("upstream URL must use http or https")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (scheme validated above)
            body = resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"upstream returned HTTP {exc.code}", status=exc.code) from None
    except (urllib.error.URLError, OSError) as exc:
        if _is_timeout(exc):
            raise ProviderError("upstream timed out", kind="timeout") from None
        raise ProviderError("could not connect to upstream") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProviderError("upstream response too large")
    try:
        out = json.loads(body)
    except ValueError:
        raise ProviderError("upstream returned invalid JSON") from None
    if not isinstance(out, dict):
        raise ProviderError("upstream returned an unexpected JSON shape")
    return out


class Provider:
    name = "base"
    configured = False

    async def complete(self, req: ChatRequest, upstream_model: str, timeout: float) -> ProviderResult:
        raise NotImplementedError

    async def health(self, deep: bool = False, timeout: float = 5.0) -> dict:
        return {"configured": self.configured, "status": "configured" if self.configured else "not_configured"}


class LocalProvider(Provider):
    name = "local"
    configured = True

    async def complete(self, req: ChatRequest, upstream_model: str, timeout: float) -> ProviderResult:
        text = req.messages[-1].content
        try:
            content = await asyncio.to_thread(local_nlp.run_task, upstream_model, text, req.max_tokens)
        except ValueError as exc:
            raise ProviderError(str(exc)) from None
        return ProviderResult(content)

    async def health(self, deep: bool = False, timeout: float = 5.0) -> dict:
        return {"configured": True, "status": "ok"}


class OpenAICompatProvider(Provider):
    """Any server exposing POST {base_url}/chat/completions (OpenAI-style), e.g. self-hosted inference servers."""

    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.configured = bool(self.base_url)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def complete(self, req: ChatRequest, upstream_model: str, timeout: float) -> ProviderResult:
        payload = {"model": upstream_model, "max_tokens": req.max_tokens,
                   "messages": [{"role": m.role, "content": m.content} for m in req.messages]}
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        out = await asyncio.to_thread(http_json, "POST", f"{self.base_url}/chat/completions",
                                      self._headers(), payload, timeout)
        try:
            content = out["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
        except (KeyError, IndexError, TypeError):
            raise ProviderError("malformed upstream response") from None
        usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
        return ProviderResult(content, _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens")))

    async def health(self, deep: bool = False, timeout: float = 5.0) -> dict:
        if not self.configured:
            return {"configured": False, "status": "not_configured"}
        if not deep:
            return {"configured": True, "status": "configured"}
        try:
            await asyncio.to_thread(http_json, "GET", f"{self.base_url}/models", self._headers(), None, timeout)
        except ProviderError as exc:
            return {"configured": True, "status": "unreachable", "detail": str(exc)}
        return {"configured": True, "status": "ok"}


class AnthropicProvider(Provider):
    """Anthropic Messages API adapter (POST {base_url}/v1/messages)."""

    name = "anthropic"

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com", version: str = "2023-06-01") -> None:
        self.api_key, self.base_url, self.version = api_key, base_url.rstrip("/"), version
        self.configured = bool(api_key)

    async def complete(self, req: ChatRequest, upstream_model: str, timeout: float) -> ProviderResult:
        system = "\n\n".join(m.content for m in req.messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in req.messages if m.role != "system"]
        if not convo or convo[0]["role"] != "user":
            raise ProviderError("conversation must start with a user message for this provider")
        payload: dict = {"model": upstream_model, "max_tokens": req.max_tokens, "messages": convo}
        if system:
            payload["system"] = system
        if req.temperature is not None:
            payload["temperature"] = min(req.temperature, 1.0)
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}
        out = await asyncio.to_thread(http_json, "POST", f"{self.base_url}/v1/messages", headers, payload, timeout)
        try:
            blocks = [b["text"] for b in out["content"] if b.get("type") == "text"]
        except (KeyError, TypeError, AttributeError):
            raise ProviderError("malformed upstream response") from None
        if not blocks:
            raise ProviderError("upstream response contained no text")
        usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
        return ProviderResult("".join(blocks), _int(usage.get("input_tokens")), _int(usage.get("output_tokens")))


def _int(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def build_providers(settings) -> dict[str, Provider]:
    return {
        "local": LocalProvider(),
        "openai_compat": OpenAICompatProvider(settings.openai_base_url, settings.openai_api_key),
        "anthropic": AnthropicProvider(settings.anthropic_api_key, settings.anthropic_base_url,
                                       settings.anthropic_version),
    }
