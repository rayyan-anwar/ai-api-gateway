"""Request model and strict validation (independent of any web framework).

Author: Rayyan Anwar
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .errors import GatewayError

ROLES = ("system", "user", "assistant")
MAX_MESSAGES, MAX_CONTENT_CHARS, MAX_TOTAL_CHARS, MAX_TOKENS_LIMIT, MAX_MODEL_LEN = 50, 20_000, 100_000, 4096, 100
ALLOWED_FIELDS = {"model", "messages", "max_tokens", "temperature"}
_REQ_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    max_tokens: int = 256
    temperature: float | None = None


def _bad(msg: str) -> GatewayError:
    return GatewayError(422, "invalid_request", msg)


def parse_chat_request(raw) -> ChatRequest:
    if not isinstance(raw, dict):
        raise _bad("request body must be a JSON object")
    extra = set(raw) - ALLOWED_FIELDS
    if extra:
        raise _bad(f"unsupported fields: {sorted(extra)} (streaming and tools are not supported)")
    model = raw.get("model")
    if not isinstance(model, str) or not 1 <= len(model) <= MAX_MODEL_LEN:
        raise _bad(f"'model' must be a string of 1-{MAX_MODEL_LEN} characters")
    msgs = raw.get("messages")
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= MAX_MESSAGES:
        raise _bad(f"'messages' must be a list of 1-{MAX_MESSAGES} items")
    parsed, total = [], 0
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise _bad(f"messages[{i}] must have exactly 'role' and 'content'")
        if m["role"] not in ROLES:
            raise _bad(f"messages[{i}].role must be one of {list(ROLES)}")
        c = m["content"]
        if not isinstance(c, str) or not c.strip() or len(c) > MAX_CONTENT_CHARS:
            raise _bad(f"messages[{i}].content must be a non-blank string up to {MAX_CONTENT_CHARS} characters")
        total += len(c)
        parsed.append(Message(m["role"], c))
    if total > MAX_TOTAL_CHARS:
        raise _bad(f"total message content exceeds {MAX_TOTAL_CHARS} characters")
    if parsed[-1].role != "user":
        raise _bad("the last message must have role 'user'")
    mt = raw.get("max_tokens", 256)
    if isinstance(mt, bool) or not isinstance(mt, int) or not 1 <= mt <= MAX_TOKENS_LIMIT:
        raise _bad(f"'max_tokens' must be an integer in [1, {MAX_TOKENS_LIMIT}]")
    temp = raw.get("temperature")
    if temp is not None and (isinstance(temp, bool) or not isinstance(temp, (int, float)) or not 0 <= temp <= 2):
        raise _bad("'temperature' must be a number in [0, 2]")
    return ChatRequest(model, tuple(parsed), mt, None if temp is None else float(temp))


def sanitize_request_id(value: str | None) -> str:
    return value if value and _REQ_ID.match(value) else uuid.uuid4().hex


def estimate_tokens(text: str) -> int:
    """Rough estimate (~4 characters per token); always flagged as estimated in responses."""
    return max(1, len(text) // 4)
