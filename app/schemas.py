"""Pydantic models used for HTTP validation and OpenAPI documentation (Pydantic v2).

Author: Rayyan Anwar
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=100, description="Route alias from GET /v1/models")
    messages: list[MessageIn] = Field(min_length=1, max_length=50)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float | None = Field(None, ge=0, le=2)
