"""Fault-injection providers used by tests and evaluation (never registered by default).

Author: Rayyan Anwar
"""
from __future__ import annotations

import asyncio

from .errors import ProviderError
from .models import ChatRequest
from .providers import Provider, ProviderResult


class FailingProvider(Provider):
    name = "failing"
    configured = True

    def __init__(self, status: int | None = 500) -> None:
        self.status, self.calls = status, 0

    async def complete(self, req: ChatRequest, upstream_model: str, timeout: float) -> ProviderResult:
        self.calls += 1
        raise ProviderError("simulated failure", status=self.status)


class SlowProvider(Provider):
    name = "slow"
    configured = True

    def __init__(self, delay_s: float = 1.0) -> None:
        self.delay_s, self.calls = delay_s, 0

    async def complete(self, req: ChatRequest, upstream_model: str, timeout: float) -> ProviderResult:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return ProviderResult("slow answer")
