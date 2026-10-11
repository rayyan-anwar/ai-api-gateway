"""Exception types shared across the gateway.

Author: Rayyan Anwar
"""
from __future__ import annotations


class GatewayError(Exception):
    """An error that maps directly to an HTTP response."""

    def __init__(self, status: int, code: str, message: str, headers: dict | None = None,
                 details: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.headers = headers or {}
        self.details = details or {}


class ProviderError(Exception):
    """A failure talking to an upstream provider. ``kind`` is 'error' or 'timeout'."""

    def __init__(self, message: str, status: int | None = None, kind: str = "error") -> None:
        super().__init__(message)
        self.status, self.kind = status, kind


class ConfigError(ValueError):
    """Invalid gateway configuration."""
