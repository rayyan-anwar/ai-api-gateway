"""Gateway core: authentication, rate limiting, routing with fallback, timeouts, usage metrics.

Framework-independent so it can be tested without a web server. Author: Rayyan Anwar
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import uuid

from .config import Settings
from .errors import GatewayError, ProviderError
from .metrics import Metrics
from .models import ChatRequest, estimate_tokens, parse_chat_request
from .providers import Provider
from .ratelimit import RateDecision, TokenBucketLimiter

log = logging.getLogger("gateway")


class Gateway:
    def __init__(self, settings: Settings, providers: dict[str, Provider], limiter: TokenBucketLimiter | None = None,
                 metrics: Metrics | None = None) -> None:
        self.settings, self.providers = settings, providers
        self.limiter = limiter or TokenBucketLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)
        self.metrics = metrics or Metrics()

    # ---- access control -------------------------------------------------
    def authenticate(self, presented: str | None, client_ip: str = "unknown") -> str:
        """Return a stable, non-reversible client id. Raises 401 when keys are configured and none matches."""
        if not self.settings.api_keys:
            return f"ip:{client_ip}"
        if presented:
            for key in self.settings.api_keys:
                if hmac.compare_digest(presented.encode(), key.encode()):
                    return "key:" + hashlib.sha256(key.encode()).hexdigest()[:12]
        self.metrics.inc("gateway_auth_failures_total")
        raise GatewayError(401, "unauthorized", "missing or invalid API key",
                           headers={"WWW-Authenticate": "Bearer"})

    def check_rate(self, client_id: str) -> RateDecision:
        d = self.limiter.check(client_id)
        if not d.allowed:
            self.metrics.inc("gateway_rate_limited_total")
            raise GatewayError(429, "rate_limited", "rate limit exceeded; retry later",
                               headers={"Retry-After": str(d.retry_after_s), "X-RateLimit-Limit": str(d.limit),
                                        "X-RateLimit-Remaining": "0"})
        return d

    # ---- routing --------------------------------------------------------
    def models(self) -> list[dict]:
        out = []
        for alias, targets in self.settings.routes.items():
            out.append({"model": alias, "targets": [
                {"provider": t.provider, "model": t.model,
                 "configured": bool(self.providers.get(t.provider) and self.providers[t.provider].configured)}
                for t in targets]})
        return out

    async def chat(self, raw, client_id: str = "anonymous") -> dict:
        started = time.perf_counter()
        label = "unknown"
        try:
            req = parse_chat_request(raw)
            targets = self.settings.routes.get(req.model)
            if not targets:
                raise GatewayError(404, "model_not_found", f"unknown model '{req.model}'; see GET /v1/models")
            label = req.model
            body = await self._dispatch(req, targets)
        except GatewayError as exc:
            self.metrics.inc("gateway_requests_total", model=label, status=str(exc.status))
            raise
        latency = (time.perf_counter() - started) * 1e3
        body["latency_ms"] = round(latency, 2)
        self.metrics.inc("gateway_requests_total", model=label, status="200")
        self.metrics.observe("gateway_request_latency_ms", latency)
        return body

    async def _dispatch(self, req: ChatRequest, targets) -> dict:
        timeout = self.settings.request_timeout_s
        attempts: list[dict] = []
        for t in targets:
            prov = self.providers.get(t.provider)
            if prov is None or not prov.configured:
                attempts.append({"provider": t.provider, "model": t.model, "outcome": "not_configured"})
                self.metrics.inc("gateway_provider_attempts_total", provider=t.provider, outcome="not_configured")
                continue
            t0 = time.perf_counter()
            outcome, status, res = "error", None, None
            try:
                res = await asyncio.wait_for(prov.complete(req, t.model, timeout), timeout)
                outcome = "ok"
            except (asyncio.TimeoutError, TimeoutError):
                outcome = "timeout"
            except ProviderError as exc:
                outcome, status = ("timeout" if exc.kind == "timeout" else "error"), exc.status
            except Exception:  # a buggy adapter must not take the gateway down
                log.exception("unexpected provider failure", extra={"fields": {"provider": t.provider}})
            ms = round((time.perf_counter() - t0) * 1e3, 2)
            attempt = {"provider": t.provider, "model": t.model, "outcome": outcome, "latency_ms": ms}
            if status is not None:
                attempt["upstream_status"] = status
            attempts.append(attempt)
            self.metrics.inc("gateway_provider_attempts_total", provider=t.provider, outcome=outcome)
            if outcome == "ok":
                return self._success(req, t, res, attempts)
            if len(attempts) < len(targets):
                self.metrics.inc("gateway_fallbacks_total", provider=t.provider)
        outcomes = {a["outcome"] for a in attempts}
        if outcomes == {"not_configured"}:
            raise GatewayError(503, "no_provider_configured", "no configured provider for this model",
                               details={"attempts": attempts})
        if outcomes <= {"timeout", "not_configured"}:
            raise GatewayError(504, "upstream_timeout", "all providers timed out", details={"attempts": attempts})
        raise GatewayError(502, "upstream_error", "all providers failed", details={"attempts": attempts})

    def _success(self, req: ChatRequest, target, res, attempts: list[dict]) -> dict:
        est = res.prompt_tokens is None or res.completion_tokens is None
        pt = res.prompt_tokens if res.prompt_tokens is not None else sum(estimate_tokens(m.content) for m in req.messages)
        ct = res.completion_tokens if res.completion_tokens is not None else estimate_tokens(res.content)
        self.metrics.inc("gateway_tokens_total", pt, kind="prompt", provider=target.provider)
        self.metrics.inc("gateway_tokens_total", ct, kind="completion", provider=target.provider)
        return {
            "id": "gw-" + uuid.uuid4().hex[:16],
            "model": req.model,
            "provider": target.provider,
            "upstream_model": target.model,
            "content": res.content,
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "estimated": est},
            "fallback_used": len(attempts) > 1,
            "attempts": attempts,
        }

    # ---- health ---------------------------------------------------------
    async def readiness(self, deep: bool = False) -> dict:
        health = {name: await p.health(deep) for name, p in self.providers.items()}
        ok = ("ok", "configured")
        uncovered = [alias for alias, ts in self.settings.routes.items()
                     if not any(health.get(t.provider, {}).get("status") in ok for t in ts)]
        return {"ready": not uncovered, "providers": health, "routes_without_working_provider": uncovered}
