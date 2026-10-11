"""Thread-safe token-bucket rate limiter with an injectable clock.

Author: Rayyan Anwar
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_s: int = 0


class TokenBucketLimiter:
    def __init__(self, rate_per_min: float, burst: int, clock=time.monotonic, max_buckets: int = 10_000) -> None:
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate_per_s = max(0.0, rate_per_min) / 60.0
        self.enabled = rate_per_min > 0
        self.burst, self.clock, self.max_buckets = burst, clock, max_buckets
        self._buckets: dict[str, list[float]] = {}  # key -> [tokens, last_seen]
        self._lock = threading.Lock()

    def check(self, key: str) -> RateDecision:
        if not self.enabled:
            return RateDecision(True, 0, 0)
        now = self.clock()
        with self._lock:
            if key not in self._buckets and len(self._buckets) >= self.max_buckets:
                oldest = sorted(self._buckets, key=lambda k: self._buckets[k][1])[: max(1, self.max_buckets // 10)]
                for k in oldest:
                    del self._buckets[k]
            tokens, last = self._buckets.get(key, [float(self.burst), now])
            tokens = min(float(self.burst), tokens + (now - last) * self.rate_per_s)
            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[key] = [tokens, now]
                return RateDecision(True, self.burst, int(tokens))
            self._buckets[key] = [tokens, now]
            wait = math.ceil((1.0 - tokens) / self.rate_per_s)
            return RateDecision(False, self.burst, 0, max(1, wait))
