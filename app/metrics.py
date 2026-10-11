"""In-process usage metrics with JSON and Prometheus text exposition.

Author: Rayyan Anwar
"""
from __future__ import annotations

import threading
import time

BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple, float] = {}
        self._hist: dict[str, dict] = {}
        self.started = time.time()

    def inc(self, name: str, value: float = 1, **labels) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            h = self._hist.setdefault(name, {"count": 0, "sum": 0.0, "buckets": [0] * len(BUCKETS_MS)})
            h["count"] += 1
            h["sum"] += value_ms
            for i, b in enumerate(BUCKETS_MS):
                if value_ms <= b:
                    h["buckets"][i] += 1

    def snapshot(self) -> dict:
        with self._lock:
            counters = [{"name": n, "labels": dict(lbl), "value": v} for (n, lbl), v in sorted(self._counters.items())]
            hist = {n: {"count": h["count"], "sum_ms": round(h["sum"], 3),
                        "buckets_le_ms": dict(zip(map(str, BUCKETS_MS), h["buckets"]))} for n, h in self._hist.items()}
        return {"uptime_s": round(time.time() - self.started, 1), "counters": counters, "histograms": hist}

    def total(self, name: str, **labels) -> float:
        want = {k: str(v) for k, v in labels.items()}
        with self._lock:
            return sum(v for (n, lbl), v in self._counters.items() if n == name and want.items() <= dict(lbl).items())

    def prometheus(self) -> str:
        out: list[str] = []
        with self._lock:
            names = sorted({n for n, _ in self._counters})
            for n in names:
                out.append(f"# TYPE {n} counter")
                for (name, lbl), v in sorted(self._counters.items()):
                    if name == n:
                        inner = ",".join(f'{k}="{_esc(val)}"' for k, val in lbl)
                        out.append(f"{n}{{{inner}}} {v:g}" if inner else f"{n} {v:g}")
            for n, h in sorted(self._hist.items()):
                out.append(f"# TYPE {n} histogram")
                for b, c in zip(BUCKETS_MS, h["buckets"]):
                    out.append(f'{n}_bucket{{le="{b}"}} {c}')
                out.append(f'{n}_bucket{{le="+Inf"}} {h["count"]}')
                out.append(f"{n}_sum {h['sum']:.3f}")
                out.append(f"{n}_count {h['count']}")
        return "\n".join(out) + "\n"
