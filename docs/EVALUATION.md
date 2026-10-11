# Evaluation

Author: Rayyan Anwar

`python scripts/evaluate.py` writes `results/results.json` and `results/RESULTS.md`. The committed copy came from running it in the build sandbox (1 shared CPU core, Python 3.12).

## What is measured
| Area | Method | Observed in committed run |
|---|---|---|
| Pipeline latency (local provider) | 300 sequential requests through `Gateway.chat`, p50/p95/p99 | sub-millisecond (see RESULTS.md) |
| Concurrency | 400 requests in batches of 50 `asyncio.gather` | thousands of req/s in-process |
| Rate limiter accuracy | burst 10, 60/min, injected clock: 30 immediate requests, then 5 s later 10 attempts | 10 and 5 allowed, exactly as expected |
| Fallback | failing primary + local secondary, 100 requests | 100/100 succeeded via fallback |
| Timeout enforcement | 0.1 s timeout vs 2 s provider | HTTP 504 after about 0.1 s |
| Local sentiment sanity | 28 author-written sentences | 0.964 accuracy; one negation-without-lexicon-word case missed |

## Important caveats
- Latency/throughput are **in-process core numbers**, excluding HTTP parsing, TLS, network and real providers. They say nothing about end-to-end latency with real upstreams. Measure those in your own deployment (see `docs/DEPLOYMENT.md`).
- The sentiment set was written by the same author as the lexicon: it is a regression/sanity check, not evidence of general accuracy.
- Summarisation and keyword extraction are classical heuristics and are not scored here; no ROUGE-style claims are made.
- Adapters for OpenAI-compatible and Anthropic APIs are verified only against a local fake server that mimics the documented request/response shapes, not against the live services.
