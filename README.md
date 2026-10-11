# AI API Gateway

**Author: Rayyan Anwar** — AI/ML Engineer | Researcher | Technology Manager | Software Developer

A small, readable gateway that sits in front of AI providers and gives you request routing with ordered fallback, per-attempt timeouts, strict validation, API-key auth, token-bucket rate limiting, liveness/readiness checks, structured JSON logs and usage metrics (JSON and Prometheus). It ships with a **local provider** (summariser, sentiment, keywords) so everything works with no paid API.

## Implemented vs optional

| Feature | Status |
|---|---|
| Routing by model alias, ordered fallback across providers | Implemented, tested |
| Timeouts (asyncio + socket), 502/503/504 semantics | Implemented, tested |
| Strict request validation, consistent error shape | Implemented, tested |
| API-key auth (constant-time compare, hashed client ids) | Implemented, tested |
| Token-bucket rate limiting + `Retry-After` / `X-RateLimit-*` | Implemented, tested (incl. thread safety) |
| Metrics: counters, latency histogram, JSON + Prometheus | Implemented, tested |
| JSON logs with request ids | Implemented (logging output covered indirectly) |
| Local provider: extractive summary, lexicon sentiment, keywords | Implemented, tested |
| OpenAI-compatible adapter, Anthropic Messages adapter | Implemented; tested **only against a local fake upstream** |
| HTTP layer (FastAPI) and console UI | Implemented; API tests written but **not executed in the authoring sandbox** |
| Streaming, tool calls, distributed rate limiting, caching, per-key quotas/billing | Not implemented |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python -m pytest -q
uvicorn app.main:app --port 8080       # console: http://127.0.0.1:8080  ·  docs: /docs
```
```bash
curl -s localhost:8080/v1/chat -H 'content-type: application/json' \
  -d '{"model":"local-summarizer","messages":[{"role":"user","content":"First point. Second point. Third point. Fourth point."}],"max_tokens":128}'
```
With `GATEWAY_API_KEYS` set, add `-H "Authorization: Bearer <your-key>"`.

### Routes and fallback
Define aliases in a JSON file (`config/routes.example.json`) and point `GATEWAY_ROUTES_FILE` at it:
```json
{"model": "assistant", "targets": [
  {"provider": "openai_compat", "model": "YOUR_UPSTREAM_MODEL"},
  {"provider": "local", "model": "local/summarizer"}]}
```
Providers: `local`, `openai_compat` (needs `OPENAI_COMPAT_BASE_URL`), `anthropic` (needs `ANTHROPIC_API_KEY`). Unconfigured providers are skipped.

### Endpoints
`POST /v1/chat` · `GET /v1/models` · `GET /health` · `GET /ready` · `GET /ready/deep` · `GET /metrics` · `GET /api/metrics` · `GET /api/info` · `/docs`

Errors always look like `{"error": {"code", "message", "request_id", "details?"}}`.

## Results
`python scripts/evaluate.py` → `results/RESULTS.md` (committed copy from a real run). Numbers are in-process core measurements; see [`docs/EVALUATION.md`](docs/EVALUATION.md) for what they do and do not show.

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md) · [System card](docs/MODEL_CARD.md) · [Deployment](docs/DEPLOYMENT.md)

## Testing status
- 33 tests executed with `python -m unittest` and passing: parsing, local tools, rate limiter, metrics, config, gateway routing/fallback/timeouts/auth, and both HTTP adapters against a loopback fake upstream.
- 6 HTTP API tests (`TestApi`) are included but were **skipped** in the authoring sandbox (FastAPI and httpx could not be installed offline). They run in CI. Run `pytest` yourself before trusting the HTTP layer.
- Docker image not built locally; Ruff not run (stdlib syntax/unused-import check used instead).

## Known limitations
- Local sentiment is a tiny English lexicon; local summariser is a frequency heuristic.
- Rate limits and metrics are per process and in memory.
- Real OpenAI/Anthropic endpoints are untested; token usage is estimated when upstreams omit it.
- No content moderation, prompt-injection filtering or factuality checks on pass-through responses.
- With `GATEWAY_API_KEYS` empty, authentication is off: do not expose publicly.

## GitHub repository metadata
- **Description:** FastAPI gateway for AI providers: routing with fallback, timeouts, rate limiting, auth, health checks and usage metrics, with a local no-API-key provider.
- **Topics:** `api-gateway`, `llm`, `fastapi`, `rate-limiting`, `observability`, `prometheus`, `mlops`, `python`

## License
MIT © Rayyan Anwar. See [LICENSE](LICENSE).
