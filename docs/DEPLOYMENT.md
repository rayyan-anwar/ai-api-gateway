# Deployment

Author: Rayyan Anwar

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # edit; leave GATEWAY_API_KEYS empty only for local work
set -a; source .env; set +a
python -m pytest -q
uvicorn app.main:app --port 8080  # console at http://127.0.0.1:8080
curl -s localhost:8080/v1/chat -H 'content-type: application/json' \
  -d '{"model":"local-sentiment","messages":[{"role":"user","content":"great and fast"}]}'
python scripts/evaluate.py
```

## Docker
```bash
docker compose up --build   # http://127.0.0.1:8080, health: /health, readiness: /ready
```
Status: Dockerfile/Compose written but **not built or run** in the authoring sandbox. CI includes a `docker build` job.

## Cloud (Cloud Run, ECS/Fargate, Azure Container Apps, Fly.io, Render, Kubernetes)
1. Build and push the image to your registry (requires your account).
2. Configure env vars from `.env.example` through the platform's **secret manager**, never in the image or repo. Set `GATEWAY_API_KEYS` to one or more long random keys.
3. Terminate HTTPS at the load balancer. Behind a proxy, the limiter sees the proxy's IP when auth is off; use API keys (the limiter keys on a hash of the key) rather than relying on IP.
4. Probes: liveness `GET /health`; readiness `GET /ready` (503 if a route has no configured provider). `GET /ready/deep` calls upstreams and needs a key.
5. Scrape `GET /metrics` (Prometheus text) with the gateway key; metrics are per instance and reset on restart.
6. **Scaling caveat:** rate-limit buckets live in process memory, so N replicas allow roughly N times the limit. A shared store (e.g. Redis) is not implemented.
7. Upstream credentials (`OPENAI_COMPAT_API_KEY`, `ANTHROPIC_API_KEY`) are required only for those providers; the author has not supplied or tested any.
8. Rollback: redeploy the previous image tag; the service is stateless.

Not deployed anywhere by the author; no hosted demo is claimed.

## Screenshots
Start the service, open the console, send a request and refresh metrics; capture with your OS tool into `docs/screenshots/`. None are bundled (none could be captured in the build sandbox).
