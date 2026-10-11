"""FastAPI application for the AI API Gateway.

Author: Rayyan Anwar
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __author__, __version__
from .config import Settings, load_settings, setup_logging
from .errors import GatewayError
from .gateway import Gateway
from .models import sanitize_request_id
from .providers import Provider, build_providers
from .schemas import ChatBody

log = logging.getLogger("gateway.http")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


def create_app(settings: Settings | None = None, providers: dict[str, Provider] | None = None,
               configure_logging: bool = False) -> FastAPI:
    settings = settings or load_settings()
    if configure_logging:
        setup_logging(settings.log_level)
    gw = Gateway(settings, providers or build_providers(settings))
    app = FastAPI(
        title="AI API Gateway",
        version=__version__,
        description="Routing, fallback, timeouts, rate limiting, auth, health checks and usage metrics in front of "
                    "AI providers. Author: Rayyan Anwar.",
        contact={"name": __author__},
        license_info={"name": "MIT"},
    )
    app.state.gateway = gw

    def rid(request: Request):
        return getattr(request.state, "request_id", None)

    def error(request: Request, status: int, code: str, message: str, headers=None, details=None):
        body = {"error": {"code": code, "message": message, "request_id": rid(request)}}
        if details:
            body["error"]["details"] = details
        return JSONResponse(status_code=status, content=body, headers=headers)

    @app.exception_handler(GatewayError)
    async def on_gateway_error(request: Request, exc: GatewayError):
        return error(request, exc.status, exc.code, exc.message, exc.headers, exc.details)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        errs = [{"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))} for e in exc.errors()]
        return error(request, 422, "invalid_request", "request validation failed", details={"errors": errs})

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        return error(request, exc.status_code, "http_error", str(exc.detail))

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request.state.request_id = sanitize_request_id(request.headers.get("x-request-id"))
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        log.info("request", extra={"fields": {
            "request_id": request.state.request_id, "method": request.method, "path": request.url.path,
            "status": response.status_code, "latency_ms": round((time.perf_counter() - t0) * 1e3, 2),
            "client": getattr(request.state, "client_id", None)}})
        return response

    async def guard(request: Request, response: Response) -> str:
        auth = request.headers.get("authorization", "")
        presented = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key")
        client_id = gw.authenticate(presented, request.client.host if request.client else "unknown")
        request.state.client_id = client_id
        decision = gw.check_rate(client_id)
        if decision.limit:
            response.headers["X-RateLimit-Limit"] = str(decision.limit)
            response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        return client_id

    @app.post("/v1/chat", summary="Route a chat request to a provider (with fallback)")
    async def chat(body: ChatBody, client_id: str = Depends(guard)):
        return await gw.chat(body.model_dump(exclude_none=True), client_id)

    @app.get("/v1/models", summary="List route aliases and their targets")
    async def models(client_id: str = Depends(guard)):
        return {"data": gw.models()}

    @app.get("/health", summary="Liveness")
    async def health():
        return {"status": "ok", "version": __version__, "author": __author__}

    @app.get("/ready", summary="Readiness (provider configuration, no outbound calls)")
    async def ready():
        r = await gw.readiness(deep=False)
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    @app.get("/ready/deep", summary="Readiness with live upstream checks (requires API key if configured)")
    async def ready_deep(client_id: str = Depends(guard)):
        r = await gw.readiness(deep=True)
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    @app.get("/metrics", summary="Prometheus text metrics", response_class=PlainTextResponse)
    async def metrics(client_id: str = Depends(guard)):
        return PlainTextResponse(gw.metrics.prometheus(), media_type="text/plain; version=0.0.4")

    @app.get("/api/metrics", summary="Usage metrics as JSON")
    async def metrics_json(client_id: str = Depends(guard)):
        return gw.metrics.snapshot()

    @app.get("/api/info", summary="Public service information")
    async def info():
        return {"name": "AI API Gateway", "author": __author__, "version": __version__,
                "auth_required": bool(settings.api_keys), "models": [m["model"] for m in gw.models()]}

    @app.get("/", include_in_schema=False)
    async def index():
        if not FRONTEND.exists():
            raise StarletteHTTPException(404, "frontend not found")
        return FileResponse(FRONTEND)

    return app


app = create_app(configure_logging=True)
