"""Tests for the gateway core, adapters (against a local fake upstream) and, when FastAPI is installed, the HTTP API.

Runs under `python -m unittest` and `pytest`. Author: Rayyan Anwar
"""
import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import local_nlp
from app.config import ConfigError, Settings, Target, load_routes, load_settings
from app.errors import GatewayError, ProviderError
from app.gateway import Gateway
from app.metrics import Metrics
from app.models import Message, ChatRequest, estimate_tokens, parse_chat_request, sanitize_request_id
from app.providers import AnthropicProvider, LocalProvider, OpenAICompatProvider, http_json
from app.ratelimit import TokenBucketLimiter
from app.simulated import FailingProvider, SlowProvider

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
TEXT = ("The cache layer cut response time in half. Engineers said deployments are smoother. "
        "The dashboards were easy to read. The cache layer will expand to other services. Lunch was at noon.")


def req(text="hello world", model="m", **kw):
    return ChatRequest(model, (Message("user", text),), **kw)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestLocalNlp(unittest.TestCase):
    def test_summarize_keeps_order_and_is_shorter(self):
        s = local_nlp.summarize(TEXT, 2)
        self.assertEqual(len(local_nlp.split_sentences(s)), 2)
        self.assertNotIn("Lunch", s)
        self.assertEqual(s, local_nlp.summarize(TEXT, 2))
        self.assertLess(len(s), len(TEXT))

    def test_short_text_returned_whole(self):
        self.assertEqual(local_nlp.summarize("One. Two.", 3), "One. Two.")

    def test_sentiment_cases(self):
        f = local_nlp.sentiment
        self.assertEqual(f("The service was great and the staff were friendly.")["label"], "positive")
        self.assertEqual(f("Terrible, slow and buggy.")["label"], "negative")
        self.assertEqual(f("This is not good.")["label"], "negative")
        self.assertEqual(f("It was not bad at all.")["label"], "positive")
        self.assertEqual(f("It isn't helpful.")["label"], "negative")
        self.assertEqual(f("The meeting is at noon.")["label"], "neutral")
        self.assertGreater(f("very good")["positive_weight"], f("good")["positive_weight"])
        self.assertEqual(f("")["label"], "neutral")

    def test_keywords_and_stopwords(self):
        kw = dict(local_nlp.keywords("cache cache layer layer layer the and speed", 5))
        self.assertEqual(kw["layer"], 3)
        self.assertNotIn("the", kw)
        self.assertEqual(local_nlp.keywords("the and of", 3), [])

    def test_run_task_unknown_model(self):
        with self.assertRaises(ValueError):
            local_nlp.run_task("local/nope", "x", 10)
        self.assertIn("label", local_nlp.run_task("local/sentiment", "great", 10))


class TestRequestParsing(unittest.TestCase):
    good = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def test_valid(self):
        r = parse_chat_request({**self.good, "max_tokens": 10, "temperature": 0.5})
        self.assertEqual((r.model, r.max_tokens, r.temperature), ("m", 10, 0.5))

    def test_invalid_variants(self):
        m = self.good["messages"]
        bad = [None, [], {}, {**self.good, "stream": True}, {**self.good, "model": ""}, {**self.good, "model": 5},
               {**self.good, "messages": []}, {**self.good, "messages": "hi"},
               {**self.good, "messages": [{"role": "bot", "content": "x"}]},
               {**self.good, "messages": [{"role": "user"}]},
               {**self.good, "messages": [{"role": "user", "content": "  "}]},
               {**self.good, "messages": [{"role": "user", "content": "x" * 20_001}]},
               {**self.good, "messages": [{"role": "assistant", "content": "x"}]},
               {**self.good, "messages": m * 51}, {**self.good, "max_tokens": 0}, {**self.good, "max_tokens": True},
               {**self.good, "max_tokens": 5000}, {**self.good, "temperature": 3}, {**self.good, "temperature": "hot"}]
        for b in bad:
            with self.assertRaises(GatewayError, msg=str(b)[:80]) as cm:
                parse_chat_request(b)
            self.assertEqual(cm.exception.status, 422)

    def test_request_id_and_tokens(self):
        self.assertEqual(sanitize_request_id("abc-123"), "abc-123")
        self.assertNotEqual(sanitize_request_id("bad id\n"), "bad id\n")
        self.assertEqual(len(sanitize_request_id(None)), 32)
        self.assertEqual(estimate_tokens(""), 1)
        self.assertEqual(estimate_tokens("x" * 40), 10)


class TestRateLimiter(unittest.TestCase):
    def test_burst_then_deny_then_refill(self):
        clk = Clock()
        rl = TokenBucketLimiter(60, 3, clock=clk)
        decisions = [rl.check("a") for _ in range(4)]
        self.assertEqual([d.allowed for d in decisions], [True, True, True, False])
        self.assertEqual(decisions[0].remaining, 2)
        self.assertEqual(decisions[3].retry_after_s, 1)
        clk.t += 1.0
        self.assertTrue(rl.check("a").allowed)
        self.assertFalse(rl.check("a").allowed)

    def test_keys_are_independent_and_capped_at_burst(self):
        clk = Clock()
        rl = TokenBucketLimiter(60, 2, clock=clk)
        rl.check("a"); rl.check("a")
        self.assertFalse(rl.check("a").allowed)
        self.assertTrue(rl.check("b").allowed)
        clk.t += 1000
        self.assertEqual([rl.check("a").allowed for _ in range(3)], [True, True, False])

    def test_disabled_and_invalid(self):
        self.assertTrue(TokenBucketLimiter(0, 1).check("x").allowed)
        with self.assertRaises(ValueError):
            TokenBucketLimiter(10, 0)

    def test_bucket_count_is_bounded(self):
        rl = TokenBucketLimiter(60, 2, clock=Clock(), max_buckets=10)
        for i in range(50):
            rl.check(f"k{i}")
        self.assertLessEqual(len(rl._buckets), 10)

    def test_thread_safety_exact_count(self):
        rl = TokenBucketLimiter(60, 50, clock=Clock())
        results = []
        def worker():
            for _ in range(25):
                results.append(rl.check("shared").allowed)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        self.assertEqual(sum(results), 50)
        self.assertEqual(len(results), 200)


class TestMetrics(unittest.TestCase):
    def test_counters_histogram_and_prometheus(self):
        m = Metrics()
        m.inc("reqs_total", model="a", status="200"); m.inc("reqs_total", 2, model="a", status="200")
        m.inc("reqs_total", model='we"ird\n', status="500")
        self.assertEqual(m.total("reqs_total", model="a"), 3)
        self.assertEqual(m.total("reqs_total"), 4)
        m.observe("lat_ms", 7); m.observe("lat_ms", 300)
        text = m.prometheus()
        self.assertIn('reqs_total{model="a",status="200"} 3', text)
        self.assertIn('we\\"ird\\n', text)
        self.assertIn('lat_ms_bucket{le="10"} 1', text)
        self.assertIn('lat_ms_bucket{le="500"} 2', text)
        self.assertIn('lat_ms_bucket{le="+Inf"} 2', text)
        self.assertIn("lat_ms_count 2", text)
        snap = m.snapshot()
        json.dumps(snap)
        self.assertEqual(snap["histograms"]["lat_ms"]["count"], 2)


class TestConfig(unittest.TestCase):
    def test_defaults_and_env(self):
        s = load_settings({})
        self.assertEqual((s.api_keys, s.rate_limit_rpm, s.rate_limit_burst), ((), 60.0, 10))
        self.assertIn("local-summarizer", s.routes)
        s = load_settings({"GATEWAY_API_KEYS": "a, b ,", "GATEWAY_RATE_LIMIT_RPM": "30", "OPENAI_COMPAT_BASE_URL": "http://x/v1/"})
        self.assertEqual((s.api_keys, s.rate_limit_rpm, s.openai_base_url), (("a", "b"), 30.0, "http://x/v1"))

    def test_invalid_env(self):
        for env in ({"GATEWAY_RATE_LIMIT_RPM": "fast"}, {"GATEWAY_RATE_LIMIT_BURST": "0"},
                    {"GATEWAY_REQUEST_TIMEOUT_S": "0"}, {"GATEWAY_REQUEST_TIMEOUT_S": "9999"},
                    {"GATEWAY_ROUTES_FILE": "/nonexistent/routes.json"}):
            with self.assertRaises(ConfigError, msg=str(env)):
                load_settings(env)

    def test_routes_validation_and_file(self):
        ok = {"routes": [{"model": "a", "targets": [{"provider": "local", "model": "local/summarizer"}]}]}
        self.assertEqual(load_routes(ok)["a"][0], Target("local", "local/summarizer"))
        bads = [{}, {"routes": []}, {"routes": [{"model": "a", "targets": []}]},
                {"routes": [{"model": "a", "targets": [{"provider": "nope", "model": "x"}]}]},
                {"routes": [ok["routes"][0], ok["routes"][0]]}, {"routes": [{"targets": []}]}]
        for b in bads:
            with self.assertRaises(ConfigError, msg=str(b)):
                load_routes(b)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            p.write_text(json.dumps(ok))
            self.assertIn("a", load_settings({"GATEWAY_ROUTES_FILE": str(p)}).routes)
            p.write_text("{not json")
            with self.assertRaises(ConfigError):
                load_settings({"GATEWAY_ROUTES_FILE": str(p)})
        example = Path(__file__).resolve().parent.parent / "config" / "routes.example.json"
        self.assertIn("assistant", load_routes(json.loads(example.read_text())))


def settings_with(routes, **kw):
    return Settings(routes={k: tuple(Target(*t) for t in v) for k, v in routes.items()}, **kw)


class TestGateway(unittest.IsolatedAsyncioTestCase):
    def body(self, model="local-summarizer", text=TEXT):
        return {"model": model, "messages": [{"role": "user", "content": text}]}

    def local_gw(self, **kw):
        return Gateway(Settings(**kw), {"local": LocalProvider()})

    def test_authentication(self):
        open_gw = self.local_gw()
        self.assertEqual(open_gw.authenticate(None, "1.2.3.4"), "ip:1.2.3.4")
        gw = self.local_gw(api_keys=("secret-key",))
        for bad in (None, "", "wrong"):
            with self.assertRaises(GatewayError) as cm:
                gw.authenticate(bad)
            self.assertEqual(cm.exception.status, 401)
        cid = gw.authenticate("secret-key")
        self.assertTrue(cid.startswith("key:"))
        self.assertNotIn("secret", cid)
        self.assertEqual(gw.metrics.total("gateway_auth_failures_total"), 3)

    def test_rate_limit_error_has_headers(self):
        gw = Gateway(Settings(rate_limit_rpm=60, rate_limit_burst=2), {"local": LocalProvider()},
                     limiter=TokenBucketLimiter(60, 2, clock=Clock()))
        gw.check_rate("c"); gw.check_rate("c")
        with self.assertRaises(GatewayError) as cm:
            gw.check_rate("c")
        self.assertEqual(cm.exception.status, 429)
        self.assertEqual(cm.exception.headers["Retry-After"], "1")
        self.assertEqual(gw.metrics.total("gateway_rate_limited_total"), 1)

    async def test_local_routes_work(self):
        gw = self.local_gw()
        r = await gw.chat(self.body())
        self.assertEqual((r["provider"], r["fallback_used"], r["usage"]["estimated"]), ("local", False, True))
        self.assertNotIn("Lunch", r["content"])
        s = json.loads((await gw.chat(self.body("local-sentiment", "I love this great tool")))["content"])
        self.assertEqual(s["label"], "positive")
        k = (await gw.chat(self.body("local-keywords")))["content"]
        self.assertIn("cache", k)
        self.assertEqual(gw.metrics.total("gateway_requests_total", status="200"), 3)

    async def test_errors_unknown_model_and_invalid(self):
        gw = self.local_gw()
        with self.assertRaises(GatewayError) as cm:
            await gw.chat(self.body("nope"))
        self.assertEqual(cm.exception.status, 404)
        with self.assertRaises(GatewayError) as cm:
            await gw.chat({"model": "local-summarizer"})
        self.assertEqual(cm.exception.status, 422)
        self.assertEqual(gw.metrics.total("gateway_requests_total", model="unknown", status="404"), 1)

    async def test_fallback_on_failure(self):
        failing = FailingProvider(500)
        gw = Gateway(settings_with({"x": [("failing", "m"), ("local", "local/summarizer")]}),
                     {"failing": failing, "local": LocalProvider()})
        r = await gw.chat(self.body("x"))
        self.assertTrue(r["fallback_used"])
        self.assertEqual([a["outcome"] for a in r["attempts"]], ["error", "ok"])
        self.assertEqual(r["attempts"][0]["upstream_status"], 500)
        self.assertEqual(failing.calls, 1)
        self.assertEqual(gw.metrics.total("gateway_fallbacks_total"), 1)

    async def test_all_failed_is_502_and_all_timeouts_is_504(self):
        gw = Gateway(settings_with({"x": [("failing", "m")]}), {"failing": FailingProvider()})
        with self.assertRaises(GatewayError) as cm:
            await gw.chat(self.body("x"))
        self.assertEqual(cm.exception.status, 502)
        self.assertEqual(cm.exception.details["attempts"][0]["outcome"], "error")
        gw = Gateway(settings_with({"x": [("slow", "m")]}, request_timeout_s=0.1), {"slow": SlowProvider(5)})
        t0 = time.perf_counter()
        with self.assertRaises(GatewayError) as cm:
            await gw.chat(self.body("x"))
        self.assertEqual(cm.exception.status, 504)
        self.assertLess(time.perf_counter() - t0, 1.0)

    async def test_timeout_then_fallback(self):
        gw = Gateway(settings_with({"x": [("slow", "m"), ("local", "local/keywords")]}, request_timeout_s=0.1),
                     {"slow": SlowProvider(5), "local": LocalProvider()})
        r = await gw.chat(self.body("x"))
        self.assertEqual((r["provider"], r["attempts"][0]["outcome"]), ("local", "timeout"))

    async def test_not_configured_provider(self):
        providers = {"local": LocalProvider(), "openai_compat": OpenAICompatProvider("")}
        gw = Gateway(settings_with({"only": [("openai_compat", "m")], "both": [("openai_compat", "m"),
                                                                              ("local", "local/keywords")]}), providers)
        with self.assertRaises(GatewayError) as cm:
            await gw.chat(self.body("only"))
        self.assertEqual(cm.exception.status, 503)
        r = await gw.chat(self.body("both"))
        self.assertTrue(r["fallback_used"])
        ready = await gw.readiness()
        self.assertFalse(ready["ready"])
        self.assertEqual(ready["routes_without_working_provider"], ["only"])

    async def test_buggy_provider_does_not_crash_gateway(self):
        class Buggy(FailingProvider):
            async def complete(self, *a, **k):
                raise RuntimeError("bug")
        gw = Gateway(settings_with({"x": [("buggy", "m"), ("local", "local/keywords")]}),
                     {"buggy": Buggy(), "local": LocalProvider()})
        with self.assertLogs("gateway", level="ERROR"):
            r = await gw.chat(self.body("x"))
        self.assertEqual(r["provider"], "local")

    async def test_concurrent_requests(self):
        gw = self.local_gw()
        results = await asyncio.gather(*[gw.chat(self.body()) for _ in range(40)])
        self.assertEqual(len({r["id"] for r in results}), 40)
        self.assertEqual(gw.metrics.total("gateway_requests_total", status="200"), 40)


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):  # client-side timeouts close sockets mid-write; ignore
        pass


class FakeUpstream:
    def __init__(self):
        self.requests, self.mode, self.delay = [], "ok", 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None):
                data = raw if raw is not None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _record(self, body):
                outer.requests.append({"path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})

            def do_GET(self):
                self._record(None)
                self._send(200, {"data": []})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self._record(json.loads(self.rfile.read(n) or b"{}"))
                if outer.delay:
                    time.sleep(outer.delay)
                if outer.mode == "error500":
                    return self._send(500, {"error": "boom secret-detail"})
                if outer.mode == "badjson":
                    return self._send(200, raw=b"not json")
                if outer.mode == "malformed":
                    return self._send(200, {"unexpected": True})
                if self.path.endswith("/chat/completions"):
                    return self._send(200, {"choices": [{"message": {"role": "assistant", "content": "hello from upstream"}}],
                                            "usage": {"prompt_tokens": 11, "completion_tokens": 3}})
                if self.path.endswith("/v1/messages"):
                    return self._send(200, {"content": [{"type": "text", "text": "anthropic-style reply"}],
                                            "usage": {"input_tokens": 7, "output_tokens": 4}})
                self._send(404, {})

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestAdapters(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.up = FakeUpstream()
        self.addCleanup(self.up.close)

    async def test_openai_compat_success_request_shape_and_usage(self):
        p = OpenAICompatProvider(self.up.url + "/v1", "sk-test")
        r = await p.complete(ChatRequest("m", (Message("system", "be brief"), Message("user", "hi")), 50, 0.2), "up-model", 5)
        self.assertEqual((r.content, r.prompt_tokens, r.completion_tokens), ("hello from upstream", 11, 3))
        sent = self.up.requests[0]
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertEqual(sent["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(sent["body"]["model"], "up-model")
        self.assertEqual(sent["body"]["max_tokens"], 50)
        self.assertEqual(sent["body"]["temperature"], 0.2)
        self.assertEqual(sent["body"]["messages"][0], {"role": "system", "content": "be brief"})

    async def test_openai_compat_failure_modes(self):
        p = OpenAICompatProvider(self.up.url + "/v1")
        self.up.mode = "error500"
        with self.assertRaises(ProviderError) as cm:
            await p.complete(req(), "m", 5)
        self.assertEqual(cm.exception.status, 500)
        self.assertNotIn("secret-detail", str(cm.exception))
        for mode in ("badjson", "malformed"):
            self.up.mode = mode
            with self.assertRaises(ProviderError, msg=mode):
                await p.complete(req(), "m", 5)
        self.up.mode, self.up.delay = "ok", 1.0
        with self.assertRaises(ProviderError) as cm:
            await p.complete(req(), "m", 0.2)
        self.assertEqual(cm.exception.kind, "timeout")
        dead = OpenAICompatProvider("http://127.0.0.1:9/v1")
        with self.assertRaises(ProviderError) as cm:
            await dead.complete(req(), "m", 1)
        self.assertEqual(cm.exception.kind, "error")

    async def test_openai_compat_health(self):
        self.assertEqual((await OpenAICompatProvider("").health())["status"], "not_configured")
        p = OpenAICompatProvider(self.up.url + "/v1")
        self.assertEqual((await p.health())["status"], "configured")
        self.assertEqual((await p.health(deep=True))["status"], "ok")
        self.assertEqual((await OpenAICompatProvider("http://127.0.0.1:9/v1").health(deep=True, timeout=1))["status"], "unreachable")

    async def test_anthropic_adapter(self):
        p = AnthropicProvider("key-xyz", self.up.url, "2023-06-01")
        r = await p.complete(ChatRequest("m", (Message("system", "rule one"), Message("system", "rule two"),
                                               Message("user", "hi")), 40, 1.7), "up-model", 5)
        self.assertEqual((r.content, r.prompt_tokens, r.completion_tokens), ("anthropic-style reply", 7, 4))
        sent = self.up.requests[0]
        self.assertEqual(sent["path"], "/v1/messages")
        self.assertEqual(sent["headers"]["x-api-key"], "key-xyz")
        self.assertEqual(sent["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(sent["body"]["system"], "rule one\n\nrule two")
        self.assertEqual(sent["body"]["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(sent["body"]["temperature"], 1.0)
        with self.assertRaises(ProviderError):
            await p.complete(ChatRequest("m", (Message("assistant", "x"), Message("user", "y"))), "m", 5)
        self.assertFalse(AnthropicProvider("").configured)

    async def test_http_json_rejects_non_http_schemes(self):
        with self.assertRaises(ProviderError):
            http_json("GET", "file:///etc/passwd", {}, None, 1)

    async def test_gateway_falls_back_from_failing_upstream_to_local(self):
        self.up.mode = "error500"
        gw = Gateway(settings_with({"assistant": [("openai_compat", "up"), ("local", "local/summarizer")]}),
                     {"openai_compat": OpenAICompatProvider(self.up.url + "/v1"), "local": LocalProvider()})
        r = await gw.chat({"model": "assistant", "messages": [{"role": "user", "content": TEXT}]})
        self.assertEqual((r["provider"], r["fallback_used"]), ("local", True))
        self.assertEqual(r["attempts"][0]["upstream_status"], 500)
        self.up.mode = "ok"
        r = await gw.chat({"model": "assistant", "messages": [{"role": "user", "content": "hi there"}]})
        self.assertEqual((r["provider"], r["usage"]["estimated"], r["usage"]["total_tokens"]), ("openai_compat", False, 14))


try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    HAVE_API = True
except ImportError:
    HAVE_API = False


@unittest.skipUnless(HAVE_API, "fastapi/httpx not installed - API tests are skipped, not passed")
class TestApi(unittest.TestCase):
    def client(self, **kw):
        from fastapi.testclient import TestClient
        from app.main import create_app
        return TestClient(create_app(Settings(**kw), {"local": LocalProvider()}))

    chat = {"model": "local-sentiment", "messages": [{"role": "user", "content": "great and fast"}]}

    def test_public_endpoints(self):
        c = self.client(api_keys=("k1",))
        self.assertEqual(c.get("/health").json()["author"], "Rayyan Anwar")
        self.assertEqual(c.get("/api/info").json()["auth_required"], True)
        self.assertIn("Rayyan Anwar", c.get("/").text)
        self.assertEqual(c.get("/ready").status_code, 200)
        self.assertEqual(c.get("/openapi.json").json()["info"]["contact"]["name"], "Rayyan Anwar")

    def test_auth_required_and_checked_before_validation(self):
        c = self.client(api_keys=("k1",))
        self.assertEqual(c.post("/v1/chat", json=self.chat).status_code, 401)
        self.assertEqual(c.post("/v1/chat", json={"bad": 1}).status_code, 401)
        self.assertEqual(c.post("/v1/chat", json=self.chat, headers={"Authorization": "Bearer nope"}).status_code, 401)
        ok = c.post("/v1/chat", json=self.chat, headers={"Authorization": "Bearer k1"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(c.post("/v1/chat", json=self.chat, headers={"X-API-Key": "k1"}).status_code, 200)

    def test_chat_headers_and_request_id(self):
        c = self.client(rate_limit_burst=5)
        r = c.post("/v1/chat", json=self.chat, headers={"X-Request-ID": "trace-1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["x-request-id"], "trace-1")
        self.assertEqual(r.headers["x-ratelimit-remaining"], "4")
        self.assertEqual(json.loads(r.json()["content"])["label"], "positive")
        bad_id = c.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        self.assertNotEqual(bad_id.headers["x-request-id"], "bad id with spaces")

    def test_validation_errors_are_consistent(self):
        c = self.client()
        for body in ({}, {"model": "local-sentiment"}, {**self.chat, "stream": True},
                     {"model": "x", "messages": [{"role": "user", "content": ""}]}):
            r = c.post("/v1/chat", json=body)
            self.assertEqual(r.status_code, 422, msg=str(body))
            self.assertEqual(r.json()["error"]["code"], "invalid_request")
        self.assertEqual(c.post("/v1/chat", content=b"{oops", headers={"content-type": "application/json"}).status_code, 422)
        self.assertEqual(c.post("/v1/chat", json={**self.chat, "model": "nope"}).status_code, 404)
        self.assertEqual(c.get("/nope").json()["error"]["code"], "http_error")

    def test_rate_limit_returns_429_with_retry_after(self):
        c = self.client(rate_limit_rpm=1, rate_limit_burst=2)
        codes = [c.post("/v1/chat", json=self.chat).status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 429, 429])
        r = c.post("/v1/chat", json=self.chat)
        self.assertIn("retry-after", r.headers)
        self.assertEqual(r.json()["error"]["code"], "rate_limited")

    def test_models_and_metrics(self):
        c = self.client()
        self.assertTrue(any(m["model"] == "local-summarizer" for m in c.get("/v1/models").json()["data"]))
        c.post("/v1/chat", json=self.chat)
        text = c.get("/metrics").text
        self.assertIn("gateway_requests_total", text)
        self.assertIn("gateway_request_latency_ms_count", text)
        self.assertTrue(c.get("/api/metrics").json()["counters"])


if __name__ == "__main__":
    unittest.main()
