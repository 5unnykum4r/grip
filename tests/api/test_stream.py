"""Tests for the SSE streaming chat endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from grip.api.rate_limit import SlidingWindowRateLimiter
from grip.api.routers.chat import router
from grip.engines.types import AgentRunResult


def _build_test_app(engine_mock: AsyncMock) -> FastAPI:
    """Create a minimal FastAPI app with the chat router and mocked state."""
    app = FastAPI()
    app.include_router(router)

    app.state.auth_token = "test-token-123"
    app.state.engine = engine_mock
    app.state.ip_rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    app.state.token_rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    return app


def _make_engine_mock(
    response: str = "Hello!",
    iterations: int = 1,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls_made: list[str] | None = None,
) -> AsyncMock:
    mock = AsyncMock()
    mock.run.return_value = AgentRunResult(
        response=response,
        iterations=iterations,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_calls_made=tool_calls_made or [],
    )
    return mock


AUTH_HEADERS = {"Authorization": "Bearer test-token-123"}


class TestStreamEndpointContentType:
    def test_stream_endpoint_returns_event_stream(self):
        engine = _make_engine_mock()
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]


class TestStreamEndpointAuth:
    def test_stream_requires_auth(self):
        engine = _make_engine_mock()
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 401

    def test_stream_rejects_bad_token(self):
        engine = _make_engine_mock()
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401


class TestStreamEventStructure:
    def test_stream_events_include_start_message_done(self):
        engine = _make_engine_mock(
            response="Test response",
            iterations=2,
            prompt_tokens=15,
            completion_tokens=8,
            tool_calls_made=["web_search"],
        )
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Test input"},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200

        body = response.text
        assert "event: start" in body
        assert "event: message" in body
        assert "event: done" in body

    def test_stream_start_event_contains_session_key(self):
        engine = _make_engine_mock()
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello", "session_key": "test:session1"},
            headers=AUTH_HEADERS,
        )
        body = response.text
        assert '"session_key": "test:session1"' in body or '"session_key":"test:session1"' in body

    def test_stream_done_event_contains_usage(self):
        engine = _make_engine_mock(prompt_tokens=42, completion_tokens=17)
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"},
            headers=AUTH_HEADERS,
        )
        body = response.text
        assert "42" in body
        assert "17" in body

    def test_stream_error_event_on_engine_failure(self):
        engine = AsyncMock()
        engine.run.side_effect = RuntimeError("Engine blew up")
        app = _build_test_app(engine)
        client = TestClient(app)

        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200
        body = response.text
        assert "event: error" in body
        assert "Agent execution failed" in body
