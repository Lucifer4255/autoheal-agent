"""Tests for server endpoints."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

import server
from agent.models import HealResult, IssueContext, RootCause


def parse_sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for chunk in body.strip().split("\n\n"):
        if chunk.startswith("data: "):
            events.append(json.loads(chunk.removeprefix("data: ")))
    return events


def sample_heal_result() -> HealResult:
    return HealResult(
        issue_summary="ad service errors",
        investigation_steps=[],
        root_cause=RootCause(
            description="runtime panic in ad",
            confidence=0.88,
            evidence=["jaeger trace"],
            error_type="runtime_error",
        ),
        recommended_fix="Fix nil handling in ad handler.",
        action_taken="explained",
        tools_used=["jaeger"],
        tools_unavailable=["sandbox"],
    )


@pytest.fixture(autouse=True)
def clear_sessions() -> None:
    server.sessions.clear()
    yield
    server.sessions.clear()


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_chat_github_url_sets_repo(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GitHub URL anywhere in the message is parsed into deps.repo."""
    seen: dict[str, Any] = {}

    async def fake_stream(issue, deps, message_history=None):  # noqa: ANN001
        seen["repo"] = deps.repo
        yield {"type": "result", "output": sample_heal_result(), "messages": []}

    monkeypatch.setattr(server, "stream_investigate", fake_stream)

    response = await client.post(
        "/chat",
        json={"message": "https://github.com/open-telemetry/opentelemetry-demo ad service is crashing"},
    )

    assert response.status_code == 200
    assert response.headers["X-Session-Id"]
    assert seen["repo"] == "open-telemetry/opentelemetry-demo"


@pytest.mark.asyncio
async def test_chat_issue_text_returns_final_event(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream(issue: IssueContext, deps, message_history=None):  # noqa: ANN001
        yield {"type": "step", "round": 1, "tool": "jaeger", "result_summary": "Calling jaeger", "confidence_after": 0.0}
        yield {"type": "result", "output": sample_heal_result(), "messages": []}

    monkeypatch.setattr(server, "stream_investigate", fake_stream)

    response = await client.post(
        "/chat",
        json={"message": "ad service is throwing runtime panics"},
    )

    events = parse_sse_events(response.text)
    types = [event["type"] for event in events]
    assert "step" in types
    assert "final" in types
    final = next(event for event in events if event["type"] == "final")
    assert final["result"]["root_cause"]["error_type"] == "runtime_error"


@pytest.mark.asyncio
async def test_chat_mixed_message_applies_overrides_then_investigates(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_stream(issue: IssueContext, deps, message_history=None):  # noqa: ANN001
        seen["service_name"] = issue.service_name
        seen["repo"] = deps.repo
        yield {"type": "result", "output": sample_heal_result(), "messages": []}

    monkeypatch.setattr(server, "stream_investigate", fake_stream)

    response = await client.post(
        "/chat",
        json={
            "message": (
                "https://github.com/open-telemetry/opentelemetry-demo "
                "ad service is throwing runtime panics"
            )
        },
    )

    events = parse_sse_events(response.text)
    assert "final" in [event["type"] for event in events]
    assert seen["repo"] == "open-telemetry/opentelemetry-demo"


@pytest.mark.asyncio
async def test_chat_reuses_existing_session(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream(issue, deps, message_history=None):  # noqa: ANN001
        yield {"type": "result", "output": sample_heal_result(), "messages": []}

    monkeypatch.setattr(server, "stream_investigate", fake_stream)

    first = await client.post(
        "/chat",
        json={"message": "https://github.com/open-telemetry/opentelemetry-demo ad is crashing"},
    )
    session_id = first.headers["X-Session-Id"]

    second = await client.post(
        "/chat",
        json={"message": "same issue still happening", "session_id": session_id},
    )

    assert second.headers["X-Session-Id"] == session_id
    deps = server.sessions[session_id]
    assert deps.repo == "open-telemetry/opentelemetry-demo"


@pytest.mark.asyncio
async def test_chat_investigation_error_emits_error_event(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream(issue: IssueContext, deps, message_history=None):  # noqa: ANN001
        raise RuntimeError("investigation failed")
        yield  # make this an async generator

    monkeypatch.setattr(server, "stream_investigate", fake_stream)

    response = await client.post(
        "/chat",
        json={"message": "ad service is failing"},
    )

    events = parse_sse_events(response.text)
    error = next(event for event in events if event["type"] == "error")
    assert "investigation failed" in error["message"]
