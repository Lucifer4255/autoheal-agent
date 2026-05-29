"""Tests for agent.loop."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import config
from agent.fingerprint import FingerprintMatch
from agent.loop import investigate
from agent.models import AgentDeps, HealResult, IssueContext, RootCause


def make_deps(client: httpx.AsyncClient | None = None) -> AgentDeps:
    return AgentDeps(
        jaeger_url="http://jaeger.local/jaeger/ui",
        jaeger_auth=None,
        loki_url="http://loki.local",
        loki_auth=None,
        github_token="ghp_test",
        repo="open-telemetry/opentelemetry-demo",
        e2b_api_key=None,
        service_name="ad",
        http_client=client or httpx.AsyncClient(),
    )


def sample_heal_result() -> HealResult:
    return HealResult(
        issue_summary="ad service panic",
        investigation_steps=[],
        root_cause=RootCause(
            description="nil pointer in ad handler",
            confidence=0.9,
            evidence=["trace span error"],
            error_type="null_pointer",
        ),
        recommended_fix="Add nil check before dereference.",
        action_taken="explained",
        tools_used=["jaeger"],
        tools_unavailable=["sandbox"],
    )


@pytest.mark.asyncio
async def test_investigate_includes_fingerprint_in_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fp = FingerprintMatch(
        pattern="nil_pointer",
        confidence=0.90,
        error_type="null_pointer",
        hypothesis="Code dereferenced a null or nil value.",
    )

    class FakeRunResult:
        output = sample_heal_result()

    async def fake_run(prompt: str, **kwargs: Any) -> FakeRunResult:
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return FakeRunResult()

    monkeypatch.setattr("agent.loop.fingerprint", lambda _text: fp)
    monkeypatch.setattr("agent.loop.agent.run", fake_run)

    client = httpx.AsyncClient()
    try:
        issue = IssueContext(description="panic: nil pointer dereference in ad")
        result = await investigate(issue, make_deps(client))
    finally:
        await client.aclose()

    assert result.issue_summary == "ad service panic"
    assert "Pre-investigation hypothesis (unverified)" in captured["prompt"]
    assert "nil_pointer" in captured["prompt"]
    assert captured["kwargs"]["deps"].service_name == "ad"
    assert "usage_limits" in captured["kwargs"]


@pytest.mark.asyncio
async def test_high_confidence_fingerprint_still_calls_agent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fp = FingerprintMatch(
        pattern="nil_pointer",
        confidence=config.FASTPATH_CONFIDENCE,
        error_type="null_pointer",
        hypothesis="Code dereferenced a null or nil value.",
    )

    class FakeRunResult:
        output = sample_heal_result()

    async def fake_run(prompt: str, **kwargs: Any) -> FakeRunResult:
        calls.append(prompt)
        return FakeRunResult()

    monkeypatch.setattr("agent.loop.fingerprint", lambda _text: fp)
    monkeypatch.setattr("agent.loop.agent.run", fake_run)

    client = httpx.AsyncClient()
    try:
        await investigate(
            IssueContext(description="NullPointerException in ad service"),
            make_deps(client),
        )
    finally:
        await client.aclose()

    assert len(calls) == 1
    assert "High-confidence hint" in calls[0]
