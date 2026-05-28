"""Tests for agent.models and config-backed AgentDeps."""

from __future__ import annotations

import os

import httpx
import pytest
from pydantic import ValidationError

from agent.models import AgentDeps, RootCause, normalize_repo


@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JAEGER_URL", "http://localhost:8080/jaeger/ui")
    monkeypatch.setenv("LOKI_URL", "http://localhost:3100")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly_test")
    monkeypatch.setenv("E2B_API_KEY", "e2b_test")


def test_root_cause_confidence_bounds() -> None:
    RootCause(
        description="test",
        confidence=0.5,
        evidence=["x"],
        error_type="unknown",
    )

    with pytest.raises(ValidationError):
        RootCause(
            description="test",
            confidence=1.5,
            evidence=["x"],
            error_type="unknown",
        )


def test_from_env_reads_defaults(env_vars: None) -> None:
    deps = AgentDeps.from_env(http_client=httpx.AsyncClient())
    try:
        assert deps.jaeger_url == "http://localhost:8080/jaeger/ui"
        assert deps.loki_url == "http://localhost:3100"
        assert deps.github_token == "ghp_test"
        assert deps.tavily_key == "tvly_test"
        assert deps.e2b_api_key == "e2b_test"
        assert deps.repo is None
        assert deps.service_name is None
    finally:
        os.environ.pop("JAEGER_URL", None)


def test_apply_overrides_merges_setup_chat_values(env_vars: None) -> None:
    deps = AgentDeps.from_env(http_client=httpx.AsyncClient()).apply_overrides(
        {
            "repo": "https://github.com/open-telemetry/opentelemetry-demo/",
            "service_name": "ad",
        }
    )
    assert deps.repo == "open-telemetry/opentelemetry-demo"
    assert deps.service_name == "ad"


def test_normalize_repo_accepts_slug() -> None:
    assert normalize_repo("open-telemetry/opentelemetry-demo") == "open-telemetry/opentelemetry-demo"


def test_capability_lists(env_vars: None) -> None:
    base = AgentDeps.from_env(http_client=httpx.AsyncClient())
    partial = base.apply_overrides({"repo": "open-telemetry/opentelemetry-demo"})

    assert "jaeger" in partial.configured_capabilities()
    assert "loki" in partial.configured_capabilities()
    assert "github" in partial.configured_capabilities()
    assert "web_search" in partial.configured_capabilities()
    assert "sandbox" in partial.configured_capabilities()
    assert partial.unavailable_capabilities() == []
    assert partial.needs_input() == []

    missing_repo = base.apply_overrides({})
    assert "github" not in missing_repo.configured_capabilities()
    assert "github" in missing_repo.unavailable_capabilities()
    assert missing_repo.needs_input() == ["repo"]
