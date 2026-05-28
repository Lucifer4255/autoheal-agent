"""Tests for agent capabilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import FunctionToolset

from agent.capabilities.github import GitHubCapability
from agent.capabilities.jaeger import JaegerCapability, extract_error_spans, query_traces
from agent.capabilities.loki import LokiCapability, build_logql, query_logs
from agent.capabilities.web_search import WebSearchCapability, web_search
from agent.models import AgentDeps


def make_deps(
    client: httpx.AsyncClient,
    **overrides: Any,
) -> AgentDeps:
    values = {
        "jaeger_url": "http://jaeger.local/jaeger/ui",
        "jaeger_auth": None,
        "loki_url": "http://loki.local",
        "loki_auth": None,
        "github_token": "ghp_test",
        "repo": "open-telemetry/opentelemetry-demo",
        "e2b_api_key": None,
        "tavily_key": "tvly_test",
        "service_name": "ad",
        "http_client": client,
    }
    values.update(overrides)
    return AgentDeps(**values)


def ctx_for(deps: AgentDeps) -> SimpleNamespace:
    return SimpleNamespace(deps=deps)


def test_capabilities_return_none_when_disabled() -> None:
    assert WebSearchCapability(enabled=False).get_toolset() is None
    assert JaegerCapability(enabled=False).get_toolset() is None
    assert LokiCapability(enabled=False).get_toolset() is None
    assert GitHubCapability(github_token=None, repo="owner/repo").get_toolset() is None
    assert GitHubCapability(github_token="ghp_test", repo=None).get_toolset() is None


def test_enabled_http_capabilities_return_function_toolsets() -> None:
    assert isinstance(WebSearchCapability(enabled=True).get_toolset(), FunctionToolset)
    assert isinstance(JaegerCapability(enabled=True).get_toolset(), FunctionToolset)
    assert isinstance(LokiCapability(enabled=True).get_toolset(), FunctionToolset)


def test_github_capability_returns_mcp_toolset_when_enabled() -> None:
    toolset = GitHubCapability(
        github_token="ghp_test",
        repo="open-telemetry/opentelemetry-demo",
    ).get_toolset()

    assert isinstance(toolset, MCPToolset)


@pytest.mark.asyncio
async def test_web_search_returns_tool_result_on_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, json={"error": "boom"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await web_search(ctx_for(make_deps(client)), query="otel error")
    finally:
        await client.aclose()

    assert result.tool_name == "web_search"
    assert result.success is False
    assert result.error


@pytest.mark.asyncio
async def test_jaeger_returns_tool_result_on_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, json={"error": "boom"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await query_traces(ctx_for(make_deps(client)), service="ad")
    finally:
        await client.aclose()

    assert result.tool_name == "query_traces"
    assert result.success is False
    assert result.error


@pytest.mark.asyncio
async def test_loki_returns_tool_result_on_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request, json={"error": "boom"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await query_logs(ctx_for(make_deps(client)), service="ad", query="error")
    finally:
        await client.aclose()

    assert result.tool_name == "query_logs"
    assert result.success is False
    assert result.error


@pytest.mark.asyncio
async def test_jaeger_includes_auth_header_and_uses_relative_api_path() -> None:
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        return httpx.Response(200, request=request, json={"data": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        deps = make_deps(client, jaeger_auth="Bearer jaeger-token")
        result = await query_traces(ctx_for(deps), service="ad")
    finally:
        await client.aclose()

    assert result.success is True
    assert seen["authorization"] == "Bearer jaeger-token"
    assert seen["url"].startswith("http://jaeger.local/jaeger/ui/api/traces")


@pytest.mark.asyncio
async def test_loki_includes_auth_header_and_compose_service_query() -> None:
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            request=request,
            json={"data": {"result": [{"stream": {}, "values": []}]}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        deps = make_deps(client, loki_auth="Bearer loki-token")
        result = await query_logs(ctx_for(deps), service="ad", query="NullPointerException")
    finally:
        await client.aclose()

    assert result.success is True
    assert seen["authorization"] == "Bearer loki-token"
    assert "query_range" in seen["url"]
    assert result.data["logql"] == '{compose_service="ad"} |= "NullPointerException"'


def test_build_logql_escapes_label_and_query() -> None:
    assert build_logql('ad"svc', 'error "boom"') == (
        '{compose_service="ad\\"svc"} |= "error \\"boom\\""'
    )


@pytest.mark.asyncio
async def test_extract_error_spans_normalizes_error_spans() -> None:
    trace = {
        "data": [
            {
                "traceID": "trace-1",
                "processes": {"p1": {"serviceName": "ad"}},
                "spans": [
                    {
                        "spanID": "span-1",
                        "processID": "p1",
                        "operationName": "GET /ads",
                        "duration": 123,
                        "startTime": 456,
                        "tags": [
                            {"key": "error", "value": True},
                            {"key": "code.filepath", "value": "src/ad.py"},
                            {"key": "code.lineno", "value": 42},
                        ],
                        "logs": [],
                    }
                ],
            }
        ]
    }

    client = httpx.AsyncClient()
    try:
        result = await extract_error_spans(ctx_for(make_deps(client)), trace)
    finally:
        await client.aclose()

    assert result.success is True
    assert result.data["error_span_count"] == 1
    assert result.data["error_spans"][0]["file_hint"] == "src/ad.py:42"
