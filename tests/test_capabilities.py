"""Tests for agent capabilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import FunctionToolset

from agent import sandbox_subagent as sandbox_module
from agent.capabilities.github import GitHubCapability
from agent.capabilities.jaeger import JaegerCapability, extract_error_spans, query_traces
from agent.capabilities.loki import LokiCapability, build_logql, query_logs
from agent.capabilities.sandbox import SandboxCapability, reproduce_in_sandbox
from agent.capabilities.web_search import WebSearchCapability, web_search
from agent.models import AgentDeps, SandboxResult
from agent.registry import build_capabilities


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
    return SimpleNamespace(deps=deps, usage=None)


def test_capabilities_return_none_when_disabled() -> None:
    assert WebSearchCapability(enabled=False).get_toolset() is None
    assert JaegerCapability(enabled=False).get_toolset() is None
    assert LokiCapability(enabled=False).get_toolset() is None
    assert GitHubCapability(github_token=None, repo="owner/repo").get_toolset() is None
    assert GitHubCapability(github_token="ghp_test", repo=None).get_toolset() is None
    assert (
        SandboxCapability(
            e2b_api_key=None, github_token="ghp_test", repo="owner/repo"
        ).get_toolset()
        is None
    )
    assert (
        SandboxCapability(
            e2b_api_key="e2b_test", github_token=None, repo="owner/repo"
        ).get_toolset()
        is None
    )
    assert (
        SandboxCapability(e2b_api_key="e2b_test", github_token="ghp_test", repo=None).get_toolset()
        is None
    )


def test_enabled_http_capabilities_return_function_toolsets() -> None:
    assert isinstance(WebSearchCapability(enabled=True).get_toolset(), FunctionToolset)
    assert isinstance(JaegerCapability(enabled=True).get_toolset(), FunctionToolset)
    assert isinstance(LokiCapability(enabled=True).get_toolset(), FunctionToolset)
    assert isinstance(
        SandboxCapability(
            e2b_api_key="e2b_test",
            github_token="ghp_test",
            repo="owner/repo",
        ).get_toolset(),
        FunctionToolset,
    )


def test_github_capability_returns_mcp_toolset_when_enabled() -> None:
    toolset = GitHubCapability(
        github_token="ghp_test",
        repo="open-telemetry/opentelemetry-demo",
    ).get_toolset()

    assert isinstance(toolset, MCPToolset)


@pytest.mark.asyncio
async def test_registry_includes_sandbox_capability_when_configured() -> None:
    client = httpx.AsyncClient()
    try:
        deps = make_deps(client, e2b_api_key="e2b_test")
        capabilities = build_capabilities(deps)
    finally:
        await client.aclose()

    assert any(isinstance(capability, SandboxCapability) for capability in capabilities)


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "error_type", "file_path", "expected_reason"),
    [
        ({}, "infra", "src/ad.py", "not sandbox-friendly"),
        ({"e2b_api_key": None}, "runtime_error", "src/ad.py", "E2B API key"),
        ({"github_token": None}, "runtime_error", "src/ad.py", "GitHub token"),
        ({"repo": None}, "runtime_error", "src/ad.py", "Repository"),
        ({}, "runtime_error", "   ", "suspect file path"),
    ],
)
async def test_reproduce_in_sandbox_returns_skipped_results(
    overrides: dict[str, Any],
    error_type: str,
    file_path: str,
    expected_reason: str,
) -> None:
    client = httpx.AsyncClient()
    try:
        deps_overrides = {"e2b_api_key": "e2b_test", **overrides}
        deps = make_deps(client, **deps_overrides)
        result = await reproduce_in_sandbox(
            ctx_for(deps),
            hypothesis="ad service crashes on empty response",
            error_type=error_type,
            file_path=file_path,
        )
    finally:
        await client.aclose()

    assert result.reproduced is False
    assert result.attempts == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.repro_script == ""
    assert result.skip_reason is not None
    assert expected_reason in result.skip_reason


@pytest.mark.asyncio
async def test_reproduce_in_sandbox_delegates_to_subagent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    expected = SandboxResult(
        reproduced=True,
        confirmed_file="src/ad.py",
        confirmed_line=42,
        stdout="boom",
        stderr="",
        exit_code=1,
        repro_script="python repro.py",
        attempts=1,
    )

    class FakeRunResult:
        output = expected

    class FakeSandboxSubagent:
        async def run(self, *args: Any, **kwargs: Any) -> FakeRunResult:
            calls["args"] = args
            calls["kwargs"] = kwargs
            return FakeRunResult()

    client = httpx.AsyncClient()
    try:
        deps = make_deps(client, e2b_api_key="e2b_test")
        monkeypatch.setattr(
            "agent.capabilities.sandbox.sandbox_subagent",
            FakeSandboxSubagent(),
        )

        result = await reproduce_in_sandbox(
            ctx_for(deps),
            hypothesis="ad service crashes on empty response",
            error_type="runtime_error",
            file_path="src/ad.py",
        )
    finally:
        await client.aclose()

    assert result == expected
    assert calls["kwargs"]["deps"] is deps
    assert calls["kwargs"]["usage"] is None
    assert calls["kwargs"]["usage_limits"].request_limit == 5
    assert "src/ad.py" in calls["args"][0]


@pytest.mark.asyncio
async def test_e2b_tools_handle_create_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("create failed")

    monkeypatch.setattr(sandbox_module.Sandbox, "create", staticmethod(fake_create))

    client = httpx.AsyncClient()
    try:
        result = await sandbox_module.create_sandbox(
            ctx_for(make_deps(client, e2b_api_key="e2b_test"))
        )
    finally:
        await client.aclose()

    assert result.tool_name == "create_sandbox"
    assert result.success is False
    assert "create failed" in (result.error or "")


@pytest.mark.asyncio
async def test_e2b_tools_run_read_and_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    killed = {"value": False}

    class FakeCommandResult:
        stdout = "hello"
        stderr = ""
        exit_code = 0

    class FakeCommands:
        def run(self, command: str, timeout: int) -> FakeCommandResult:
            assert command == "python repro.py"
            assert timeout == 30
            return FakeCommandResult()

    class FakeSandbox:
        commands = FakeCommands()

        def kill(self) -> None:
            killed["value"] = True

    def fake_create(*args: Any, **kwargs: Any) -> FakeSandbox:
        assert kwargs["api_key"] == "e2b_test"
        return FakeSandbox()

    monkeypatch.setattr(sandbox_module.Sandbox, "create", staticmethod(fake_create))

    client = httpx.AsyncClient()
    try:
        ctx = ctx_for(make_deps(client, e2b_api_key="e2b_test"))
        create_result = await sandbox_module.create_sandbox(ctx)
        sandbox_id = create_result.data["sandbox_id"]
        run_result = await sandbox_module.run_command(
            ctx,
            sandbox_id=sandbox_id,
            command="python repro.py",
            timeout_seconds=30,
        )
        output_result = await sandbox_module.read_output(ctx, sandbox_id=sandbox_id)
        terminate_result = await sandbox_module.terminate(ctx, sandbox_id=sandbox_id)
    finally:
        await client.aclose()

    assert create_result.success is True
    assert run_result.data["stdout"] == "hello"
    # history now contains full per-command records
    assert output_result.data["command_count"] == 1
    assert output_result.data["history"][0]["command"] == "python repro.py"
    assert terminate_result.data["terminated"] is True
    assert killed["value"] is True
