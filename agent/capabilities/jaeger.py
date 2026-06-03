"""Jaeger trace query capability."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent.models import AgentDeps, ToolResult


@dataclass
class JaegerCapability(AbstractCapability[AgentDeps]):
    """Provides read-only Jaeger trace lookup tools."""

    enabled: bool

    def get_toolset(self) -> AgentToolset[AgentDeps] | None:
        if not self.enabled:
            return None
        return FunctionToolset(
            tools=[list_trace_services, query_traces, get_trace, extract_error_spans]
        )


async def list_trace_services(ctx: RunContext[AgentDeps]) -> ToolResult:
    """List the exact `service.name` values Jaeger knows about.

    Call this BEFORE query_traces so you use real names instead of guessing — the OTel
    service.name is often shorter than the code/module name (e.g. `ad`, not `adservice`).
    """
    if not ctx.deps.jaeger_url:
        return _failure("list_trace_services", "Jaeger URL is not configured.")

    try:
        response = await ctx.deps.http_client.get(
            _api_url(ctx.deps.jaeger_url, "/api/services"),
            headers=_headers(ctx.deps.jaeger_auth),
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return _failure("list_trace_services", str(exc))

    services = payload.get("data") or []
    return ToolResult(
        tool_name="list_trace_services",
        success=True,
        data={"services": sorted(services), "count": len(services)},
    )


async def query_traces(
    ctx: RunContext[AgentDeps],
    service: str,
    time_window_minutes: int = 10,
    limit: int = 20,
) -> ToolResult:
    """Query recent traces for a service.

    `service` is the OpenTelemetry `service.name` attribute and rarely matches the
    class or module name visible inside log lines. If you guess and get zero traces,
    do NOT keep guessing variants — read the repo's `docker-compose.yml` (or
    `compose.yaml`) via the GitHub tools to see the real service names.

    Returns a summary of each recent trace (services touched and which spans look like
    errors). To dig into a specific trace, pass it to `extract_error_spans`, or fetch the
    full trace with `get_trace`.
    """
    if not ctx.deps.jaeger_url:
        return _failure("query_traces", "Jaeger URL is not configured.")

    end_us = int(time.time() * 1_000_000)
    start_us = end_us - (time_window_minutes * 60 * 1_000_000)

    try:
        response = await ctx.deps.http_client.get(
            _api_url(ctx.deps.jaeger_url, "/api/traces"),
            params={
                "service": service,
                "start": start_us,
                "end": end_us,
                "limit": limit,
            },
            headers=_headers(ctx.deps.jaeger_auth),
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return _failure("query_traces", str(exc))

    traces = payload.get("data", [])

    # Zero traces means the service.name is almost certainly wrong (not that the service
    # is healthy). Return a FAILURE so it doesn't count as evidence, and steer the agent
    # toward finding the real name.
    if not traces:
        return ToolResult(
            tool_name="query_traces",
            success=False,
            data={
                "service": service,
                "time_window_minutes": time_window_minutes,
                "trace_count": 0,
                "traces": [],
            },
            error=(
                f'No traces for service.name="{service}" in the last {time_window_minutes}m. '
                f"Two likely causes: (1) the time window is too narrow — retry with a larger "
                f"time_window_minutes (e.g. 30 or 60); (2) the name is wrong — the OTel "
                f"service.name rarely matches the class/module name, so call list_trace_services "
                f"for the exact value. Widen the window first, then correct the name. Do NOT "
                f"keep guessing variants."
            ),
        )

    return ToolResult(
        tool_name="query_traces",
        success=True,
        data={
            "service": service,
            "time_window_minutes": time_window_minutes,
            "trace_count": len(traces),
            "traces": [_summarize_trace(trace) for trace in traces],
        },
    )


async def get_trace(ctx: RunContext[AgentDeps], trace_id: str) -> ToolResult:
    """Fetch one trace by ID."""
    if not ctx.deps.jaeger_url:
        return _failure("get_trace", "Jaeger URL is not configured.")

    try:
        response = await ctx.deps.http_client.get(
            _api_url(ctx.deps.jaeger_url, f"/api/traces/{trace_id}"),
            headers=_headers(ctx.deps.jaeger_auth),
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return _failure("get_trace", str(exc))

    return ToolResult(
        tool_name="get_trace",
        success=True,
        data={"trace_id": trace_id, "trace": payload},
    )


async def extract_error_spans(
    ctx: RunContext[AgentDeps],
    trace: dict[str, Any],
) -> ToolResult:
    """Extract spans that Jaeger marks as errors or that contain error-like logs."""
    del ctx
    traces = trace.get("data", trace)
    if isinstance(traces, dict):
        traces = [traces]

    error_spans: list[dict[str, Any]] = []
    for trace_item in traces if isinstance(traces, list) else []:
        error_spans.extend(_error_spans_for_trace(trace_item))

    return ToolResult(
        tool_name="extract_error_spans",
        success=True,
        data={"error_span_count": len(error_spans), "error_spans": error_spans},
    )


def _error_spans_for_trace(trace_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the error spans in one Jaeger trace, each enriched with a file hint.

    Used by `extract_error_spans` to surface the failing spans in a trace.
    """
    processes = trace_item.get("processes", {})
    spans: list[dict[str, Any]] = []
    for span in trace_item.get("spans", []):
        tags = _tags_to_dict(span.get("tags", []))
        logs = span.get("logs", [])
        if not _is_error_span(tags, logs):
            continue
        process = processes.get(span.get("processID"), {})
        spans.append(
            {
                "trace_id": trace_item.get("traceID"),
                "span_id": span.get("spanID"),
                "operation": span.get("operationName"),
                "service": process.get("serviceName"),
                "duration": span.get("duration"),
                "start_time": span.get("startTime"),
                "tags": tags,
                "logs": logs,
                "file_hint": _file_hint(tags, logs),
            }
        )
    return spans


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(auth: str | None) -> dict[str, str]:
    return {"Authorization": auth} if auth else {}


def _failure(tool_name: str, error: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=False, data={}, error=error)


def _summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})
    return {
        "trace_id": trace.get("traceID"),
        "span_count": len(spans),
        "services": sorted(
            {
                process.get("serviceName")
                for process in processes.values()
                if process.get("serviceName")
            }
        ),
        "error_spans": [
            span.get("spanID")
            for span in spans
            if _is_error_span(_tags_to_dict(span.get("tags", [])), span.get("logs", []))
        ],
    }


def _tags_to_dict(tags: list[dict[str, Any]]) -> dict[str, Any]:
    return {tag.get("key"): tag.get("value") for tag in tags if tag.get("key")}


def _is_error_span(tags: dict[str, Any], logs: list[dict[str, Any]]) -> bool:
    # Explicit error=true tag (OTel convention)
    if tags.get("error") in {True, "true", "True", 1, "1"}:
        return True
    # OTel status code ERROR (otel.status_description alone is set on OK spans too)
    if tags.get("otel.status_code") in {"ERROR", "error", "2"}:
        return True
    # exception.type is only set on actual exception spans
    if "exception.type" in tags or "exception.message" in tags:
        return True
    # Scan log fields structurally — avoid str(log) which matches service/op names
    for log in logs:
        fields = _tags_to_dict(log.get("fields", []) if isinstance(log, dict) else [])
        level = str(fields.get("level", fields.get("severity", ""))).lower()
        if level in {"error", "fatal", "critical"}:
            return True
        if fields.get("exception.type") or fields.get("error"):
            return True
    return False


def _file_hint(tags: dict[str, Any], logs: list[dict[str, Any]]) -> str | None:
    candidate_keys = ("code.filepath", "code.file.path", "file", "filename", "source.file")
    for key in candidate_keys:
        value = tags.get(key)
        if isinstance(value, str) and value:
            line = tags.get("code.lineno") or tags.get("line")
            return f"{value}:{line}" if line else value

    for log in logs:
        fields = log.get("fields", []) if isinstance(log, dict) else []
        field_tags = _tags_to_dict(fields)
        for key in candidate_keys:
            value = field_tags.get(key)
            if isinstance(value, str) and value:
                return value
    return None
