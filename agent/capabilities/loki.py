"""Loki log query capability."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent.models import AgentDeps, ToolResult


@dataclass
class LokiCapability(AbstractCapability[AgentDeps]):
    """Provides read-only Loki log query tools."""

    enabled: bool

    def get_toolset(self) -> AgentToolset[AgentDeps] | None:
        if not self.enabled:
            return None
        return FunctionToolset(tools=[query_logs, get_log_context])


async def query_logs(
    ctx: RunContext[AgentDeps],
    service: str,
    query: str,
    time_window_minutes: int = 10,
    limit: int = 100,
) -> ToolResult:
    """Query Loki logs for a service and a search term.

    `service` is the value of the `compose_service` label and rarely matches the
    class or module name visible inside log lines. If you guess and get an empty
    result, do NOT keep guessing variants — read the repo's `docker-compose.yml`
    (or `compose.yaml`) via the GitHub tools to see the real service names.
    """
    if not ctx.deps.loki_url:
        return _failure("query_logs", "Loki URL is not configured.")

    end_ns = int(time.time() * 1_000_000_000)
    start_ns = end_ns - (time_window_minutes * 60 * 1_000_000_000)
    logql = build_logql(service, query)

    try:
        response = await ctx.deps.http_client.get(
            _api_url(ctx.deps.loki_url, "/loki/api/v1/query_range"),
            params={
                "query": logql,
                "start": start_ns,
                "end": end_ns,
                "limit": limit,
                "direction": "backward",
            },
            headers=_headers(ctx.deps.loki_auth),
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return _failure("query_logs", str(exc))

    return ToolResult(
        tool_name="query_logs",
        success=True,
        data={
            "service": service,
            "query": query,
            "logql": logql,
            "entries": _flatten_streams(payload),
        },
    )


async def get_log_context(
    ctx: RunContext[AgentDeps],
    timestamp: str,
    service: str,
    lines: int = 20,
) -> ToolResult:
    """Fetch log lines around a timestamp for a service.

    `service` must be a valid `compose_service` label value (see `query_logs`).
    `timestamp` accepts epoch seconds/ms/µs/ns or ISO-8601 (e.g. "2026-05-28T19:26:40Z").
    """
    if not ctx.deps.loki_url:
        return _failure("get_log_context", "Loki URL is not configured.")

    center_ns = _parse_loki_timestamp(timestamp)
    logql = f'{{compose_service="{_escape_label(service)}"}}'

    try:
        response = await ctx.deps.http_client.get(
            _api_url(ctx.deps.loki_url, "/loki/api/v1/query"),
            params={
                "query": logql,
                "time": center_ns,
                "limit": lines,
                "direction": "backward",
            },
            headers=_headers(ctx.deps.loki_auth),
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return _failure("get_log_context", str(exc))

    return ToolResult(
        tool_name="get_log_context",
        success=True,
        data={
            "service": service,
            "timestamp": timestamp,
            "logql": logql,
            "entries": _flatten_streams(payload),
        },
    )


def build_logql(service: str, query: str) -> str:
    return f'{{compose_service="{_escape_label(service)}"}} |= "{_escape_query(query)}"'


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(auth: str | None) -> dict[str, str]:
    return {"Authorization": auth} if auth else {}


def _failure(tool_name: str, error: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=False, data={}, error=error)


def _flatten_streams(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for stream in payload.get("data", {}).get("result", []):
        labels = stream.get("stream", {})
        for timestamp, line in stream.get("values", []):
            entries.append({"timestamp": timestamp, "line": line, "labels": labels})
    return entries


def _parse_loki_timestamp(timestamp: str) -> int:
    """Convert a timestamp string to Loki nanoseconds.

    Accepts numeric epochs (sec/ms/µs/ns) and ISO-8601 strings (e.g. '2026-05-28T19:26:40Z').
    Jaeger returns startTime in microseconds. JavaScript Date.now() is milliseconds.
    Thresholds (powers of 10 midpoints):
      < 1e10  → seconds    → multiply by 1e9
      < 1e13  → milliseconds → multiply by 1e6
      < 1e16  → microseconds → multiply by 1e3
      >= 1e16 → already nanoseconds
    """
    from datetime import datetime

    value = timestamp.strip()
    # ISO-8601 with 'T' separator — LLMs sometimes pass this instead of an epoch
    if "T" in value:
        iso = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return int(dt.timestamp() * 1_000_000_000)
    if "." in value:
        return int(float(value) * 1_000_000_000)
    parsed = int(value)
    if parsed < 10_000_000_000:  # seconds
        return parsed * 1_000_000_000
    if parsed < 10_000_000_000_000:  # milliseconds
        return parsed * 1_000_000
    if parsed < 10_000_000_000_000_000:  # microseconds (Jaeger)
        return parsed * 1_000
    return parsed  # already nanoseconds


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
