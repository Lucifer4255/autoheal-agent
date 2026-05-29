"""Receipt ledger — wraps every parent-agent toolset to record what actually ran.

LedgerToolset intercepts every tool call via WrapperToolset.call_tool, extracts key
facts (family, success, service, file_path) from the real tool result, and appends a
ToolCallRecord to ctx.deps.run_evidence.calls.  The model never writes to the ledger.

Tool-family sets are defined here as the single source of truth and re-exported so
confidence.py can import them without duplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from agent.models import AgentDeps, ToolCallRecord, ToolResult

# ---------------------------------------------------------------------------
# Tool-family classification (single source of truth — also used by confidence.py)
# ---------------------------------------------------------------------------

JAEGER_TOOLS: frozenset[str] = frozenset({"query_traces", "get_trace", "extract_error_spans"})
LOKI_TOOLS: frozenset[str] = frozenset({"query_logs", "get_log_context"})
GITHUB_TOOLS: frozenset[str] = frozenset({
    "search_code", "get_file_contents", "list_files",
    "create_or_update_file", "push_files",
})

_COMPOSE_SERVICE_RE = re.compile(r'compose_service="([^"]+)"')
_FILE_HINT_RE = re.compile(r'([^\s:]+\.\w+:\d+)')


def _classify(tool_name: str) -> str:
    if tool_name in JAEGER_TOOLS:
        return "jaeger"
    if tool_name in LOKI_TOOLS:
        return "loki"
    if tool_name in GITHUB_TOOLS:
        return "github"
    return "other"


# ---------------------------------------------------------------------------
# Per-family normalizers — extract facts, no comprehension
# ---------------------------------------------------------------------------

def _normalize_jaeger(tool_name: str, result: Any) -> ToolCallRecord:
    service: str | None = None
    file_path: str | None = None
    success = True

    if isinstance(result, ToolResult):
        success = result.success
        data = result.data or {}
        # extract_error_spans emits file_hint like "src/Foo.java:42"
        for span in data.get("error_spans", []):
            hint = span.get("file_hint", "")
            if hint:
                file_path = hint.split(":")[0]
            svc = span.get("service") or data.get("service")
            if svc:
                service = svc
        if not service:
            service = data.get("service") or data.get("serviceName")
    elif isinstance(result, str) and "failed" in result.lower():
        success = False

    return ToolCallRecord(
        tool=tool_name, family="jaeger",
        success=success, service=service, file_path=file_path,
    )


def _normalize_loki(tool_name: str, result: Any, tool_args: dict[str, Any]) -> ToolCallRecord:
    service: str | None = None
    error_signal: str | None = None
    success = True

    # Service comes from the LogQL label in the args the model passed
    query = tool_args.get("query", "") or tool_args.get("logql", "")
    m = _COMPOSE_SERVICE_RE.search(query)
    if m:
        service = m.group(1)
    if not service:
        service = tool_args.get("service")

    if isinstance(result, ToolResult):
        success = result.success
        data = result.data or {}
        # Presence of log lines with error keywords is a weak error signal
        lines = data.get("lines") or []
        for line in lines:
            text = str(line).lower()
            if any(k in text for k in ("error", "fail", "exception", "unavailable")):
                error_signal = "error_lines_present"
                break
    elif isinstance(result, str) and "failed" in result.lower():
        success = False

    return ToolCallRecord(
        tool=tool_name, family="loki",
        success=success, service=service, error_signal=error_signal,
    )


def _normalize_github(tool_name: str, result: Any, tool_args: dict[str, Any]) -> ToolCallRecord:
    file_path: str | None = None
    success = True

    if isinstance(result, str):
        # MCP returns a string; error strings contain "failed:" or "Tool '...' failed"
        if "failed" in result.lower() or "not found" in result.lower():
            success = False
        # Try to extract a file path from the content (e.g. the "path" key often printed)
        m = re.search(r'"path":\s*"([^"]+)"', result)
        if m:
            file_path = m.group(1)
    elif isinstance(result, dict):
        success = not bool(result.get("error"))
        file_path = result.get("path") or result.get("name")
    elif isinstance(result, ToolResult):
        success = result.success
        file_path = (result.data or {}).get("path")

    # Fall back to what the model requested
    if not file_path:
        file_path = tool_args.get("path") or tool_args.get("file_path")

    return ToolCallRecord(
        tool=tool_name, family="github",
        success=success, file_path=file_path,
    )


# ---------------------------------------------------------------------------
# LedgerToolset
# ---------------------------------------------------------------------------

@dataclass
class LedgerToolset(WrapperToolset[AgentDeps]):
    """Transparent wrapper that stamps a receipt for every real tool execution."""

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDeps],
        tool: ToolsetTool,
    ) -> Any:
        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
        except Exception:
            ctx.deps.run_evidence.calls.append(
                ToolCallRecord(tool=name, family=_classify(name), success=False)
            )
            raise

        family = _classify(name)
        if family == "jaeger":
            record = _normalize_jaeger(name, result)
        elif family == "loki":
            record = _normalize_loki(name, result, tool_args)
        elif family == "github":
            record = _normalize_github(name, result, tool_args)
        else:
            success = not (isinstance(result, ToolResult) and not result.success)
            record = ToolCallRecord(tool=name, family="other", success=success)

        ctx.deps.run_evidence.calls.append(record)
        return result
