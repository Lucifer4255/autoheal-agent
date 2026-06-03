"""Targeted source-read capability — fetch just the lines around a known location.

The GitHub MCP `get_file_contents` tool returns whole files (no line range), which is
expensive once a trace has already anchored you to a specific line. `get_file_slice` reads
only the window around that line via the GitHub Contents API.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent.models import AgentDeps, ToolResult


@dataclass
class SourceCapability(AbstractCapability[AgentDeps]):
    """Provides get_file_slice when a GitHub token and repository are configured."""

    github_token: str | None
    repo: str | None

    def get_toolset(self) -> AgentToolset[AgentDeps] | None:
        if not (self.github_token and self.repo):
            return None
        return FunctionToolset(tools=[get_file_slice])


async def get_file_slice(
    ctx: RunContext[AgentDeps],
    path: str,
    around_line: int,
    context: int = 40,
    ref: str | None = None,
) -> ToolResult:
    """Read only the lines around a known location — cheaper than get_file_contents.

    Pass the `path` and line from a trace's file_hint (e.g. path="src/.../CartService.cs",
    around_line=83). Returns the window [around_line-context, around_line+context] with line
    numbers, instead of the whole file. Prefer this once a trace or search has anchored you to
    a specific line; only fall back to get_file_contents when you genuinely need the full file.
    """
    deps = ctx.deps
    if not (deps.github_token and deps.repo):
        return _failure("get_file_slice", "GitHub token/repo not configured.")

    params = {"ref": ref} if ref else {}
    try:
        resp = await deps.http_client.get(
            f"https://api.github.com/repos/{deps.repo}/contents/{path.lstrip('/')}",
            headers={
                "Authorization": f"Bearer {deps.github_token}",
                "Accept": "application/vnd.github.raw+json",
            },
            params=params,
            timeout=20.0,
        )
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:
        return _failure("get_file_slice", str(exc))

    lines = text.splitlines()
    total = len(lines)
    lo = max(1, around_line - context)
    hi = min(total, around_line + context)
    window = "\n".join(f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1)) if total else ""

    return ToolResult(
        tool_name="get_file_slice",
        success=True,
        data={
            "path": path,
            "around_line": around_line,
            "start_line": lo,
            "end_line": hi,
            "total_lines": total,
            "slice": window,
        },
    )


def _failure(tool_name: str, error: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=False, data={}, error=error)
