"""GitHub MCP read-only code access capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastmcp.client.transports import StdioTransport
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AgentToolset

from agent.models import AgentDeps


async def _surface_mcp_errors(
    ctx,
    call_tool,
    tool_name: str,
    tool_args: dict[str, Any],
) -> Any:
    """Catch MCP protocol exceptions and return them to the model as tool output.

    Default MCPToolset behaviour: protocol-level errors (e.g. 'Resource not found'
    from a missing file path) propagate out of agent.iter() and kill the run.
    Here we convert them into a string result the LLM sees, so it can retry the
    call with different arguments (a corrected file path, branch, etc.).
    """
    try:
        return await call_tool(tool_name, tool_args)
    except Exception as exc:
        return f"Tool '{tool_name}' failed: {type(exc).__name__}: {exc}"


@dataclass
class GitHubCapability(AbstractCapability[AgentDeps]):
    """Provides GitHub MCP tools when token and repository are configured."""

    github_token: str | None
    repo: str | None

    def get_toolset(self) -> AgentToolset[AgentDeps] | None:
        if not (self.github_token and self.repo):
            return None
        return MCPToolset(
            StdioTransport(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.github_token},
            ),
            process_tool_call=_surface_mcp_errors,
        )
