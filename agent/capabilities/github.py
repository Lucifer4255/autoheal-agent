"""GitHub MCP read-only code access capability."""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp.client.transports import StdioTransport
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AgentToolset

from agent.models import AgentDeps


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
            )
        )
