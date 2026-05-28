"""Tavily web search capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent.models import AgentDeps, ToolResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass
class WebSearchCapability(AbstractCapability[AgentDeps]):
    """Provides Tavily-backed web search when the session has a Tavily key."""

    enabled: bool

    def get_toolset(self) -> AgentToolset[AgentDeps] | None:
        if not self.enabled:
            return None
        return FunctionToolset(tools=[web_search])


async def web_search(
    ctx: RunContext[AgentDeps],
    query: str,
    max_results: int = 5,
) -> ToolResult:
    """Search the web with Tavily and return normalized result snippets."""
    if not ctx.deps.tavily_key:
        return ToolResult(
            tool_name="web_search",
            success=False,
            data={},
            error="Tavily API key is not configured.",
        )

    try:
        response = await ctx.deps.http_client.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {ctx.deps.tavily_key}"},
            json={
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return ToolResult(
            tool_name="web_search",
            success=False,
            data={},
            error=str(exc),
        )

    results: list[dict[str, Any]] = []
    for item in payload.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )

    return ToolResult(
        tool_name="web_search",
        success=True,
        data={"query": query, "results": results},
    )
