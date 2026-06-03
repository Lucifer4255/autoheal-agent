"""Investigation loop wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from pydantic_ai import Agent, FunctionToolCallEvent
from pydantic_ai.usage import UsageLimits

from agent.capabilities.github import GitHubCapability
from agent.capabilities.source import SourceCapability
from agent.core import agent
from agent.fingerprint import fingerprint
from agent.models import AgentDeps, HealResult, IssueContext, RunEvidence
from agent.prompts import build_user_prompt
from agent.registry import build_capabilities
from agent.verification.ledger import LedgerToolset

# ---------------------------------------------------------------------------
# Toolset builder (shared by both run paths)
# ---------------------------------------------------------------------------

async def _github_repo_reachable(deps: AgentDeps) -> bool:
    """Quick HEAD check so we don't crash the run on a bad repo or token."""
    if not (deps.github_token and deps.repo):
        return False
    try:
        resp = await deps.http_client.get(
            f"https://api.github.com/repos/{deps.repo}",
            headers={"Authorization": f"Bearer {deps.github_token}"},
            timeout=5.0,
        )
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _build_toolsets(deps: AgentDeps) -> list:
    capabilities = build_capabilities(deps)
    github_ok = await _github_repo_reachable(deps)
    toolsets = []
    for cap in capabilities:
        # GitHub MCP and the get_file_slice source reader both need a reachable repo.
        if isinstance(cap, (GitHubCapability, SourceCapability)) and not github_ok:
            continue
        ts = cap.get_toolset()
        if ts is not None:
            toolsets.append(LedgerToolset(wrapped=ts))
    return toolsets


# ---------------------------------------------------------------------------
# investigate — non-streaming path
# Online LLM judge fires automatically via the OnlineEvaluation capability on
# the agent (see core.py), covering this path and stream_investigate alike.
# ---------------------------------------------------------------------------

async def investigate(issue: IssueContext, deps: AgentDeps) -> HealResult:
    """Run a full investigation and return a structured HealResult.

    The fingerprint pre-check narrows the agent's first evidence pass when a
    high-confidence pattern is found, but never skips tool investigation entirely.
    """
    fp = fingerprint(issue.description)
    deps.run_evidence = RunEvidence()

    toolsets = await _build_toolsets(deps)

    result = await agent.run(
        build_user_prompt(issue, fp),
        deps=deps,
        toolsets=toolsets,
        usage_limits=UsageLimits(),
    )
    return result.output


# ---------------------------------------------------------------------------
# stream_investigate — streaming path, judge fires as background task
# ---------------------------------------------------------------------------

async def stream_investigate(
    issue: IssueContext,
    deps: AgentDeps,
    message_history: list | None = None,
) -> AsyncIterator[dict]:
    """Stream investigation progress as dicts.

    Yields {"type": "step", ...} for each tool call as it happens,
    then {"type": "result", "output": HealResult, "messages": [...]} when done.
    The `messages` list is the full conversation history to pass back as
    `message_history=` on the next call to maintain context across turns.
    """
    fp = fingerprint(issue.description)
    deps.run_evidence = RunEvidence()
    toolsets = await _build_toolsets(deps)
    prompt = build_user_prompt(issue, fp)

    round_num = 0
    async with agent.iter(
        prompt,
        deps=deps,
        toolsets=toolsets,
        usage_limits=UsageLimits(),
        message_history=message_history,
    ) as run:
        async for node in run:
            if Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            round_num += 1
                            yield {
                                "type": "step",
                                "round": round_num,
                                "tool": event.part.tool_name,
                                "result_summary": f"Calling {event.part.tool_name}",
                                "confidence_after": 0.0,
                            }

    yield {
        "type": "result",
        "output": run.result.output,
        "messages": run.result.all_messages(),
    }
