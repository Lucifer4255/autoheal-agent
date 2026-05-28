"""Investigation loop wrapper."""

from __future__ import annotations

from pydantic_ai.usage import UsageLimits

import config
from agent.core import agent
from agent.fingerprint import fingerprint
from agent.models import AgentDeps, HealResult, IssueContext
from agent.prompts import build_user_prompt
from agent.registry import build_capabilities


async def investigate(issue: IssueContext, deps: AgentDeps) -> HealResult:
    """Run a full investigation and return a structured HealResult.

    The fingerprint pre-check narrows the agent's first evidence pass when a
    high-confidence pattern is found, but never skips tool investigation entirely.
    """
    fp = fingerprint(issue.description)

    capabilities = build_capabilities(deps)
    toolsets = [ts for cap in capabilities if (ts := cap.get_toolset()) is not None]

    result = await agent.run(
        build_user_prompt(issue, fp),
        deps=deps,
        toolsets=toolsets,
        usage_limits=UsageLimits(request_limit=config.MAX_TOOL_CALLS),
    )
    return result.output
