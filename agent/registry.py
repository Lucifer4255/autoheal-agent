"""Capability toolset builder."""

from __future__ import annotations

from pydantic_ai.capabilities import AbstractCapability

from agent.capabilities.github import GitHubCapability
from agent.capabilities.jaeger import JaegerCapability
from agent.capabilities.loki import LokiCapability
from agent.capabilities.sandbox import SandboxCapability
from agent.capabilities.source import SourceCapability
from agent.models import AgentDeps


def build_capabilities(deps: AgentDeps) -> list[AbstractCapability[AgentDeps]]:
    """Instantiate all capabilities gated by what the session has configured.

    Each capability's get_toolset() returns None if its required fields are
    absent — the agent.run() call only receives the non-None toolsets.
    """
    return [
        JaegerCapability(enabled=bool(deps.jaeger_url)),
        LokiCapability(enabled=bool(deps.loki_url)),
        GitHubCapability(github_token=deps.github_token, repo=deps.repo),
        SourceCapability(github_token=deps.github_token, repo=deps.repo),
        SandboxCapability(
            e2b_api_key=deps.e2b_api_key,
            github_token=deps.github_token,
            repo=deps.repo,
        ),
    ]
