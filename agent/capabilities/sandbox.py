"""Sandbox reproduction delegator capability."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.usage import UsageLimits

import config
from agent.models import AgentDeps, ErrorType, SandboxResult
from agent.subagents.sandbox import build_sandbox_toolsets, sandbox_subagent


@dataclass
class SandboxCapability(AbstractCapability[AgentDeps]):
    """Delegates sandbox-friendly reproduction attempts to the sandbox sub-agent."""

    e2b_api_key: str | None
    github_token: str | None
    repo: str | None

    def get_toolset(self) -> AgentToolset[AgentDeps] | None:
        if not (self.e2b_api_key and self.github_token and self.repo):
            return None
        return FunctionToolset(tools=[reproduce_in_sandbox])


async def reproduce_in_sandbox(
    ctx: RunContext[AgentDeps],
    hypothesis: str,
    error_type: ErrorType,
    file_path: str,
) -> SandboxResult:
    """Try to reproduce a code/runtime hypothesis in E2B when trigger checks pass."""
    skip_reason = _skip_reason(ctx.deps, error_type, file_path)
    if skip_reason is not None:
        return _skipped(skip_reason)

    prompt = (
        "Reproduce this suspected bug by RUNNING code in an isolated E2B sandbox.\n\n"
        f"Repository: {ctx.deps.repo}\n"
        f"Suspect file (already located — do not search for it): {file_path}\n"
        f"Error type: {error_type}\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Create a sandbox, clone the repository into it, go straight to the known suspect "
        "file, write the smallest repro that exercises it, run it, terminate the sandbox, "
        "and return SandboxResult. Do not grep/find or call GitHub — the path is known."
    )

    # No `usage=ctx.usage` — sub-agent gets its own token counter so the 300k
    # cap applies only to sandbox work, not the parent's accumulated total.
    result = await sandbox_subagent.run(
        prompt,
        deps=ctx.deps,
        toolsets=build_sandbox_toolsets(ctx.deps),
        usage_limits=UsageLimits(request_limit=50, total_tokens_limit=300_000),
    )
    # Record authoritative sandbox outcome to the per-run evidence ledger so
    # the confidence governor can grant HIGH based on a real reproduction.
    ev = ctx.deps.run_evidence
    ev.sandbox_attempted = True
    ev.sandbox_reproduced = result.output.reproduced
    ev.sandbox_confirmed_file = result.output.confirmed_file
    return result.output


def _skip_reason(deps: AgentDeps, error_type: str, file_path: str) -> str | None:
    if error_type not in config.SANDBOX_ERROR_TYPES:
        return f"Error type '{error_type}' is not sandbox-friendly."
    if not deps.e2b_api_key:
        return "E2B API key is not configured."
    if not deps.github_token:
        return "GitHub token is not configured."
    if not deps.repo:
        return "Repository is not configured."
    if not file_path.strip():
        return "A suspect file path is required for sandbox reproduction."
    return None


def _skipped(skip_reason: str) -> SandboxResult:
    return SandboxResult(
        reproduced=False,
        stdout="",
        stderr="",
        exit_code=0,
        repro_script="",
        attempts=0,
        skip_reason=skip_reason,
    )
