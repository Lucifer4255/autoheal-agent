"""Sandbox sub-agent for isolated bug reproduction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from e2b import Sandbox
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import FunctionToolset
from uuid import uuid4

from agent.aimodel import make_model
from agent.models import AgentDeps, SandboxResult, ToolResult

SANDBOX_SUBAGENT_PROMPT = """You are AutoHeal's isolated reproduction sub-agent.

Goal: validate a specific code/runtime hypothesis in an E2B sandbox.

Rules:
- Stay read-only with respect to GitHub and production systems.
- Pull or inspect only the suspect file and minimal dependencies needed for a repro.
- Write the smallest possible repro script.
- Call create_sandbox first. Pass the returned sandbox_id to every subsequent tool call.
- Run the script in E2B, capture stdout/stderr/exit code, then terminate the sandbox.
- Return SandboxResult. If reproduction is not possible, explain why in skip_reason.
"""


@dataclass
class _CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class _SandboxSession:
    sandbox: Sandbox
    history: list[_CommandResult] = field(default_factory=list)


# Keyed by the local sandbox_id UUID returned to the sub-agent
_ACTIVE_SANDBOXES: dict[str, _SandboxSession] = {}


def build_github_mcp_toolset(deps: AgentDeps) -> MCPToolset | None:
    """Build the read-only GitHub MCP toolset for a sandbox run."""
    if not (deps.github_token and deps.repo):
        return None
    return MCPToolset(
        StdioTransport(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": deps.github_token},
        )
    )


def build_sandbox_toolsets(deps: AgentDeps) -> list[FunctionToolset | MCPToolset]:
    """Return run-scoped toolsets for the sandbox sub-agent.

    e2b_toolset is intentionally NOT registered on the Agent at definition time —
    it is delivered here at run time to avoid double-registration.
    """
    toolsets: list[FunctionToolset | MCPToolset] = [e2b_toolset]
    github_toolset = build_github_mcp_toolset(deps)
    if github_toolset is not None:
        toolsets.append(github_toolset)
    return toolsets


async def create_sandbox(ctx: RunContext[AgentDeps]) -> ToolResult:
    """Create an E2B sandbox. Returns sandbox_id — pass it to every subsequent tool call."""
    if not ctx.deps.e2b_api_key:
        return _failure("create_sandbox", "E2B API key is not configured.")

    try:
        sandbox = await asyncio.to_thread(
            Sandbox.create,
            timeout=300,
            api_key=ctx.deps.e2b_api_key,
        )
    except Exception as exc:
        return _failure("create_sandbox", str(exc))

    sandbox_id = str(uuid4())
    _ACTIVE_SANDBOXES[sandbox_id] = _SandboxSession(sandbox=sandbox)
    return ToolResult(
        tool_name="create_sandbox",
        success=True,
        data={"sandbox_id": sandbox_id},
    )


async def run_command(
    ctx: RunContext[AgentDeps],
    sandbox_id: str,
    command: str,
    timeout_seconds: int = 60,
) -> ToolResult:
    """Run a shell command in the sandbox. Pass the sandbox_id from create_sandbox."""
    del ctx
    session = _ACTIVE_SANDBOXES.get(sandbox_id)
    if session is None:
        return _failure("run_command", f"No active sandbox with id '{sandbox_id}'.")

    try:
        result = await asyncio.to_thread(
            session.sandbox.commands.run, command, timeout=timeout_seconds
        )
    except Exception as exc:
        return _failure("run_command", str(exc))

    cmd_result = _CommandResult(
        command=command,
        stdout=getattr(result, "stdout", "") or "",
        stderr=getattr(result, "stderr", "") or "",
        exit_code=int(getattr(result, "exit_code", 0) or 0),
    )
    session.history.append(cmd_result)

    return ToolResult(
        tool_name="run_command",
        success=True,
        data={
            "sandbox_id": sandbox_id,
            "command": command,
            "stdout": cmd_result.stdout,
            "stderr": cmd_result.stderr,
            "exit_code": cmd_result.exit_code,
        },
    )


async def read_output(
    ctx: RunContext[AgentDeps],
    sandbox_id: str,
) -> ToolResult:
    """Read the full command history for this sandbox session."""
    del ctx
    session = _ACTIVE_SANDBOXES.get(sandbox_id)
    if session is None:
        return _failure("read_output", f"No active sandbox with id '{sandbox_id}'.")

    return ToolResult(
        tool_name="read_output",
        success=True,
        data={
            "sandbox_id": sandbox_id,
            "command_count": len(session.history),
            "history": [
                {
                    "command": r.command,
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "exit_code": r.exit_code,
                }
                for r in session.history
            ],
        },
    )


async def terminate(
    ctx: RunContext[AgentDeps],
    sandbox_id: str,
) -> ToolResult:
    """Terminate the E2B sandbox. Retryable — session stays registered until kill succeeds."""
    del ctx
    session = _ACTIVE_SANDBOXES.get(sandbox_id)
    if session is None:
        return ToolResult(
            tool_name="terminate",
            success=True,
            data={"sandbox_id": sandbox_id, "terminated": False, "reason": "No active sandbox."},
        )

    try:
        await asyncio.to_thread(session.sandbox.kill)
    except Exception as exc:
        # Leave session in _ACTIVE_SANDBOXES so the caller can retry terminate
        return _failure("terminate", str(exc))

    # Only remove after a confirmed successful kill
    _ACTIVE_SANDBOXES.pop(sandbox_id, None)
    return ToolResult(
        tool_name="terminate",
        success=True,
        data={"sandbox_id": sandbox_id, "terminated": True, "commands_run": len(session.history)},
    )


def _failure(tool_name: str, error: str) -> ToolResult:
    return ToolResult(tool_name=tool_name, success=False, data={}, error=error)


e2b_toolset: FunctionToolset = FunctionToolset(
    tools=[create_sandbox, run_command, read_output, terminate]
)

# e2b_toolset is NOT passed here — it is delivered at run time via build_sandbox_toolsets
# to avoid double-registration (pydantic-ai appends run-time toolsets on top of
# definition-time ones, which would register all four tools twice).
sandbox_subagent: Agent[AgentDeps, SandboxResult] = Agent(
    make_model(),
    deps_type=AgentDeps,
    output_type=SandboxResult,
    instructions=SANDBOX_SUBAGENT_PROMPT,
)
