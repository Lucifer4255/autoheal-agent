"""Sandbox sub-agent for isolated bug reproduction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from e2b import Sandbox
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

from agent.aimodel import make_model
from agent.models import AgentDeps, SandboxResult, ToolResult

SANDBOX_SUBAGENT_PROMPT = """You are AutoHeal's isolated reproduction sub-agent.

Goal: actually RUN code in an E2B sandbox to confirm or refute a specific hypothesis.
You are a code runner, not an investigator. The parent agent has ALREADY located the
suspect file — its path is given to you. Do NOT search for it (no grep, no find) and do
NOT call GitHub. Go straight to the known file and execute the code.

Workflow:
1. Call `create_sandbox` first. Pass the returned sandbox_id to every later tool call.
2. Call `clone_repo` to clone the target repository into the sandbox. It returns the
   local checkout path; the suspect file lives at <checkout_path>/<suspect_file_path>.
3. `cat` only the suspect file (and any direct dependency the repro genuinely needs) to
   read the code you are about to exercise. Do not explore the rest of the tree.
4. Write the SMALLEST possible repro that exercises the suspect logic — import or call the
   real code from the clone when practical — then run it with `run_command`. Install any
   needed toolchain in the sandbox first if the language requires it.
5. Capture stdout/stderr/exit code. Decide whether the hypothesis reproduced.
6. Call `terminate` to kill the sandbox.
7. Return a SandboxResult. If you cannot build a repro, explain why in skip_reason.

Rules:
- Stay read-only with respect to GitHub and production systems — only mutate the sandbox.
- The file path is already known. Never search for it; go directly to it.
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


def build_sandbox_toolsets(deps: AgentDeps) -> list[FunctionToolset]:
    """Return run-scoped toolsets for the sandbox sub-agent.

    Only the E2B toolset is delivered — the sandbox clones the repo and inspects it
    with shell tools rather than calling GitHub. e2b_toolset is intentionally NOT
    registered on the Agent at definition time to avoid double-registration.
    """
    del deps
    return [e2b_toolset]


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


async def clone_repo(
    ctx: RunContext[AgentDeps],
    sandbox_id: str,
    dest: str = "/home/user/repo",
) -> ToolResult:
    """Clone the target repository into the sandbox. Returns the local checkout path.

    The auth token is injected here and never appears in the model's context.
    """
    session = _ACTIVE_SANDBOXES.get(sandbox_id)
    if session is None:
        return _failure("clone_repo", f"No active sandbox with id '{sandbox_id}'.")
    if not ctx.deps.repo:
        return _failure("clone_repo", "No target repository is configured.")

    token = ctx.deps.github_token or ""
    auth = f"x-access-token:{token}@" if token else ""
    clone_url = f"https://{auth}github.com/{ctx.deps.repo}.git"
    command = f"git clone --depth 1 --quiet {clone_url} {dest}"

    try:
        result = await asyncio.to_thread(
            session.sandbox.commands.run, command, timeout=180
        )
    except Exception as exc:
        return _failure("clone_repo", _redact(str(exc), token))

    exit_code = int(getattr(result, "exit_code", 0) or 0)
    stderr = _redact(getattr(result, "stderr", "") or "", token)
    if exit_code != 0:
        return _failure("clone_repo", f"git clone failed (exit {exit_code}): {stderr}")

    # Record a sanitized entry so read_output never exposes the token.
    session.history.append(
        _CommandResult(
            command=f"git clone --depth 1 <repo> {dest}",
            stdout="",
            stderr=stderr,
            exit_code=exit_code,
        )
    )
    return ToolResult(
        tool_name="clone_repo",
        success=True,
        data={"sandbox_id": sandbox_id, "path": dest, "repo": ctx.deps.repo},
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


def _redact(text: str, token: str) -> str:
    """Strip the auth token from any output before it reaches the model."""
    if token and token in text:
        text = text.replace(token, "***")
    return text


e2b_toolset: FunctionToolset = FunctionToolset(
    tools=[create_sandbox, clone_repo, run_command, read_output, terminate]
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
