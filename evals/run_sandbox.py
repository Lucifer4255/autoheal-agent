"""Suite 3: sandbox sub-agent eval runner.

Three tiers, invoked by flag:
  --mock      Tier A: mock E2B, check orchestration (workflow followed, well-formed output)
  --fixture   Tier B: real E2B + fixture buggy_repo, check reproduced=True
  --otel      Tier C: real E2B + OTel demo, check honest judgment (zero false reproductions)

Usage:
    uv run python -m evals.run_sandbox --mock
    uv run python -m evals.run_sandbox --fixture
    uv run python -m evals.run_sandbox --otel
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
from pydantic_evals import Case, Dataset

from agent.models import AgentDeps, RunEvidence, SandboxResult, ToolResult
from agent.subagents.sandbox import build_sandbox_toolsets
from evals.evaluators import SandboxFollowsWorkflow, SandboxHonest, SandboxReproduced

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "buggy_repo"

# ── helpers ────────────────────────────────────────────────────────────────

def _deps(e2b_key: str | None = None) -> AgentDeps:
    import os
    return AgentDeps(
        jaeger_url=None, jaeger_auth=None,
        loki_url=None, loki_auth=None,
        github_token=os.getenv("GITHUB_TOKEN"),
        repo=os.getenv("SANDBOX_EVAL_REPO", "Lucifer4255/opentelemetry-demo"),
        e2b_api_key=e2b_key or os.getenv("E2B_API_KEY"),
        service_name=None,
        http_client=httpx.AsyncClient(),
        run_evidence=RunEvidence(),
    )


# ── Tier A — mock E2B, orchestration check ────────────────────────────────

def _mock_sandbox_toolsets():
    """Return a fake toolset list that records calls and returns canned results."""
    from pydantic_ai.toolsets import FunctionToolset
    from agent.subagents import sandbox as sb

    call_log: list[str] = []

    async def fake_create(ctx):
        call_log.append("create_sandbox")
        return ToolResult(tool_name="create_sandbox", success=True, data={"sandbox_id": "mock-id"})

    async def fake_clone(ctx, sandbox_id: str, dest: str = "/home/user/repo"):
        call_log.append("clone_repo")
        return ToolResult(tool_name="clone_repo", success=True, data={"path": dest, "sandbox_id": sandbox_id, "repo": "mock"})

    async def fake_run(ctx, sandbox_id: str, command: str, timeout_seconds: int = 60):
        call_log.append("run_command")
        # Simulate a crash for the null_deref fixture
        if "null_deref" in command or "python" in command:
            return ToolResult(tool_name="run_command", success=True,
                              data={"sandbox_id": sandbox_id, "command": command,
                                    "stdout": "", "stderr": "TypeError: 'NoneType' object is not subscriptable",
                                    "exit_code": 1})
        return ToolResult(tool_name="run_command", success=True,
                          data={"sandbox_id": sandbox_id, "command": command,
                                "stdout": "ok", "stderr": "", "exit_code": 0})

    async def fake_read(ctx, sandbox_id: str):
        call_log.append("read_output")
        return ToolResult(tool_name="read_output", success=True,
                          data={"sandbox_id": sandbox_id, "command_count": 1, "history": []})

    async def fake_terminate(ctx, sandbox_id: str):
        call_log.append("terminate")
        return ToolResult(tool_name="terminate", success=True,
                          data={"sandbox_id": sandbox_id, "terminated": True, "commands_run": 1})

    from pydantic_ai.toolsets import FunctionToolset
    toolset = FunctionToolset(tools=[fake_create, fake_clone, fake_run, fake_read, fake_terminate])
    return [toolset], call_log


async def _run_sandbox_subagent(deps: AgentDeps, hypothesis: str, file_path: str,
                                 toolsets=None) -> SandboxResult:
    from agent.subagents.sandbox import sandbox_subagent
    from pydantic_ai.usage import UsageLimits

    prompt = (
        f"Reproduce this suspected bug by RUNNING code in an isolated E2B sandbox.\n\n"
        f"Repository: {deps.repo}\n"
        f"Suspect file (already located — do not search for it): {file_path}\n"
        f"Error type: runtime_error\n"
        f"Hypothesis: {hypothesis}\n\n"
        "Create a sandbox, clone the repository into it, go straight to the known suspect "
        "file, write the smallest repro that exercises it, run it, terminate the sandbox, "
        "and return SandboxResult."
    )
    result = await sandbox_subagent.run(
        prompt,
        deps=deps,
        toolsets=toolsets or build_sandbox_toolsets(deps),
        usage_limits=UsageLimits(request_limit=20, total_tokens_limit=50_000),
    )
    return result.output


# ── Tier A dataset ─────────────────────────────────────────────────────────

def _mock_dataset() -> Dataset:
    return Dataset(
        name="sandbox_mock",
        cases=[
            Case(
                name="null_deref_orchestration",
                inputs={
                    "hypothesis": "Function crashes when passed None — dereferences None dict key",
                    "file_path": "evals/fixtures/buggy_repo/null_deref.py",
                },
                evaluators=(SandboxFollowsWorkflow(),),
            ),
            Case(
                name="parse_error_orchestration",
                inputs={
                    "hypothesis": "int() called on non-numeric string raises ValueError",
                    "file_path": "evals/fixtures/buggy_repo/parse_error.py",
                },
                evaluators=(SandboxFollowsWorkflow(),),
            ),
        ],
    )


# ── Tier B dataset ─────────────────────────────────────────────────────────

def _fixture_dataset() -> Dataset:
    return Dataset(
        name="sandbox_fixture",
        cases=[
            Case(
                name="null_deref_live",
                inputs={
                    "hypothesis": "Function crashes when passed None — dereferences None dict key",
                    "file_path": "evals/fixtures/buggy_repo/null_deref.py",
                },
                evaluators=(SandboxReproduced(),),
            ),
            Case(
                name="parse_error_live",
                inputs={
                    "hypothesis": "int() called on non-numeric string raises ValueError",
                    "file_path": "evals/fixtures/buggy_repo/parse_error.py",
                },
                evaluators=(SandboxReproduced(),),
            ),
            Case(
                name="off_by_one_live",
                inputs={
                    "hypothesis": "Index exceeds list length — off-by-one in last_item()",
                    "file_path": "evals/fixtures/buggy_repo/off_by_one.py",
                },
                evaluators=(SandboxReproduced(),),
            ),
        ],
    )


# ── Tier C dataset ─────────────────────────────────────────────────────────

def _otel_dataset() -> Dataset:
    return Dataset(
        name="sandbox_otel_judgment",
        cases=[
            Case(
                name="adFailure_judgment",
                inputs={
                    "hypothesis": "adFailure flag causes StatusRuntimeException(UNAVAILABLE) when random.nextInt(10)==0",
                    "file_path": "src/adservice/src/main/java/oteldemo/AdService.java",
                },
                metadata={"expected_reproduced": False},  # infra-gated — sandbox can't reproduce
                evaluators=(SandboxHonest(),),
            ),
            Case(
                name="cartFailure_judgment",
                inputs={
                    "hypothesis": "cartFailure flag throws ApplicationException simulating Redis unavailability",
                    "file_path": "src/cartservice/src/cartstore/ValkeyCartStore.cs",
                },
                metadata={"expected_reproduced": False},
                evaluators=(SandboxHonest(),),
            ),
        ],
    )


# ── runners ────────────────────────────────────────────────────────────────

async def run_mock():
    deps = _deps(e2b_key="mock-key")
    mock_toolsets, call_log = _mock_sandbox_toolsets()

    async def task(inputs: dict) -> SandboxResult:
        return await _run_sandbox_subagent(
            deps, inputs["hypothesis"], inputs["file_path"],
            toolsets=mock_toolsets,
        )

    report = await _mock_dataset().evaluate(task)
    report.print()


async def run_fixture():
    deps = _deps()
    if not deps.e2b_api_key:
        print("E2B_API_KEY not set — skipping fixture live run.")
        return

    async def task(inputs: dict) -> SandboxResult:
        return await _run_sandbox_subagent(deps, inputs["hypothesis"], inputs["file_path"])

    report = await _fixture_dataset().evaluate(task)
    report.print()

    reproduced = sum(
        1 for c in report.cases
        if c.scores.get("reproduced") and c.scores["reproduced"].value == 1.0
    )
    total = len(report.cases)
    print(f"\n── Fixture Reproduction: {reproduced}/{total} bugs confirmed ──")


async def run_otel():
    deps = _deps()
    if not deps.e2b_api_key:
        print("E2B_API_KEY not set — skipping OTel judgment run.")
        return

    async def task(inputs: dict) -> SandboxResult:
        return await _run_sandbox_subagent(deps, inputs["hypothesis"], inputs["file_path"])

    report = await _otel_dataset().evaluate(task)
    report.print()

    false_repros = sum(
        1 for c in report.cases
        if c.scores.get("honest_reproduction") and c.scores["honest_reproduction"].value == 0.0
    )
    print(f"\n── OTel Judgment: {false_repros} false reproduction(s) out of {len(report.cases)} ──")
    if false_repros == 0:
        print("✓ Sandbox judgment is honest — no false reproduction claims.")
    else:
        print("✗ Sandbox falsely claimed reproduction — overconfidence risk.")


async def main(tier: str):
    if tier == "mock":
        await run_mock()
    elif tier == "fixture":
        await run_fixture()
    elif tier == "otel":
        await run_otel()
    else:
        print(f"Unknown tier: {tier}. Use --mock, --fixture, or --otel.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true")
    group.add_argument("--fixture", action="store_true")
    group.add_argument("--otel", action="store_true")
    args = parser.parse_args()

    tier = "mock" if args.mock else "fixture" if args.fixture else "otel"
    asyncio.run(main(tier))
