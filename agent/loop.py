"""Investigation loop wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
from pydantic_ai import Agent, FunctionToolCallEvent
from pydantic_ai.usage import UsageLimits
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.online import evaluate, run_evaluators

import config
from agent.aimodel import make_model
from agent.capabilities.github import GitHubCapability
from agent.core import agent
from agent.fingerprint import fingerprint
from agent.models import AgentDeps, HealResult, IssueContext, RunEvidence
from agent.prompts import build_user_prompt
from agent.registry import build_capabilities
from agent.verification.ledger import LedgerToolset

# ---------------------------------------------------------------------------
# Online LLM judge — runs in background on every production investigation.
# Uses deepseek-v4-flash (same cheap model as the verifier) so it's fast
# and doesn't add meaningful cost. Results flow to Logfire automatically.
# ---------------------------------------------------------------------------

_JUDGE_RUBRIC = """
You are evaluating the output of an autonomous debugging agent that investigated a production issue.

Judge whether the diagnosis is valid and well-reasoned. Consider:
1. SERVICE — Does the root cause clearly identify a specific service (not vague or "unknown")?
2. EVIDENCE — Is the diagnosis backed by real evidence (traces, logs, source code) as listed in the evidence field?
3. FILE ANCHOR — If a file_path is given, does it plausibly belong to the identified service?
4. CONFIDENCE HONESTY — Is the stated confidence_level (HIGH/MEDIUM/LOW) consistent with the strength of evidence listed?
5. FIX — Is the recommended fix actionable and consistent with the root cause?

Pass if the diagnosis is specific, evidence-backed, and internally consistent.
Fail if the agent: guessed without evidence, pointed at the wrong service's code, claimed HIGH confidence with weak evidence, or gave a vague/generic root cause.
"""

_online_judge = LLMJudge(
    rubric=_JUDGE_RUBRIC,
    model=make_model(config.VERIFIER_MODEL),
    include_input=False,
    include_expected_output=False,
)


async def _run_online_judge_bg(output: HealResult, issue: IssueContext) -> None:
    """Fire-and-forget: run the online judge in the background, log to Logfire."""
    try:
        ctx = EvaluatorContext(
            name=f"investigate:{issue.service_name or 'unknown'}",
            inputs={"description": issue.description},
            metadata=None,
            expected_output=None,
            output=output,
            duration=0.0,
            _span_tree=None,  # type: ignore[arg-type]
            attributes={},
        )
        await run_evaluators([_online_judge], ctx)
    except Exception:
        pass  # online eval is best-effort — never block or crash the main flow


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
        if isinstance(cap, GitHubCapability) and not github_ok:
            continue
        ts = cap.get_toolset()
        if ts is not None:
            toolsets.append(LedgerToolset(wrapped=ts))
    return toolsets


# ---------------------------------------------------------------------------
# investigate — non-streaming path, decorated for online eval
# ---------------------------------------------------------------------------

@evaluate(
    _online_judge,
    target="autoheal_investigate",
    record_return=False,  # HealResult can be large; Logfire span holds the trace
)
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

    output = run.result.output
    # Fire online judge in the background — doesn't block the SSE response
    asyncio.create_task(_run_online_judge_bg(output, issue))

    yield {
        "type": "result",
        "output": output,
        "messages": run.result.all_messages(),
    }
