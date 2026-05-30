"""Main agent and result validation."""

from __future__ import annotations

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.online_capability import OnlineEvaluation

import config
from agent.aimodel import make_model
from agent.verification.confidence import BAND_RANK, assess_allowed_band, band_of_float, clamp_to_band
from agent.models import AgentDeps, HealResult
from agent.prompts import SYSTEM_PROMPT_BASE, build_dynamic_prompt
from agent.subagents.sandbox import sandbox_subagent as sandbox_subagent
from agent.verification.verifier import run_verifier

# ---------------------------------------------------------------------------
# Online LLM judge — fires in the background after every agent run and streams
# pass/fail + reason to Logfire's Live Evaluations page. Attached as a
# capability so it covers BOTH run paths (agent.run and agent.iter).
# ---------------------------------------------------------------------------

_ONLINE_JUDGE_RUBRIC = """
You are evaluating an autonomous debugging agent's root cause diagnosis (a HealResult).

Pass if the diagnosis is specific and internally consistent:
1. SERVICE — names a specific responsible service (not vague or "unknown").
2. EVIDENCE — backed by real evidence (traces/logs/source) in the evidence field.
3. FILE ANCHOR — if a file_path is given, it plausibly belongs to that service.
4. CONFIDENCE HONESTY — stated confidence_level matches the strength of evidence.
5. FIX — recommended fix is actionable and consistent with the root cause.

Fail if the agent guessed without evidence, anchored to the wrong service's code,
claimed HIGH confidence on weak evidence, or gave a generic/vague root cause.
"""

_online_capabilities = []
if config.ONLINE_EVAL_ENABLED:
    _online_capabilities.append(
        OnlineEvaluation(
            evaluators=[
                LLMJudge(
                    rubric=_ONLINE_JUDGE_RUBRIC,
                    model=make_model(config.ONLINE_EVAL_MODEL),
                    include_input=True,
                )
            ]
        )
    )

# ---------------------------------------------------------------------------
# Main investigation agent
# ---------------------------------------------------------------------------
# Contract for capabilities:
#   - Each capability implements AbstractCapability[AgentDeps]
#   - get_toolset() returns an AgentToolset or None (if not configured)
#   - Tool functions take ctx: RunContext[AgentDeps] as first arg
#   - All tool functions return ToolResult — never raise
#
# Contract for the loop:
#   - Call agent.run(prompt, deps=deps, toolsets=capabilities, usage_limits=...)
#   - Access result.output → HealResult
# ---------------------------------------------------------------------------

agent: Agent[AgentDeps, HealResult] = Agent(
    make_model(),
    deps_type=AgentDeps,
    output_type=HealResult,
    instructions=SYSTEM_PROMPT_BASE,
    retries=config.OUTPUT_RETRIES,
    capabilities=_online_capabilities,
)


@agent.instructions
def dynamic_instructions(ctx: RunContext[AgentDeps]) -> str:
    """Appended each run — tells the agent exactly which tools are available."""
    return build_dynamic_prompt(ctx.deps)


@agent.output_validator
async def normalize_capability_fields(
    ctx: RunContext[AgentDeps], result: HealResult
) -> HealResult:
    """Fix two fields the LLM consistently hallucinates:

    - tools_unavailable: must reflect what the SESSION lacks, not what the LLM didn't use.
    - action_taken: must be 'sandbox_enriched' only if the sandbox tool was actually called.
    """
    result.tools_unavailable = ctx.deps.unavailable_capabilities()

    sandbox_called = any("sandbox" in t.lower() or "reproduce" in t.lower() for t in result.tools_used)
    if result.action_taken == "sandbox_enriched" and not sandbox_called:
        result.action_taken = "explained"

    return result


@agent.output_validator
async def govern_confidence(ctx: RunContext[AgentDeps], result: HealResult) -> HealResult:
    """Evidence-gated confidence governor.

    1. Floor check: if confidence < RETRY_CONFIDENCE, demand more evidence.
    2. Ceiling check: compute the max band the collected evidence supports.
       - If the model overclaimed AND we haven't already retried: one ModelRetry naming
         the missing evidence (the agent can then choose to run the sandbox, read source, etc.)
       - If overclaimed on the second pass (or evidence still insufficient): clamp the float
         and the band, write confidence_note explaining what would raise it.
    3. Always set confidence_level from the final (possibly clamped) float.
    """
    rc = result.root_cause
    ev = ctx.deps.run_evidence

    # 1. Floor
    if rc.confidence < config.RETRY_CONFIDENCE:
        raise ModelRetry(
            f"Root cause confidence {rc.confidence:.2f} is below the minimum "
            f"{config.RETRY_CONFIDENCE}. Collect more evidence before returning a result."
        )

    # 2. Ceiling
    allowed_band, missing_msg = assess_allowed_band(result, ev)
    claimed_band = band_of_float(rc.confidence)

    if BAND_RANK[claimed_band] > BAND_RANK[allowed_band]:
        if not ev.overclaim_retried:
            ev.overclaim_retried = True
            raise ModelRetry(
                f"You claimed {claimed_band.upper()} confidence but the evidence only supports "
                f"{allowed_band.upper()}. {missing_msg} Then finalize."
            )
        # Second pass still overclaiming — clamp silently
        rc.confidence = clamp_to_band(rc.confidence, allowed_band)
        if not result.confidence_note:
            result.confidence_note = missing_msg

    # 3. Always set the band from the final float
    rc.confidence_level = band_of_float(rc.confidence)

    # 4. Optional receipt-reading verifier (downgrade-only, gated by config)
    if config.VERIFIER_ENABLED and BAND_RANK[rc.confidence_level] >= BAND_RANK[config.VERIFIER_MIN_BAND]:
        verdict = await run_verifier(result, ev, rc.confidence_level)
        if verdict.decision == "downgrade" and BAND_RANK[verdict.target_band] < BAND_RANK[rc.confidence_level]:
            rc.confidence = clamp_to_band(rc.confidence, verdict.target_band)
            rc.confidence_level = band_of_float(rc.confidence)
            note = f"Verifier downgraded to {verdict.target_band.upper()}: {verdict.reason}"
            result.confidence_note = note if not result.confidence_note else f"{result.confidence_note} | {note}"

    return result
