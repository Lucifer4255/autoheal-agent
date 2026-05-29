"""Receipt-reading, downgrade-only confidence verifier.

A small fast sub-agent (deepseek-v4-flash by default) that reads the RECEIPTS from the
real tool executions alongside the claimed root cause, and judges whether the evidence
genuinely supports the claimed conclusion.

Critically:
- It reads receipts (code-stamped facts), NOT the investigator's "trust me" prose.
- It can only LOWER confidence (enforced by min() in the governor, not by trust).
- Even a hallucinating verifier only over-cautions → LOW; it cannot produce a false HIGH.
- It is only invoked when the deterministic ceiling is MEDIUM or HIGH (skip LOW — no point).
"""

from __future__ import annotations

import json

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

import config
from agent.aimodel import make_model
from agent.models import AgentDeps, ConfidenceLevel, HealResult, RunEvidence

VERIFIER_PROMPT = """You are a strict evidence verifier for a code debugging agent.

You will be given:
1. A claimed root cause (what the investigator concluded)
2. The actual tool receipts (what tools REALLY ran and what they returned)

Your job is to check whether the receipts GENUINELY support the claim.
You are NOT re-investigating. You are checking internal consistency.

Ask yourself:
- Do the trace error and log lines describe the SAME failure, or just share a service name?
- Is the cited file/line actually present in the receipt for the file that was read?
- Does the sandbox stdout/stderr actually demonstrate the hypothesized bug?
- Is the service in the traces the same as the service in the logs?

Be strict. If the evidence is weak, ambiguous, or doesn't clearly support the claim,
downgrade. Only "keep" if the receipts clearly and unambiguously support the conclusion.

Return a VerifierVerdict with:
- decision: "keep" or "downgrade"
- target_band: the band you assign ("high", "medium", or "low")
- reason: one concise sentence explaining your judgment
"""


class VerifierVerdict(BaseModel):
    decision: str           # "keep" | "downgrade"
    target_band: ConfidenceLevel
    reason: str


_verifier_agent: Agent[None, VerifierVerdict] = Agent(
    make_model(config.VERIFIER_MODEL),
    output_type=VerifierVerdict,
    instructions=VERIFIER_PROMPT,
)


def _serialize_receipts(run_evidence: RunEvidence) -> str:
    """Serialize receipts into a compact, readable form for the verifier prompt."""
    records = []
    for c in run_evidence.calls:
        rec: dict = {"tool": c.tool, "family": c.family, "success": c.success}
        if c.service:
            rec["service"] = c.service
        if c.file_path:
            rec["file_path"] = c.file_path
        if c.error_signal:
            rec["error_signal"] = c.error_signal
        records.append(rec)

    sandbox_info = {
        "attempted": run_evidence.sandbox_attempted,
        "reproduced": run_evidence.sandbox_reproduced,
        "confirmed_file": run_evidence.sandbox_confirmed_file,
    }
    return json.dumps({"tool_receipts": records, "sandbox": sandbox_info}, indent=2)


async def run_verifier(
    result: HealResult,
    run_evidence: RunEvidence,
    current_band: ConfidenceLevel,
) -> VerifierVerdict:
    """Run the verifier sub-agent. Returns a verdict (decision, target_band, reason).

    The verifier is given the claimed root cause + serialized receipts.
    It never sees the investigator's prose reasoning.
    """
    rc = result.root_cause
    receipts_text = _serialize_receipts(run_evidence)

    prompt = (
        f"CLAIMED ROOT CAUSE:\n"
        f"  description: {rc.description}\n"
        f"  error_type: {rc.error_type}\n"
        f"  file_path: {rc.file_path or 'not specified'}\n"
        f"  line_number: {rc.line_number or 'not specified'}\n"
        f"  current_band: {current_band.upper()}\n\n"
        f"TOOL RECEIPTS (what actually ran):\n{receipts_text}\n\n"
        f"Verify whether the receipts support this conclusion. "
        f"Your target_band must be ≤ {current_band} (you may only downgrade, not upgrade)."
    )

    verifier_result = await _verifier_agent.run(prompt)
    verdict = verifier_result.output

    # Hard-enforce downgrade-only — the verifier cannot raise the band
    if _band_rank(verdict.target_band) > _band_rank(current_band):
        verdict.target_band = current_band
        verdict.decision = "keep"

    return verdict


def _band_rank(band: ConfidenceLevel) -> int:
    return {"low": 0, "medium": 1, "high": 2}[band]
