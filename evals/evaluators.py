"""pydantic_evals evaluators for AutoHeal.

Per-case graders used across all three suites.  Calibration metrics (precision@HIGH,
overconfidence_rate, coverage@HIGH) are aggregated from case-level scores in the runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge

import config
from agent.aimodel import make_model
from agent.models import HealResult


# ── helpers ────────────────────────────────────────────────────────────────

def _rc(output: HealResult):
    return output.root_cause


# ── Suite 1 / whole-agent evaluators ──────────────────────────────────────

_DIAGNOSIS_RUBRIC = """
You are grading an automated debugging agent's root cause diagnosis.

The EXPECTED OUTPUT describes what the correct answer looks like:
- 'service': which service is responsible
- 'flag': the feature flag that was enabled (describes the injected failure behavior)
- 'file_substring': if set, a substring that should appear in the diagnosed file path

The ACTUAL OUTPUT is a HealResult from the agent containing:
- root_cause.description: the agent's explanation
- root_cause.error_type: classification of the error
- root_cause.file_path: the file the agent anchored to
- root_cause.evidence: list of evidence collected

Grade PASS if:
1. The agent correctly identified the affected service (matches 'service')
2. The root cause description plausibly explains the failure behavior described by the 'flag'
   (e.g. for 'adServiceFailure' — generates errors 1/10 of the time — any of code_logic,
   runtime_error, config_error are acceptable; the key is that the description explains
   the failure mechanism correctly)
3. The diagnosis is evidence-backed (evidence list is non-empty and relevant)

Grade FAIL if:
- Wrong service identified
- Root cause description is generic/vague with no specific mechanism
- Agent investigated a completely unrelated service's code
- No evidence collected
"""

def make_diagnosis_judge() -> LLMJudge:
    """Build the LLM judge for offline investigation evals.

    Uses the same cheap model as the verifier (deepseek-v4-flash via config)
    so judge calls don't dominate eval cost.
    """
    return LLMJudge(
        rubric=_DIAGNOSIS_RUBRIC,
        model=make_model(config.VERIFIER_MODEL),
        include_input=False,
        include_expected_output=True,  # judge reads expected service + flag behavior
    )


@dataclass
class FileMatch(Evaluator):
    """1.0 if expected file substring appears in root_cause.file_path (case-insensitive)."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        expected = (ctx.expected_output or {}).get("file_substring", "")
        if not expected:
            return 1.0  # no file constraint for this case
        file_path = _rc(ctx.output).file_path or ""
        return 1.0 if expected.lower() in file_path.lower() else 0.0

    def get_default_evaluation_name(self) -> str:
        return "file_match"


@dataclass
class BandFloor(Evaluator):
    """1.0 if confidence_level >= expected_band_floor from case metadata."""

    _RANK = {"low": 0, "medium": 1, "high": 2}

    def evaluate(self, ctx: EvaluatorContext) -> float:
        floor = (ctx.metadata or {}).get("expected_band_floor", "low")
        actual_band = _rc(ctx.output).confidence_level
        return 1.0 if self._RANK[actual_band] >= self._RANK[floor] else 0.0

    def get_default_evaluation_name(self) -> str:
        return "band_floor"


@dataclass
class ConfidenceLevelScore(Evaluator):
    """Records the raw confidence_level as a numeric score for calibration rollup.

    high → 2, medium → 1, low → 0. Not a pass/fail — used by the runner to
    compute precision@HIGH and overconfidence_rate across cases.
    """

    _RANK = {"low": 0, "medium": 1, "high": 2}

    def evaluate(self, ctx: EvaluatorContext) -> float:
        return float(self._RANK[_rc(ctx.output).confidence_level])

    def get_default_evaluation_name(self) -> str:
        return "confidence_level_score"


@dataclass
class EfficiencyEvaluator(Evaluator):
    """Records tool-call count as a score (lower is better — for tracking, not gating)."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        ev = ctx.output.root_cause  # reach deps via attribute stored by task fn
        # The runner stores ledger call count in attributes via set_eval_attribute
        return float(ctx.attributes.get("ledger_call_count", 0))

    def get_default_evaluation_name(self) -> str:
        return "tool_call_count"


# ── Suite 2 / verifier evaluators ─────────────────────────────────────────

@dataclass
class VerifierVerdictMatch(Evaluator):
    """For verifier cases: checks the verdict matches expected decision + band.

    expected_output: {"decision": "keep"|"downgrade", "target_band": "high"|"medium"|"low"}
    output: VerifierVerdict (has .decision and .target_band)
    """

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float]:
        expected = ctx.expected_output or {}
        output = ctx.output
        decision_ok = float(output.decision == expected.get("decision", ""))
        band_ok = float(output.target_band == expected.get("target_band", ""))
        return {"decision_match": decision_ok, "band_match": band_ok}


@dataclass
class FalseKeepRate(Evaluator):
    """1.0 (bad) when the verifier kept but should have downgraded; 0.0 otherwise.

    This is the headline verifier metric — a false keep lets a wrong HIGH through.
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        expected_decision = (ctx.expected_output or {}).get("decision", "")
        actual_decision = ctx.output.decision
        if expected_decision == "downgrade" and actual_decision == "keep":
            return 1.0  # false keep — the dangerous direction
        return 0.0

    def get_default_evaluation_name(self) -> str:
        return "false_keep"


# ── Suite 3 / sandbox evaluators ──────────────────────────────────────────

@dataclass
class SandboxReproduced(Evaluator):
    """1.0 if sandbox reproduced the bug (for fixture-repo tier B cases)."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        return 1.0 if ctx.output.reproduced else 0.0

    def get_default_evaluation_name(self) -> str:
        return "reproduced"


@dataclass
class SandboxHonest(Evaluator):
    """For OTel flag cases (Tier C): 1.0 if the sandbox did NOT falsely claim reproduction.

    Reproduction is expected to fail (infra not available in sandbox). A false
    reproduced=True is the critical failure — it would wrongly unlock HIGH.
    """

    def evaluate(self, ctx: EvaluatorContext) -> float:
        # If the case metadata says reproduction is NOT expected, penalise false positives
        expected_reproduced = (ctx.metadata or {}).get("expected_reproduced", None)
        if expected_reproduced is False and ctx.output.reproduced is True:
            return 0.0  # false reproduction claim
        return 1.0

    def get_default_evaluation_name(self) -> str:
        return "honest_reproduction"


@dataclass
class SandboxFollowsWorkflow(Evaluator):
    """1.0 if the sandbox result indicates the full workflow ran (for Tier A mock check)."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        result = ctx.output
        # repro_script non-empty means the agent wrote one; attempts > 0 means it ran
        script_written = bool(result.repro_script and result.repro_script.strip())
        attempted = result.attempts > 0 or result.skip_reason is not None
        return 1.0 if (script_written or attempted) else 0.0

    def get_default_evaluation_name(self) -> str:
        return "workflow_followed"
