"""Suite 2: verifier eval runner.

Fully deterministic — no live tools, no E2B.
Measures false-keep rate and downgrade recall across synthetic receipt + claim cases.

Usage:
    uv run python -m evals.run_verifier
"""

from __future__ import annotations

import asyncio

from pydantic_evals.evaluators import EvaluatorContext

from agent.verification.verifier import run_verifier
from evals.datasets.verifier_cases import build_dataset


def _wrap_rc(rc):
    """Wrap a bare RootCause into a minimal HealResult for run_verifier."""
    from agent.models import HealResult
    return HealResult(
        issue_summary="eval case",
        investigation_steps=[],
        root_cause=rc,
        recommended_fix="n/a",
        action_taken="explained",
        tools_used=[],
        tools_unavailable=[],
    )


async def _task(inputs: dict) -> object:
    return await run_verifier(
        result=_wrap_rc(inputs["root_cause"]),
        run_evidence=inputs["run_evidence"],
        current_band=inputs["current_band"],
    )


async def main():
    dataset = build_dataset()
    report = await dataset.evaluate(_task, name="verifier_eval")

    print("\n" + "="*60)
    print("VERIFIER EVAL RESULTS")
    print("="*60)
    report.print()

    # Aggregate false-keep rate from report.cases[*].scores
    false_keep_total = 0
    contradicted_total = 0
    for case_result in report.cases:
        meta = case_result.metadata or {}
        if meta.get("category") in ("contradicted", "ambiguous"):
            contradicted_total += 1
            fk_score = case_result.scores.get("false_keep")
            if fk_score is not None and fk_score.value == 1.0:
                false_keep_total += 1

    print("\n── Calibration Rollup ──")
    if contradicted_total:
        fkr = false_keep_total / contradicted_total
        print(f"  false_keep_rate  : {fkr:.2%}  ({false_keep_total}/{contradicted_total})")
        print(f"  downgrade_recall : {1 - fkr:.2%}")
    else:
        print("  (no contradicted/ambiguous cases)")

    print()
    if false_keep_total == 0:
        print("✓ VERIFIER SAFE — zero false keeps. VERIFIER_ENABLED=true is justified.")
    else:
        print("✗ VERIFIER UNSAFE — false keeps present. Do NOT enable in production yet.")


if __name__ == "__main__":
    asyncio.run(main())
