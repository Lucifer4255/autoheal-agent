"""Suite 1: end-to-end investigation eval runner.

Modes (EVAL_MODE env var):
  record  — live run, writes tool responses to cassettes/
  replay  — CI run, replays cassettes (only LLM varies)

Usage:
    EVAL_MODE=record uv run python -m evals.run_investigation
    EVAL_MODE=replay uv run python -m evals.run_investigation   # default
"""

from __future__ import annotations

import asyncio
import os

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

import httpx
from pydantic_evals import set_eval_attribute

from agent.loop import _build_toolsets, _github_repo_reachable
from agent.models import AgentDeps, IssueContext, RunEvidence
from agent.verification.ledger import LedgerToolset
from evals.datasets.flagd_cases import build_dataset
from evals.replay import RecordingToolset, ReplayToolset

EVAL_MODE = os.getenv("EVAL_MODE", "replay")  # record | replay


def _make_deps(http_client: httpx.AsyncClient) -> AgentDeps:
    return AgentDeps(
        jaeger_url=os.getenv("JAEGER_URL", "http://localhost:8080/jaeger/ui"),
        jaeger_auth=None,
        loki_url=os.getenv("LOKI_URL", "http://localhost:3100"),
        loki_auth=None,
        github_token=os.getenv("GITHUB_TOKEN"),
        repo=os.getenv("EVAL_REPO", "Lucifer4255/opentelemetry-demo"),
        e2b_api_key=os.getenv("E2B_API_KEY"),
        service_name=None,
        http_client=http_client,
        run_evidence=RunEvidence(),
    )


async def _build_eval_toolsets(deps: AgentDeps, case_name: str) -> list:
    """Build toolsets wrapped in LedgerToolset + cassette wrapper (record or replay)."""
    from agent.capabilities.github import GitHubCapability
    from agent.registry import build_capabilities

    capabilities = build_capabilities(deps)
    github_ok = await _github_repo_reachable(deps)
    toolsets = []

    for cap in capabilities:
        if isinstance(cap, GitHubCapability) and not github_ok:
            continue
        ts = cap.get_toolset()
        if ts is None:
            continue
        # Innermost: the real toolset
        # Middle: cassette wrapper (records or replays)
        # Outer: ledger (stamps receipts even during eval)
        if EVAL_MODE == "record":
            ts = RecordingToolset(wrapped=ts, case_name=case_name)
        else:
            try:
                ts = ReplayToolset(wrapped=ts, case_name=case_name)
            except FileNotFoundError as e:
                print(f"  [warn] {e} — running without replay for {case_name}")
        toolsets.append(LedgerToolset(wrapped=ts))

    return toolsets


async def _investigate_case(inputs: dict) -> object:
    """Task function: run a full investigation for one eval case."""
    from agent.core import agent
    from agent.fingerprint import fingerprint
    from agent.prompts import build_user_prompt
    from pydantic_ai.usage import UsageLimits

    case_name = inputs.get("_case_name", "unknown")

    async with httpx.AsyncClient() as http_client:
        deps = _make_deps(http_client)
        deps.run_evidence = RunEvidence()

        issue = IssueContext(
            description=inputs["description"],
            service_name=inputs.get("service_name"),
        )

        fp = fingerprint(issue.description)
        prompt = build_user_prompt(issue, fp)
        toolsets = await _build_eval_toolsets(deps, case_name)

        result = await agent.run(
            prompt,
            deps=deps,
            toolsets=toolsets,
            # Generous bound so cases don't crash mid-investigation. Over-investigation
            # is addressed via prompts/governor tuning, not a hard cap that throws.
            usage_limits=UsageLimits(request_limit=60),
        )

        # Store ledger call count as an attribute for EfficiencyEvaluator
        set_eval_attribute("ledger_call_count", len(deps.run_evidence.calls))

        # Print per-case summary so we can see actual classifications
        rc = result.output.root_cause
        print(
            f"\n  [{case_name}] error_type={rc.error_type} band={rc.confidence_level} "
            f"file={rc.file_path} tools={len(deps.run_evidence.calls)}"
        )

        return result.output


async def main():
    print(f"\nEVAL_MODE={EVAL_MODE}")
    dataset = build_dataset()

    # Inject case_name into inputs so the task fn can key the cassette
    for case in dataset.cases:
        case.inputs["_case_name"] = case.name

    report = await dataset.evaluate(_investigate_case, name=f"investigation_{EVAL_MODE}")

    print("\n" + "=" * 60)
    print("INVESTIGATION EVAL RESULTS")
    print("=" * 60)
    report.print()

    # ── Calibration rollup ─────────────────────────────────────────────────
    # precision@HIGH, overconfidence_rate, coverage@HIGH
    # "correct" is now decided by the LLM judge (a pass/fail assertion), which
    # holistically grades service + mechanism + evidence — replacing the old
    # ServiceMatch + ErrorTypeMatch exact-match scores.
    high_total = 0
    high_correct = 0
    correct_total = 0
    high_among_correct = 0

    def _judge_passed(case) -> bool:
        # LLMJudge emits an assertion; find it among the case assertions.
        for name, result in (case.assertions or {}).items():
            if "judge" in name.lower() or "llm" in name.lower():
                return bool(result.value)
        # Fallback: if exactly one assertion exists, use it
        if case.assertions and len(case.assertions) == 1:
            return bool(next(iter(case.assertions.values())).value)
        return False

    for c in report.cases:
        correct = _judge_passed(c)

        conf_score = c.scores.get("confidence_level_score")
        is_high = conf_score is not None and conf_score.value == 2.0  # high → 2

        if is_high:
            high_total += 1
            if correct:
                high_correct += 1
        if correct:
            correct_total += 1
            if is_high:
                high_among_correct += 1

    print("\n── Calibration Rollup ──")
    precision = high_correct / high_total if high_total else float("nan")
    overconf = (high_total - high_correct) / high_total if high_total else float("nan")
    coverage = high_among_correct / correct_total if correct_total else float("nan")

    print(f"  precision@HIGH    : {precision:.1%}  ({high_correct}/{high_total} HIGH cases correct)")
    print(f"  overconfidence    : {overconf:.1%}  (wrong HIGH / all HIGH)")
    print(f"  coverage@HIGH     : {coverage:.1%}  (HIGH among correct cases)")
    print(f"  correct cases     : {correct_total}/{len(report.cases)}")

    print()
    if high_total > 0 and overconf == 0.0:
        print("✓ ZERO overconfidence — precision@HIGH is 100%.")
    elif high_total == 0:
        print("△ No HIGH cases emitted — consider tuning confidence thresholds.")
    else:
        print(f"✗ Overconfidence present — {high_total - high_correct} wrong-HIGH case(s).")


if __name__ == "__main__":
    asyncio.run(main())
