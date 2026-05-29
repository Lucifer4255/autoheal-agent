"""Suite 2 verifier dataset — synthetic receipts + claimed root cause → expected verdict.

No live anything. Fully deterministic.
"""

from __future__ import annotations

from pydantic_evals import Case, Dataset

from agent.models import RootCause, RunEvidence, ToolCallRecord
from agent.verification.verifier import VerifierVerdict
from evals.evaluators import FalseKeepRate, VerifierVerdictMatch


def _jaeger(service: str = "ad") -> ToolCallRecord:
    return ToolCallRecord(tool="query_traces", family="jaeger", success=True, service=service)


def _loki(service: str = "ad") -> ToolCallRecord:
    return ToolCallRecord(tool="query_logs", family="loki", success=True, service=service)


def _github(file_path: str = "src/AdService.java") -> ToolCallRecord:
    return ToolCallRecord(tool="get_file_contents", family="github", success=True, file_path=file_path)


def _rc(file_path: str | None = "src/AdService.java", service_hint: str = "ad") -> RootCause:
    return RootCause(
        description=f"runtime error in {service_hint} service",
        confidence=0.9,
        evidence=[f"trace shows error in {service_hint}"],
        error_type="runtime_error",
        file_path=file_path,
    )


def _ev(*records: ToolCallRecord, **kw) -> RunEvidence:
    ev = RunEvidence(**kw)
    ev.calls.extend(records)
    return ev


_EVALUATORS = (VerifierVerdictMatch(), FalseKeepRate())


def build_dataset() -> Dataset:
    return Dataset(cases=[
        # ── SUPPORTED cases → expect keep ─────────────────────────────────
        Case(
            name="supported_full_stack",
            inputs={
                "root_cause": _rc(),
                "run_evidence": _ev(_jaeger("ad"), _loki("ad"), _github()),
                "current_band": "high",
            },
            expected_output={"decision": "keep", "target_band": "high"},
            metadata={"category": "supported"},
            evaluators=_EVALUATORS,
        ),
        Case(
            name="supported_with_sandbox_repro",
            inputs={
                "root_cause": _rc(),
                "run_evidence": _ev(
                    _github(),
                    sandbox_reproduced=True,
                    sandbox_confirmed_file="src/AdService.java",
                ),
                "current_band": "high",
            },
            expected_output={"decision": "keep", "target_band": "high"},
            metadata={"category": "supported"},
            evaluators=_EVALUATORS,
        ),
        # ── CONTRADICTED cases → expect downgrade ─────────────────────────
        Case(
            name="service_mismatch_jaeger_vs_loki",
            inputs={
                "root_cause": _rc(service_hint="ad"),
                "run_evidence": _ev(_jaeger("ad"), _loki("cart")),  # different services
                "current_band": "medium",
            },
            expected_output={"decision": "downgrade", "target_band": "low"},
            metadata={"category": "contradicted"},
            evaluators=_EVALUATORS,
        ),
        Case(
            name="claimed_file_not_in_receipts",
            inputs={
                "root_cause": _rc(file_path="src/NonExistentFile.java"),
                "run_evidence": _ev(_jaeger("ad"), _loki("ad"), _github("src/AdService.java")),
                "current_band": "high",
            },
            expected_output={"decision": "downgrade", "target_band": "medium"},
            metadata={"category": "contradicted"},
            evaluators=_EVALUATORS,
        ),
        Case(
            name="sandbox_not_reproduced",
            inputs={
                "root_cause": _rc(),
                "run_evidence": _ev(_github(), sandbox_attempted=True, sandbox_reproduced=False),
                "current_band": "medium",
            },
            expected_output={"decision": "downgrade", "target_band": "low"},
            metadata={"category": "contradicted"},
            evaluators=_EVALUATORS,
        ),
        # ── AMBIGUOUS / THIN cases → expect downgrade ─────────────────────
        Case(
            name="single_weak_receipt_only",
            inputs={
                "root_cause": _rc(),
                "run_evidence": _ev(_loki("ad")),  # only one log query
                "current_band": "medium",
            },
            expected_output={"decision": "downgrade", "target_band": "low"},
            metadata={"category": "ambiguous"},
            evaluators=_EVALUATORS,
        ),
        Case(
            name="no_receipts_at_all",
            inputs={
                "root_cause": _rc(),
                "run_evidence": RunEvidence(),
                "current_band": "medium",
            },
            expected_output={"decision": "downgrade", "target_band": "low"},
            metadata={"category": "ambiguous"},
            evaluators=_EVALUATORS,
        ),
    ])
