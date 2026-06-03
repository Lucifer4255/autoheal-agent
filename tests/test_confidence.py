"""Unit tests for the evidence-gated confidence governor.

assess_allowed_band now reads RunEvidence.calls (code-stamped receipts),
not result.tools_used (LLM self-report).  Helpers build ledger entries directly.
"""

from __future__ import annotations

import pytest

from agent.verification.confidence import (
    BAND_RANK,
    assess_allowed_band,
    band_of_float,
    clamp_to_band,
)
from agent.models import (
    HealResult,
    RootCause,
    RunEvidence,
    ToolCallRecord,
)


# ── band_of_float ──────────────────────────────────────────────────────────

def test_band_low_boundary():
    assert band_of_float(0.0) == "low"
    assert band_of_float(0.54) == "low"


def test_band_medium_boundary():
    assert band_of_float(0.55) == "medium"
    assert band_of_float(0.84) == "medium"


def test_band_high_boundary():
    assert band_of_float(0.85) == "high"
    assert band_of_float(1.0) == "high"


# ── clamp_to_band ──────────────────────────────────────────────────────────

def test_clamp_high_is_noop():
    assert clamp_to_band(0.95, "high") == 0.95


def test_clamp_medium_caps_at_0_8():
    assert clamp_to_band(0.95, "medium") == 0.8


def test_clamp_low_caps_at_0_5():
    assert clamp_to_band(0.9, "low") == 0.5


def test_clamp_already_within_band():
    assert clamp_to_band(0.6, "medium") == 0.6


# ── helpers ────────────────────────────────────────────────────────────────

def _rc(**kw) -> RootCause:
    defaults = dict(description="test", confidence=0.9, evidence=[], error_type="runtime_error", file_path=None)
    defaults.update(kw)
    return RootCause(**defaults)


def _result(rc: RootCause) -> HealResult:
    return HealResult(
        issue_summary="test", investigation_steps=[], root_cause=rc,
        recommended_fix="fix it", action_taken="explained",
        tools_used=[], tools_unavailable=[],
    )


def _ev(*records: ToolCallRecord, **kw) -> RunEvidence:
    """Build a RunEvidence with the given ToolCallRecords + optional kwargs."""
    ev = RunEvidence(**kw)
    ev.calls.extend(records)
    return ev


def _jaeger(service: str = "ad") -> ToolCallRecord:
    return ToolCallRecord(tool="query_traces", family="jaeger", success=True, service=service)


def _loki(service: str = "ad") -> ToolCallRecord:
    return ToolCallRecord(tool="query_logs", family="loki", success=True, service=service)


def _github(file_path: str = "src/Foo.java") -> ToolCallRecord:
    return ToolCallRecord(tool="get_file_contents", family="github", success=True, file_path=file_path)


# ── assess_allowed_band — ledger-based ────────────────────────────────────

def test_empty_ledger_is_low():
    result = _result(_rc(error_type="unknown", file_path=None))
    band, msg = assess_allowed_band(result, RunEvidence())
    assert band == "low"
    assert "MEDIUM" in msg


def test_loki_only_is_low():
    result = _result(_rc())
    band, _ = assess_allowed_band(result, _ev(_loki()))
    assert band == "low"


def test_jaeger_only_is_low():
    result = _result(_rc())
    band, _ = assess_allowed_band(result, _ev(_jaeger()))
    assert band == "low"


def test_obs_corroboration_same_service_gives_medium():
    result = _result(_rc())
    band, msg = assess_allowed_band(result, _ev(_jaeger("ad"), _loki("ad")))
    assert band == "medium"
    assert "HIGH" in msg


def test_obs_corroboration_different_services_is_low():
    # Jaeger and Loki point at different services → services don't agree → no corroboration
    result = _result(_rc())
    band, _ = assess_allowed_band(result, _ev(_jaeger("ad"), _loki("cart")))
    assert band == "low"


def test_obs_corroboration_cosmetic_name_difference_still_agrees():
    # Jaeger service.name "cart" vs Loki compose_service "cartservice" are the same
    # service — normalization must let them corroborate (the LOW-clamp bug we hit).
    result = _result(_rc())
    band, _ = assess_allowed_band(result, _ev(_jaeger("cart"), _loki("cartservice")))
    assert band == "medium"

    band2, _ = assess_allowed_band(result, _ev(_jaeger("cart"), _loki("cart-service")))
    assert band2 == "medium"


def test_source_anchored_gives_medium():
    result = _result(_rc(file_path="src/Foo.java"))
    band, _ = assess_allowed_band(result, _ev(_github()))
    assert band == "medium"


def test_source_without_obs_is_medium():
    result = _result(_rc(file_path="src/Foo.java"))
    band, msg = assess_allowed_band(result, _ev(_jaeger(), _github()))
    assert band == "medium"
    assert "HIGH" in msg


def test_source_and_obs_same_service_is_high():
    result = _result(_rc(file_path="src/Foo.java"))
    band, msg = assess_allowed_band(result, _ev(_jaeger("ad"), _loki("ad"), _github()))
    assert band == "high"
    assert msg == ""


def test_source_and_sandbox_repro_is_high():
    result = _result(_rc(file_path="src/Foo.java"))
    ev = _ev(_github(), sandbox_reproduced=True)
    band, msg = assess_allowed_band(result, ev)
    assert band == "high"
    assert msg == ""


def test_unknown_error_type_never_high():
    result = _result(_rc(error_type="unknown", file_path="src/Foo.java"))
    ev = _ev(_jaeger("ad"), _loki("ad"), _github(), sandbox_reproduced=True)
    band, _ = assess_allowed_band(result, ev)
    assert band == "low"


def test_sandbox_reproduced_without_source_anchor_is_medium():
    result = _result(_rc(file_path=None))
    ev = _ev(_jaeger("ad"), _loki("ad"), sandbox_reproduced=True)
    band, _ = assess_allowed_band(result, ev)
    assert band == "medium"


def test_failed_tool_calls_do_not_count():
    result = _result(_rc(file_path="src/Foo.java"))
    ev = _ev(
        ToolCallRecord(tool="query_traces", family="jaeger", success=False, service="ad"),
        ToolCallRecord(tool="query_logs", family="loki", success=False, service="ad"),
        _github(),
    )
    band, _ = assess_allowed_band(result, ev)
    # No successful obs calls → obs_corroboration False → only source → medium
    assert band == "medium"


# ── BAND_RANK ordering ─────────────────────────────────────────────────────

def test_band_rank_order():
    assert BAND_RANK["low"] < BAND_RANK["medium"] < BAND_RANK["high"]
