"""Unit tests for the receipt-reading, downgrade-only verifier."""

from __future__ import annotations

import pytest

from agent.models import HealResult, RootCause, RunEvidence, ToolCallRecord
from agent.verification.verifier import VerifierVerdict, _band_rank, _serialize_receipts, run_verifier


# ── helpers ────────────────────────────────────────────────────────────────

def _result(file_path: str | None = "src/Foo.java") -> HealResult:
    return HealResult(
        issue_summary="test",
        investigation_steps=[],
        root_cause=RootCause(
            description="race condition in Foo",
            confidence=0.88,
            evidence=[],
            error_type="race_condition",
            file_path=file_path,
        ),
        recommended_fix="fix it",
        action_taken="explained",
        tools_used=[],
        tools_unavailable=[],
    )


def _ev_with_records() -> RunEvidence:
    ev = RunEvidence()
    ev.calls.append(ToolCallRecord(tool="query_traces", family="jaeger", success=True, service="ad"))
    ev.calls.append(ToolCallRecord(tool="query_logs", family="loki", success=True, service="ad"))
    ev.calls.append(ToolCallRecord(tool="get_file_contents", family="github", success=True, file_path="src/Foo.java"))
    return ev


# ── _serialize_receipts ────────────────────────────────────────────────────

def test_serialize_receipts_includes_calls():
    ev = _ev_with_records()
    text = _serialize_receipts(ev)
    assert "query_traces" in text
    assert "query_logs" in text
    assert "get_file_contents" in text
    assert "jaeger" in text


def test_serialize_receipts_includes_sandbox():
    ev = RunEvidence(sandbox_attempted=True, sandbox_reproduced=True, sandbox_confirmed_file="src/Foo.java")
    text = _serialize_receipts(ev)
    assert '"reproduced": true' in text
    assert "src/Foo.java" in text


# ── _band_rank ─────────────────────────────────────────────────────────────

def test_band_rank_ordering():
    assert _band_rank("low") < _band_rank("medium") < _band_rank("high")


# ── downgrade-only enforcement (no LLM call needed) ───────────────────────

@pytest.mark.asyncio
async def test_verifier_downgrade_lowers_band(monkeypatch: pytest.MonkeyPatch):
    async def fake_run(prompt, **kwargs):
        class FakeResult:
            output = VerifierVerdict(decision="downgrade", target_band="medium", reason="weak evidence")
        return FakeResult()

    import agent.verification.verifier as v_module
    monkeypatch.setattr(v_module._verifier_agent, "run", fake_run)

    result = _result()
    ev = _ev_with_records()
    verdict = await run_verifier(result, ev, "high")
    assert verdict.decision == "downgrade"
    assert verdict.target_band == "medium"


@pytest.mark.asyncio
async def test_verifier_keep_leaves_band_unchanged(monkeypatch: pytest.MonkeyPatch):
    async def fake_run(prompt, **kwargs):
        class FakeResult:
            output = VerifierVerdict(decision="keep", target_band="high", reason="evidence is solid")
        return FakeResult()

    import agent.verification.verifier as v_module
    monkeypatch.setattr(v_module._verifier_agent, "run", fake_run)

    result = _result()
    ev = _ev_with_records()
    verdict = await run_verifier(result, ev, "high")
    assert verdict.decision == "keep"
    assert verdict.target_band == "high"


@pytest.mark.asyncio
async def test_verifier_cannot_raise_band(monkeypatch: pytest.MonkeyPatch):
    """Even if verifier returns 'high', it should be clamped when current_band is 'medium'."""
    async def fake_run(prompt, **kwargs):
        class FakeResult:
            output = VerifierVerdict(decision="keep", target_band="high", reason="looks great")
        return FakeResult()

    import agent.verification.verifier as v_module
    monkeypatch.setattr(v_module._verifier_agent, "run", fake_run)

    result = _result()
    ev = _ev_with_records()
    verdict = await run_verifier(result, ev, "medium")
    assert _band_rank(verdict.target_band) <= _band_rank("medium")
