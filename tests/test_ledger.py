"""Unit tests for the receipt ledger."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.toolsets.wrapper import WrapperToolset

from agent.verification.ledger import LedgerToolset, _classify, _normalize_github, _normalize_jaeger, _normalize_loki
from agent.models import AgentDeps, RunEvidence, ToolResult


# ── helpers ────────────────────────────────────────────────────────────────

def _make_deps() -> AgentDeps:
    import httpx
    return AgentDeps(
        jaeger_url=None, jaeger_auth=None,
        loki_url=None, loki_auth=None,
        github_token=None, repo=None,
        e2b_api_key=None, service_name=None,
        http_client=httpx.AsyncClient(),
        run_evidence=RunEvidence(),
    )


def _ctx(deps: AgentDeps) -> SimpleNamespace:
    return SimpleNamespace(deps=deps)


def _ledger_with_mock(return_value: Any, monkeypatch: pytest.MonkeyPatch) -> LedgerToolset:
    """Build a LedgerToolset whose parent call_tool is mocked to return return_value."""
    async def fake_super_call(self, name, tool_args, ctx, tool):
        return return_value
    monkeypatch.setattr(WrapperToolset, "call_tool", fake_super_call)
    return LedgerToolset(wrapped=None)  # wrapped unused since parent is mocked


# ── _classify ─────────────────────────────────────────────────────────────

def test_classify_jaeger():
    assert _classify("query_traces") == "jaeger"
    assert _classify("get_trace") == "jaeger"
    assert _classify("extract_error_spans") == "jaeger"


def test_classify_loki():
    assert _classify("query_logs") == "loki"
    assert _classify("get_log_context") == "loki"


def test_classify_github():
    assert _classify("get_file_contents") == "github"
    assert _classify("search_code") == "github"


def test_classify_other():
    assert _classify("reproduce_in_sandbox") == "other"
    assert _classify("unknown_tool") == "other"


# ── normalizers ────────────────────────────────────────────────────────────

def test_normalize_jaeger_success_with_file_hint():
    result = ToolResult(
        tool_name="extract_error_spans",
        success=True,
        data={
            "error_spans": [
                {"file_hint": "src/ad/AdService.java:42", "service": "ad"}
            ]
        },
    )
    record = _normalize_jaeger("extract_error_spans", result)
    assert record.success is True
    assert record.file_path == "src/ad/AdService.java"
    assert record.service == "ad"
    assert record.family == "jaeger"


def test_normalize_jaeger_failure():
    result = ToolResult(tool_name="query_traces", success=False, data={}, error="timeout")
    record = _normalize_jaeger("query_traces", result)
    assert record.success is False


def test_normalize_loki_extracts_service_from_logql():
    result = ToolResult(
        tool_name="query_logs",
        success=True,
        data={"lines": ["fail: something broke"]},
    )
    record = _normalize_loki("query_logs", result, {"query": '{compose_service="cart"} |= "fail"'})
    assert record.service == "cart"
    assert record.error_signal == "error_lines_present"
    assert record.family == "loki"


def test_normalize_loki_service_from_arg():
    result = ToolResult(tool_name="query_logs", success=True, data={})
    record = _normalize_loki("query_logs", result, {"service": "payment"})
    assert record.service == "payment"


def test_normalize_github_success_with_path():
    record = _normalize_github("get_file_contents", '{"path": "src/Foo.java", "content": "..."}', {})
    assert record.success is True
    assert record.file_path == "src/Foo.java"


def test_normalize_github_failure_string():
    record = _normalize_github("get_file_contents", "Tool 'get_file_contents' failed: Not Found", {})
    assert record.success is False


def test_normalize_github_fallback_to_args():
    record = _normalize_github("get_file_contents", '{"content": "..."}', {"path": "src/Bar.java"})
    assert record.file_path == "src/Bar.java"


# ── LedgerToolset integration ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ledger_records_successful_jaeger_call(monkeypatch: pytest.MonkeyPatch):
    deps = _make_deps()
    ctx = _ctx(deps)
    result = ToolResult(tool_name="query_traces", success=True, data={"service": "ad", "traces": []})
    toolset = _ledger_with_mock(result, monkeypatch)
    await toolset.call_tool("query_traces", {}, ctx, None)

    assert len(deps.run_evidence.calls) == 1
    rec = deps.run_evidence.calls[0]
    assert rec.tool == "query_traces"
    assert rec.family == "jaeger"
    assert rec.success is True


@pytest.mark.asyncio
async def test_ledger_records_failure_on_exception(monkeypatch: pytest.MonkeyPatch):
    async def fake_super_raise(self, name, tool_args, ctx, tool):
        raise RuntimeError("boom")
    monkeypatch.setattr(WrapperToolset, "call_tool", fake_super_raise)

    deps = _make_deps()
    ctx = _ctx(deps)
    toolset = LedgerToolset(wrapped=None)
    with pytest.raises(RuntimeError):
        await toolset.call_tool("query_traces", {}, ctx, None)

    assert len(deps.run_evidence.calls) == 1
    assert deps.run_evidence.calls[0].success is False


@pytest.mark.asyncio
async def test_ledger_accumulates_multiple_calls(monkeypatch: pytest.MonkeyPatch):
    deps = _make_deps()
    ctx = _ctx(deps)
    result = ToolResult(tool_name="query_logs", success=True, data={})
    toolset = _ledger_with_mock(result, monkeypatch)
    await toolset.call_tool("query_logs", {"service": "cart"}, ctx, None)
    await toolset.call_tool("get_log_context", {"service": "cart"}, ctx, None)

    assert len(deps.run_evidence.calls) == 2
    assert all(r.family == "loki" for r in deps.run_evidence.calls)


# ── RunEvidence helpers ────────────────────────────────────────────────────

def test_family_ok_true_when_successful_call_present():
    ev = RunEvidence()
    from agent.models import ToolCallRecord
    ev.calls.append(ToolCallRecord(tool="query_traces", family="jaeger", success=True))
    assert ev.family_ok("jaeger") is True
    assert ev.family_ok("loki") is False


def test_family_ok_false_when_only_failure():
    ev = RunEvidence()
    from agent.models import ToolCallRecord
    ev.calls.append(ToolCallRecord(tool="query_traces", family="jaeger", success=False))
    assert ev.family_ok("jaeger") is False


def test_services_seen_returns_set():
    ev = RunEvidence()
    from agent.models import ToolCallRecord
    ev.calls.append(ToolCallRecord(tool="query_traces", family="jaeger", success=True, service="ad"))
    ev.calls.append(ToolCallRecord(tool="get_trace", family="jaeger", success=True, service="ad"))
    assert ev.services_seen("jaeger") == {"ad"}
