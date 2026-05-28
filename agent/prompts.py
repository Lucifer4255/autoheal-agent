"""System and user prompt builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import config
from agent.models import AgentDeps, IssueContext

if TYPE_CHECKING:
    from agent.fingerprint import FingerprintMatch

SYSTEM_PROMPT_BASE = """You are AutoHeal, an autonomous debugging agent for production incidents.

Your job is to investigate an issue using the capabilities available in this session, identify the
most likely root cause, and recommend a fix. You must be evidence-driven and read-only.

Rules:
- Never write code, open pull requests, modify infrastructure, or claim to have applied a fix.
- Use observability evidence first when available: traces, then logs, then source code at anchored paths.
- Treat any pre-investigation hypothesis as unverified until supported by tool evidence.
- Populate RootCause.confidence honestly from the strength of the evidence you collected.
- If evidence is weak or conflicting, say so in confidence_note and keep confidence below 0.85.
- Only call tools that are available for this session.
- Prefer concise investigation. Stop once confidence is high enough to explain the issue clearly.
"""


def build_dynamic_prompt(deps: AgentDeps) -> str:
    configured = deps.configured_capabilities()
    unavailable = deps.unavailable_capabilities()

    lines = [
        "Session capability status:",
        f"- Available: {', '.join(configured) if configured else 'none'}",
        f"- Unavailable: {', '.join(unavailable) if unavailable else 'none'}",
    ]

    if deps.repo:
        lines.append(f"- Target repository: {deps.repo}")
    if deps.service_name:
        lines.append(f"- Default service name: {deps.service_name}")

    if "jaeger" in configured:
        lines.append("- Start with trace evidence when the issue may involve service errors or latency.")
    if "loki" in configured:
        lines.append("- Correlate logs using the service label and error terms from traces or the issue.")
    if "github" in configured:
        lines.append("- Read source only at file paths anchored by traces or strong log evidence.")
    if "web_search" in configured:
        lines.append("- Use web search only after observability and source evidence are insufficient.")
    if "sandbox" in configured:
        lines.append(
            "- Sandbox reproduction is optional and only for code/runtime hypotheses with a file anchor."
        )

    return "\n".join(lines)


def build_user_prompt(
    issue: IssueContext,
    fingerprint_match: FingerprintMatch | None = None,
) -> str:
    service = issue.service_name or "unspecified"
    lines = [
        "Investigate the following production issue.",
        "",
        f"Issue description:\n{issue.description}",
        "",
        f"Service name: {service}",
        f"Time window (minutes): {issue.time_window_minutes}",
    ]

    if issue.trace_id:
        lines.append(f"Trace ID: {issue.trace_id}")

    if fingerprint_match is not None:
        lines.extend(
            [
                "",
                "Pre-investigation hypothesis (unverified):",
                f"- Pattern: {fingerprint_match.pattern}",
                f"- Suggested error type: {fingerprint_match.error_type}",
                f"- Hypothesis: {fingerprint_match.hypothesis}",
                f"- Pattern confidence: {fingerprint_match.confidence:.2f}",
            ]
        )
        if fingerprint_match.confidence >= config.FASTPATH_CONFIDENCE:
            lines.append(
                "This is a high-confidence hint. Prioritize confirming it with Jaeger and Loki first, "
                "then GitHub if a file anchor appears. Do not finalize without verification."
            )

    lines.extend(
        [
            "",
            "Return a structured HealResult with evidence-backed root cause and recommended fix.",
        ]
    )
    return "\n".join(lines)
