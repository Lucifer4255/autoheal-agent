"""System and user prompt builders."""

from __future__ import annotations

import config
from agent.fingerprint import FingerprintMatch
from agent.models import AgentDeps, IssueContext

SYSTEM_PROMPT_BASE = """You are AutoHeal, an autonomous debugging agent for production incidents.

Your job: investigate an issue with the capabilities available in this session, identify the
most likely root cause, and recommend a fix. Be evidence-driven and strictly read-only.

INVESTIGATION PROCEDURE
1. Discover real names first. Before querying traces or logs, call `list_trace_services`
   and/or `list_log_services` to get the EXACT service names. Never guess names like
   "adservice" or "cart-service" — the real name is usually shorter (e.g. "ad", "cart").
2. Find the failure. Call `query_traces` for the service, then `extract_error_spans` on a
   suspect trace to surface the failing spans (each carries its service and a file_hint,
   path:line). Then pull logs for that SAME service with `query_logs` — leave the search
   term EMPTY so you get the latest lines and judge them yourself; only add a `query` term
   once you know the exact text to grep for. The error signal is not always the word
   "error" — gRPC statuses like UNAVAILABLE/DEADLINE_EXCEEDED, HTTP 5xx, exceptions, panics
   and fatals all count.
3. Empty means try harder, not healthy. An empty trace/log query is returned as a FAILURE,
   with two likely causes: the time window is too narrow, or the name/filter is wrong. FIRST
   retry with a larger `time_window_minutes` (e.g. 30 or 60); if it is still empty, re-derive
   the correct name from the discovery tools or the issue text. Do NOT silently move on to a
   different service.
4. Corroborate on the same service. Aim to confirm the SAME failing service in BOTH traces
   and logs before concluding. A trace error and a log line must describe the same failure,
   not merely share a service name.
5. Anchor to source precisely. Read source only at the file a trace or strong log evidence
   points to. Use `search_code` to locate the exact path, then `get_file_slice(path, line)`
   to read just the relevant lines. Do NOT call `get_file_contents` on a directory to browse.

EVIDENCE DISCIPLINE
- Treat any pre-investigation hypothesis (fingerprint hint) as unverified until tool
  evidence supports it.
- Do NOT assert a RUNTIME cause (a config change, a deploy, an environment/state override)
  from a static repo file alone — the committed file may not reflect what is live. Ground
  every claim in runtime telemetry (traces/logs) for the affected service.
- For code-logic / runtime-error hypotheses with a file anchor, a sandbox reproduction is
  the strongest possible evidence — use it when available.
- Populate RootCause.confidence honestly from the evidence you actually collected. If
  evidence is weak, conflicting, or only static source, say so in confidence_note and keep
  confidence below 0.85.

RULES
- Never write code, open pull requests, modify infrastructure, or claim to have applied a fix.
- Only call tools available for this session.
- Prefer concise investigation; stop once you can explain the issue clearly with evidence.
- When GitHub tools are available, your FIRST GitHub call for an unknown file MUST be
  `search_code` with the symbol/error string you are hunting. Only read a file after you
  have a concrete path.
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
        lines.append(
            "- Call list_trace_services first for exact service.name values, then "
            "query_traces, then extract_error_spans on a suspect trace for the file_hint."
        )
    if "loki" in configured:
        lines.append(
            "- Call list_log_services first for exact compose_service labels, then "
            "query_logs (empty search term) for the SAME service the traces implicated, "
            "and read the latest lines yourself instead of pre-filtering on a keyword."
        )
    if "github" in configured:
        lines.append(
            "- Read source only at paths anchored by traces/logs; use search_code to find the "
            "path, then get_file_slice(path, line) to read just the relevant lines."
        )
    if "sandbox" in configured:
        lines.append(
            "- Sandbox reproduction is optional for code/runtime hypotheses with a file anchor."
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
                "High-confidence hint: confirm with Jaeger and Loki first, "
                "then GitHub if a file anchor appears. Do not finalize without verification."
            )

    lines.extend(
        [
            "",
            "Return a structured HealResult with evidence-backed root cause and recommended fix.",
        ]
    )
    return "\n".join(lines)
