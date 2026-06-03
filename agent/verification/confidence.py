"""Evidence-gated confidence governor.

Band policy
-----------
HIGH   = source_anchored AND error_type_known AND (sandbox_reproduced OR obs_corroboration)
MEDIUM = error_type_known AND (obs_corroboration OR source_anchored)
LOW    = everything else (default)

source_anchored   : ledger has a successful github call AND a file_path was extracted
obs_corroboration : ledger has successful jaeger AND loki calls
                    (Phase 2b: services seen across both families must agree)
sandbox_reproduced: run_evidence.sandbox_reproduced — authoritative, set by the sandbox tool
error_type_known  : root_cause.error_type != "unknown"

All signals read from RunEvidence.calls (code-stamped receipts), NOT result.tools_used
(LLM self-report).  Empty ledger → every family_ok() is False → band falls to LOW.
"""

from __future__ import annotations

import config

# Tool-family sets are the single source of truth in ledger.py; import them here so
# confidence.py doesn't duplicate them.
from agent.verification.ledger import GITHUB_TOOLS, JAEGER_TOOLS, LOKI_TOOLS  # noqa: F401 (re-exported)
from agent.models import ConfidenceLevel, HealResult, RunEvidence, normalize_service_name

BAND_RANK: dict[ConfidenceLevel, int] = {"low": 0, "medium": 1, "high": 2}


def band_of_float(confidence: float) -> ConfidenceLevel:
    if confidence >= config.BAND_HIGH_MIN:
        return "high"
    if confidence >= config.BAND_MEDIUM_MIN:
        return "medium"
    return "low"


def clamp_to_band(confidence: float, band: ConfidenceLevel) -> float:
    ceiling = config.BAND_CEILING[band]
    return min(confidence, ceiling)


def evidence_signals(result: HealResult, run_evidence: RunEvidence) -> dict[str, object]:
    """Raw + derived evidence signals behind the band decision (pure, no side effects).

    Exposed so the governor can log exactly WHY a band was assigned — in particular
    whether `obs_corroboration` was killed by a Jaeger/Loki service-name mismatch
    (``services_compared=True`` but ``services_agree=False``).
    """
    rc = result.root_cause

    jaeger_ok = run_evidence.family_ok("jaeger")
    loki_ok = run_evidence.family_ok("loki")
    github_ok = run_evidence.family_ok("github")
    github_file = any(c.file_path for c in run_evidence.calls if c.family == "github" and c.success)

    jaeger_services = run_evidence.services_seen("jaeger")
    loki_services = run_evidence.services_seen("loki")

    both_obs = jaeger_ok and loki_ok
    # Service agreement only matters when both families actually named a service.
    # Compare canonicalized names so cosmetic differences (cart vs cartservice vs
    # cart-service) still count as the same service.
    services_compared = both_obs and bool(jaeger_services) and bool(loki_services)
    jaeger_norm = {normalize_service_name(s) for s in jaeger_services}
    loki_norm = {normalize_service_name(s) for s in loki_services}
    services_agree = (not services_compared) or bool(jaeger_norm & loki_norm)
    obs_corroboration = both_obs and services_agree

    return {
        "error_type_known": rc.error_type != "unknown",
        "jaeger_ok": jaeger_ok,
        "loki_ok": loki_ok,
        "github_ok": github_ok,
        "github_file": github_file,
        "sandbox_reproduced": run_evidence.sandbox_reproduced,
        "jaeger_services": sorted(jaeger_services),
        "loki_services": sorted(loki_services),
        "services_compared": services_compared,
        "services_agree": services_agree,
        "obs_corroboration": obs_corroboration,
        "source_anchored": github_ok and github_file,
    }


def assess_allowed_band(
    result: HealResult,
    run_evidence: RunEvidence,
) -> tuple[ConfidenceLevel, str]:
    """Return (max_allowed_band, missing_evidence_message).

    Reads from RunEvidence.calls (code-stamped receipts) — not result.tools_used.
    The message names exactly what's needed to reach the next band.
    """
    s = evidence_signals(result, run_evidence)
    error_type_known = bool(s["error_type_known"])
    obs_corroboration = bool(s["obs_corroboration"])
    source_anchored = bool(s["source_anchored"])
    sandbox_reproduced = bool(s["sandbox_reproduced"])

    if error_type_known and source_anchored and (sandbox_reproduced or obs_corroboration):
        return "high", ""

    if error_type_known and (obs_corroboration or source_anchored):
        missing: list[str] = []
        if not sandbox_reproduced and not obs_corroboration:
            missing.append("corroborate with both Jaeger and Loki")
        if not sandbox_reproduced:
            missing.append("run a sandbox repro to confirm the exact code path")
        if not source_anchored:
            missing.append("read the trace-anchored source file via GitHub")
        msg = (
            "To reach HIGH confidence: "
            + (" OR ".join(missing) if missing else "")
            + "."
        )
        return "medium", msg

    missing_parts: list[str] = []
    if not error_type_known:
        missing_parts.append("identify the error type (currently 'unknown')")
    if not obs_corroboration:
        missing_parts.append("collect evidence from both Jaeger (traces) and Loki (logs)")
    if not source_anchored:
        missing_parts.append("read the trace-anchored source file via GitHub")
    msg = "To reach MEDIUM confidence: " + "; ".join(missing_parts) + "."
    return "low", msg
