"""Pre-tool pattern matcher for fast-path diagnosis."""

from __future__ import annotations

import re

from pydantic import BaseModel

from agent.models import ErrorType

# (regex, confidence, error_type, hypothesis)
PATTERNS: dict[str, tuple[str, float, ErrorType, str]] = {
    "oom_killed": (
        r"OOMKilled|out of memory|\bOOM\b",
        0.92,
        "oom",
        "Container or process was killed due to memory exhaustion.",
    ),
    "econnrefused": (
        r"ECONNREFUSED|connection refused",
        0.88,
        "network",
        "A downstream service or host refused the connection.",
    ),
    "deadline_exceeded": (
        r"context deadline exceeded|deadline.{0,10}exceeded",
        0.85,
        "runtime_error",
        "An upstream call timed out before completing.",
    ),
    "nil_pointer": (
        r"nil pointer|null pointer|NullPointerException",
        0.90,
        "null_pointer",
        "Code dereferenced a null or nil value.",
    ),
    "tls_error": (
        r"certificate expired|x509.*expir|x509.*not yet valid|TLS handshake",
        0.90,
        "tls",
        "TLS certificate error — may be expired, not yet valid, or a handshake failure.",
    ),
    "runtime_panic": (
        r"runtime error|panic:",
        0.85,
        "runtime_error",
        "The service hit an unhandled runtime panic or fatal error.",
    ),
}

_COMPILED: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE) for name, (pattern, *_rest) in PATTERNS.items()
}


class FingerprintMatch(BaseModel):
    pattern: str
    confidence: float
    error_type: ErrorType
    hypothesis: str


def fingerprint(text: str) -> FingerprintMatch | None:
    """Return the highest-confidence pattern match, or None if nothing matched."""
    best: FingerprintMatch | None = None

    for name, (_regex, confidence, error_type, hypothesis) in PATTERNS.items():
        if not _COMPILED[name].search(text):
            continue
        match = FingerprintMatch(
            pattern=name,
            confidence=confidence,
            error_type=error_type,
            hypothesis=hypothesis,
        )
        if best is None or match.confidence > best.confidence:
            best = match

    return best
