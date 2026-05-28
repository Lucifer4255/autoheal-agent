"""Tests for agent.fingerprint."""

from __future__ import annotations

import pytest

from agent.fingerprint import _COMPILED, PATTERNS, fingerprint


@pytest.mark.parametrize(
    ("pattern_name", "sample_text"),
    [
        ("oom_killed", "Pod was OOMKilled after exceeding memory limit"),
        ("econnrefused", "dial tcp: ECONNREFUSED connecting to payment:8080"),
        ("deadline_exceeded", "rpc error: context deadline exceeded"),
        # Isolated nil_pointer text — does not contain 'runtime error' or 'panic:'
        ("nil_pointer", "NullPointerException in CartService at line 42"),
        # tls_error: expiry-specific text, not a bare 'x509' which would be too broad
        ("tls_error", "x509: certificate has expired or is not yet valid"),
        ("runtime_panic", "panic: runtime error: index out of range"),
    ],
)
def test_each_pattern_fires(pattern_name: str, sample_text: str) -> None:
    # First assert the target pattern's regex actually matches the sample text
    assert _COMPILED[pattern_name].search(sample_text), (
        f"Pattern '{pattern_name}' did not match sample text"
    )
    # Then assert it wins the overall fingerprint (no higher-confidence pattern steals it)
    result = fingerprint(sample_text)
    assert result is not None
    assert result.pattern == pattern_name
    assert result.confidence == PATTERNS[pattern_name][1]
    assert result.error_type == PATTERNS[pattern_name][2]
    assert result.hypothesis == PATTERNS[pattern_name][3]


@pytest.mark.parametrize(
    "clean_text",
    [
        "The checkout page is slow today",
        "Users report intermittent 503 errors on the homepage",
        "Deployment completed successfully at 14:00 UTC",
        "Need help tuning database connection pool size",
        # Word-boundary check: 'zoom' must not trigger oom_killed
        "ZOOM conference call dropped unexpectedly",
        # Bare 'x509' without expiry context must not fire tls_error
        "x509: certificate signed by unknown authority",
    ],
)
def test_no_false_positives_on_unrelated_text(clean_text: str) -> None:
    assert fingerprint(clean_text) is None


def test_highest_confidence_match_wins() -> None:
    # OOM (0.92) beats runtime_panic (0.85) and nil_pointer beats runtime_panic
    text = "Container OOMKilled after panic: runtime error in worker"
    result = fingerprint(text)
    assert result is not None
    assert result.pattern == "oom_killed"
    assert result.confidence == 0.92


def test_nil_pointer_beats_runtime_panic_when_both_match() -> None:
    # This text intentionally triggers both nil_pointer (0.90) and runtime_panic (0.85)
    text = "panic: runtime error: invalid memory address or nil pointer dereference"
    assert _COMPILED["nil_pointer"].search(text)
    assert _COMPILED["runtime_panic"].search(text)
    result = fingerprint(text)
    assert result is not None
    assert result.pattern == "nil_pointer"
    assert result.confidence == 0.90


def test_empty_text_returns_none() -> None:
    assert fingerprint("") is None
