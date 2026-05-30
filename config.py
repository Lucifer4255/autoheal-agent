"""Environment loading and configuration constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MODEL_NAME = os.getenv("MODEL_NAME") or None
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or None

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "100"))
STOP_CONFIDENCE = float(os.getenv("STOP_CONFIDENCE", "0.85"))
RETRY_CONFIDENCE = float(os.getenv("RETRY_CONFIDENCE", "0.4"))
FASTPATH_CONFIDENCE = float(os.getenv("FASTPATH_CONFIDENCE", "0.9"))

JAEGER_DEFAULT_URL = os.getenv("JAEGER_DEFAULT_URL", "http://localhost:8080/jaeger/ui")
LOKI_DEFAULT_URL = os.getenv("LOKI_DEFAULT_URL", "http://localhost:3100")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

BAND_MEDIUM_MIN = float(os.getenv("BAND_MEDIUM_MIN", "0.55"))
BAND_HIGH_MIN = float(os.getenv("BAND_HIGH_MIN", "0.85"))
BAND_CEILING: dict[str, float] = {
    "low": float(os.getenv("BAND_CEIL_LOW", "0.5")),
    "medium": float(os.getenv("BAND_CEIL_MEDIUM", "0.8")),
    "high": 1.0,
}
OUTPUT_RETRIES = int(os.getenv("OUTPUT_RETRIES", "3"))

SANDBOX_ERROR_TYPES: set[str] = {
    "code_logic",
    "runtime_error",
    "null_pointer",
    "index_out_of_bounds",
    "race_condition",
    "parsing_error",
    "serialization_error",
}

CAPABILITY_NAMES: tuple[str, ...] = ("jaeger", "loki", "github", "sandbox")

# Verifier sub-agent (Phase 3 — off by default until evals justify it)
VERIFIER_ENABLED = os.getenv("VERIFIER_ENABLED", "true").lower() == "true"
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "openrouter/deepseek/deepseek-v4-flash")
VERIFIER_MIN_BAND = os.getenv("VERIFIER_MIN_BAND", "medium")

# Online LLM judge — fires in the background on every agent run, streams pass/fail
# to Logfire's Live Evaluations page. Uses the same cheap model as the verifier.
ONLINE_EVAL_ENABLED = os.getenv("ONLINE_EVAL_ENABLED", "true").lower() == "true"
ONLINE_EVAL_MODEL = os.getenv("ONLINE_EVAL_MODEL", VERIFIER_MODEL)
