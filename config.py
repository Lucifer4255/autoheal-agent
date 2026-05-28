"""Environment loading and configuration constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MODEL_NAME = os.getenv("MODEL_NAME", "openrouter/google/gemini-2.0-flash-001")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or None

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "10"))
STOP_CONFIDENCE = float(os.getenv("STOP_CONFIDENCE", "0.85"))
RETRY_CONFIDENCE = float(os.getenv("RETRY_CONFIDENCE", "0.4"))
FASTPATH_CONFIDENCE = float(os.getenv("FASTPATH_CONFIDENCE", "0.9"))

JAEGER_DEFAULT_URL = os.getenv("JAEGER_DEFAULT_URL", "http://localhost:8080/jaeger/ui")
LOKI_DEFAULT_URL = os.getenv("LOKI_DEFAULT_URL", "http://localhost:3100")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

SANDBOX_ERROR_TYPES: set[str] = {
    "code_logic",
    "runtime_error",
    "null_pointer",
    "index_out_of_bounds",
    "race_condition",
    "parsing_error",
    "serialization_error",
}

CAPABILITY_NAMES: tuple[str, ...] = ("jaeger", "loki", "github", "web_search", "sandbox")
