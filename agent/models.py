"""Pydantic and dataclass models for AutoHeal."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Literal

import httpx
from pydantic import BaseModel, Field

import config

ConfidenceLevel = Literal["high", "medium", "low"]

ErrorType = Literal[
    "code_logic",
    "runtime_error",
    "null_pointer",
    "index_out_of_bounds",
    "race_condition",
    "parsing_error",
    "serialization_error",
    "config_error",
    "infra",
    "oom",
    "tls",
    "network",
    "unknown",
]

ActionTaken = Literal["explained", "fast_path", "sandbox_enriched"]


@dataclass
class ToolCallRecord:
    """One stamped receipt from a real tool execution."""
    tool: str
    family: str                     # jaeger | loki | github | other
    success: bool
    service: str | None = None      # extracted service name (for cross-source agreement)
    file_path: str | None = None    # extracted file path
    error_signal: str | None = None # extracted error hint


@dataclass
class RunEvidence:
    """Per-run evidence ledger. Reset by the loop before each agent.run call."""
    calls: list[ToolCallRecord] = field(default_factory=list)
    sandbox_attempted: bool = False
    sandbox_reproduced: bool = False        # authoritative HIGH kill-shot
    sandbox_confirmed_file: str | None = None
    overclaim_retried: bool = False         # single-fire guard

    def family_ok(self, family: str) -> bool:
        """True if at least one successful tool call belongs to this family."""
        return any(c.family == family and c.success for c in self.calls)

    def services_seen(self, family: str) -> set[str]:
        """Set of service names seen in successful calls of this family."""
        return {c.service for c in self.calls if c.family == family and c.success and c.service}

# Matches only owner/repo — rejects deep paths like /tree/main/src
_REPO_URL_PATTERN = re.compile(r"^https?://(?:www\.)?github\.com/([^/]+/[^/]+?)/?$")


@dataclass
class AgentDeps:
    jaeger_url: str | None
    jaeger_auth: str | None
    loki_url: str | None
    loki_auth: str | None
    github_token: str | None
    repo: str | None
    e2b_api_key: str | None
    service_name: str | None
    http_client: httpx.AsyncClient
    run_evidence: RunEvidence = field(default_factory=RunEvidence)

    @classmethod
    def from_env(cls, http_client: httpx.AsyncClient | None = None) -> AgentDeps:
        return cls(
            jaeger_url=_env_or_default("JAEGER_URL", config.JAEGER_DEFAULT_URL),
            jaeger_auth=_empty_to_none(os.getenv("JAEGER_AUTH")),
            loki_url=_env_or_default("LOKI_URL", config.LOKI_DEFAULT_URL),
            loki_auth=_empty_to_none(os.getenv("LOKI_AUTH")),
            github_token=_empty_to_none(os.getenv("GITHUB_TOKEN")),
            repo=None,
            e2b_api_key=_empty_to_none(os.getenv("E2B_API_KEY")),
            service_name=None,
            http_client=http_client if http_client is not None else httpx.AsyncClient(),
        )

    def apply_overrides(self, values: dict[str, str]) -> AgentDeps:
        """Merge setup-chat key/value overrides onto this session deps."""
        field_map = {
            "jaeger_url": "jaeger_url",
            "loki_url": "loki_url",
            "jaeger_auth": "jaeger_auth",
            "loki_auth": "loki_auth",
            "github_token": "github_token",
            "repo": "repo",
            "e2b_api_key": "e2b_api_key",
            "e2b_key": "e2b_api_key",
            "service_name": "service_name",
        }
        updates: dict[str, str | None] = {}
        for key, value in values.items():
            field = field_map.get(key.strip().lower())
            if field is None:
                continue
            cleaned = value.strip()
            if not cleaned:
                # Empty value means "no change" — don't clear an env-sourced credential
                continue
            if field == "repo":
                cleaned = normalize_repo(cleaned)
            updates[field] = cleaned or None

        return replace(self, **updates)

    def configured_capabilities(self) -> list[str]:
        configured: list[str] = []
        if self.jaeger_url:
            configured.append("jaeger")
        if self.loki_url:
            configured.append("loki")
        if self.github_token and self.repo:
            configured.append("github")
        if self.e2b_api_key and self.github_token and self.repo:
            configured.append("sandbox")
        return configured

    def unavailable_capabilities(self) -> list[str]:
        configured = set(self.configured_capabilities())
        return [name for name in config.CAPABILITY_NAMES if name not in configured]

    def needs_input(self) -> list[str]:
        needs: list[str] = []
        if not self.repo:
            needs.append("repo")
        return needs


class IssueContext(BaseModel):
    description: str
    service_name: str | None = None
    trace_id: str | None = None
    time_window_minutes: int = 10


class RootCause(BaseModel):
    description: str
    file_path: str | None = None
    line_number: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = "low"
    evidence: list[str]
    error_type: ErrorType


class InvestigationStep(BaseModel):
    round: int
    tool: str
    result_summary: str
    confidence_after: float


class HealResult(BaseModel):
    issue_summary: str
    investigation_steps: list[InvestigationStep]
    root_cause: RootCause
    recommended_fix: str
    fix_code_snippet: str | None = None
    action_taken: ActionTaken
    tools_used: list[str]
    tools_unavailable: list[str]
    confidence_note: str | None = None


class SandboxResult(BaseModel):
    reproduced: bool
    confirmed_file: str | None = None
    confirmed_line: int | None = None
    stdout: str
    stderr: str
    exit_code: int
    repro_script: str
    attempts: int
    skip_reason: str | None = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: dict
    error: str | None = None


def normalize_repo(value: str) -> str:
    """Accept owner/repo or a full GitHub URL."""
    value = value.strip().strip("/")
    match = _REPO_URL_PATTERN.match(value)
    if match:
        return match.group(1).strip("/")
    return value


def normalize_service_name(name: str) -> str:
    """Canonicalize a service / compose-label name for cross-source matching.

    Jaeger's `service.name` and Loki's `compose_service` label often differ only
    cosmetically (cart / cartservice / cart-service / cart_service). Lowercase, strip
    non-alphanumerics, and drop a trailing 'service' so they compare equal.
    """
    cleaned = re.sub(r"[^a-z0-9]", "", name.lower())
    if cleaned.endswith("service") and cleaned != "service":
        cleaned = cleaned[: -len("service")]
    return cleaned


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_or_default(name: str, default: str) -> str:
    value = _empty_to_none(os.getenv(name))
    return value or default
