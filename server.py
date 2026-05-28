"""FastAPI entrypoint — single /chat endpoint, SSE streaming."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent.models import AgentDeps, IssueContext

STATIC_DIR = Path(__file__).parent / "static"

sessions: dict[str, AgentDeps] = {}
_http_client: httpx.AsyncClient | None = None

# Keys the message parser recognises as config — anything else is free-form issue text
_KNOWN_KEYS = {
    "jaeger_url", "loki_url", "jaeger_auth", "loki_auth",
    "github_token", "repo", "e2b_api_key", "e2b_key",
    "tavily_key", "tavily_api_key", "service_name",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient()
    yield
    await _http_client.aclose()
    _http_client = None


app = FastAPI(title="AutoHeal AI", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def parse_message(text: str) -> tuple[dict[str, str], str]:
    """Split a chat message into known key:value config pairs and free-form issue text.

    Lines whose key matches a known config field are extracted as config.
    Everything else (including lines with ':' in URLs or descriptions) stays as issue text.
    """
    kv: dict[str, str] = {}
    free_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip().lower() in _KNOWN_KEYS:
                kv[key.strip()] = value.strip()
                continue
        free_lines.append(line)

    return kv, "\n".join(free_lines)


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _setup_summary(deps: AgentDeps) -> str:
    configured = deps.configured_capabilities()
    needs = deps.needs_input()
    parts = []
    if configured:
        parts.append(f"Configured: {', '.join(configured)}.")
    if needs:
        parts.append(f"Still needed: {', '.join(needs)}.")
    else:
        parts.append("All set — describe your issue to start investigating.")
    return " ".join(parts)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    # Resolve or create session
    session_id = request.session_id
    if session_id and session_id in sessions:
        deps = sessions[session_id]
    else:
        session_id = str(uuid4())
        deps = AgentDeps.from_env(http_client=_http_client)

    kv_pairs, issue_text = parse_message(request.message)

    if kv_pairs:
        deps = deps.apply_overrides(kv_pairs)

    sessions[session_id] = deps

    async def stream():
        # Acknowledge any config updates first
        if kv_pairs:
            yield sse_event({
                "type": "setup",
                "session_id": session_id,
                "configured": deps.configured_capabilities(),
                "unavailable": deps.unavailable_capabilities(),
                "needs_input": deps.needs_input(),
                "message": _setup_summary(deps),
            })

        if issue_text:
            issue = IssueContext(
                description=issue_text,
                service_name=deps.service_name,
            )

            yield sse_event({
                "type": "step",
                "round": 0,
                "tool": "router",
                "result_summary": f"Investigating: {issue_text[:120]}",
                "confidence_after": 0.1,
            })

            yield sse_event({
                "type": "step",
                "round": 1,
                "tool": "session",
                "result_summary": (
                    f"Active capabilities: {', '.join(deps.configured_capabilities()) or 'none'}"
                ),
                "confidence_after": 0.2,
            })

            unavailable = deps.unavailable_capabilities()
            if unavailable:
                yield sse_event({
                    "type": "elicit",
                    "message": (
                        f"I can go deeper with: **{', '.join(unavailable)}**. "
                        "Paste the credentials in the chat to enable them and I'll continue."
                    ),
                    "wants": unavailable,
                })

            # TODO Phase 5: replace body above with agent.loop.investigate(issue, deps)
            yield sse_event({
                "type": "final",
                "session_id": session_id,
                "result": {
                    "issue_summary": issue_text,
                    "investigation_steps": [],
                    "root_cause": {
                        "description": "UI harness placeholder — real agent wires in Phase 5.",
                        "file_path": None,
                        "line_number": None,
                        "confidence": 0.0,
                        "evidence": ["Session and IssueContext are wired correctly."],
                        "error_type": "unknown",
                    },
                    "recommended_fix": (
                        "Continue with Phase 2 (fingerprint) and Phase 3 (capabilities)."
                    ),
                    "action_taken": "explained",
                    "tools_used": [],
                    "tools_unavailable": unavailable,
                },
            })

        elif not kv_pairs:
            # Empty or unrecognised message — greet and guide
            yield sse_event({
                "type": "setup",
                "session_id": session_id,
                "configured": deps.configured_capabilities(),
                "unavailable": deps.unavailable_capabilities(),
                "needs_input": deps.needs_input(),
                "message": (
                    "Hi! Describe a production issue to start investigating, "
                    "or paste credentials to configure tools (e.g. `repo: owner/name`)."
                ),
            })

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id},
    )
