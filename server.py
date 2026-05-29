"""FastAPI entrypoint — single /chat endpoint, SSE streaming."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import logfire
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent.loop import stream_investigate
from agent.models import AgentDeps, IssueContext

STATIC_DIR = Path(__file__).parent / "static"

sessions: dict[str, AgentDeps] = {}
session_history: dict[str, list] = {}
_http_client: httpx.AsyncClient | None = None

_GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+",
    re.IGNORECASE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logfire.configure()
    logfire.instrument_pydantic_ai()
    global _http_client
    _http_client = httpx.AsyncClient()
    yield
    await _http_client.aclose()
    _http_client = None


app = FastAPI(title="AutoHeal AI", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def extract_repo_url(text: str) -> str | None:
    """Pull the first GitHub URL out of the message, if any."""
    match = _GITHUB_URL_RE.search(text)
    return match.group(0) if match else None


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    session_id = request.session_id
    if session_id and session_id in sessions:
        deps = sessions[session_id]
    else:
        session_id = str(uuid4())
        deps = AgentDeps.from_env(http_client=_http_client)

    repo_url = extract_repo_url(request.message)
    if repo_url:
        deps = deps.apply_overrides({"repo": repo_url})

    sessions[session_id] = deps

    issue = IssueContext(description=request.message, service_name=deps.service_name)

    history = session_history.get(session_id)

    async def stream():
        try:
            async for event in stream_investigate(issue, deps, message_history=history):
                if event["type"] == "step":
                    yield sse_event(event)
                elif event["type"] == "result":
                    session_history[session_id] = event["messages"]
                    yield sse_event(
                        {
                            "type": "final",
                            "session_id": session_id,
                            "result": event["output"].model_dump(mode="json"),
                        }
                    )
        except Exception as exc:
            yield sse_event(
                {"type": "error", "session_id": session_id, "message": str(exc)}
            )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id},
    )
