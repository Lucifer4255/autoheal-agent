"""FastAPI entrypoint and temporary UI test harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="AutoHeal AI")
STATIC_DIR = Path(__file__).parent / "static"

# Temporary in-memory session store for the pre-agent UI harness.
sessions: dict[str, dict[str, Any]] = {}


class SessionRequest(BaseModel):
    text: str = ""


class InvestigateRequest(BaseModel):
    session_id: str
    description: str
    service_name: str | None = None
    trace_id: str | None = None
    time_window_minutes: int = 10


def parse_key_values(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse setup-chat key/value lines without failing on free-form text."""
    parsed: dict[str, str] = {}
    ignored: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            ignored.append(line)
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()

    return parsed, ignored


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session")
async def create_session(request: SessionRequest) -> dict[str, Any]:
    values, ignored = parse_key_values(request.text)
    session_id = str(uuid4())
    sessions[session_id] = values

    configured = sorted(key for key, value in values.items() if value)
    needs_input = [] if values.get("repo") else ["repo"]
    unavailable = [
        name
        for name, key in {
            "github": "github_token",
            "tavily": "tavily_key",
            "e2b": "e2b_api_key",
        }.items()
        if not values.get(key)
    ]

    return {
        "session_id": session_id,
        "configured": configured,
        "unavailable": unavailable,
        "needs_input": needs_input,
        "ignored": ignored,
        "mode": "ui_harness",
    }


@app.post("/investigate")
async def investigate(request: InvestigateRequest) -> StreamingResponse:
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    async def stream():
        yield sse_event(
            {
                "type": "step",
                "round": 0,
                "tool": "ui_harness",
                "result_summary": "Received investigation request.",
                "confidence_after": 0.1,
            }
        )
        await asyncio.sleep(0.4)

        yield sse_event(
            {
                "type": "step",
                "round": 1,
                "tool": "fingerprint",
                "result_summary": "Mock hypothesis generated. Real fingerprint runs in Phase 2.",
                "confidence_after": 0.35,
            }
        )
        await asyncio.sleep(0.4)

        yield sse_event(
            {
                "type": "confidence",
                "value": 0.35,
                "note": "This is mock streaming only; agent tools are not wired yet.",
            }
        )
        await asyncio.sleep(0.4)

        yield sse_event(
            {
                "type": "final",
                "result": {
                    "issue_summary": request.description,
                    "root_cause": {
                        "description": (
                            "UI harness placeholder. Phase 1+ will return real HealResult."
                        ),
                        "file_path": None,
                        "line_number": None,
                        "confidence": 0.0,
                        "evidence": ["FastAPI POST streaming works."],
                        "error_type": "unknown",
                    },
                    "recommended_fix": (
                        "Continue implementation and wire server.py to agent.loop.investigate()."
                    ),
                    "action_taken": "explained",
                    "tools_used": ["ui_harness"],
                    "tools_unavailable": ["jaeger", "loki", "github", "web_search", "sandbox"],
                },
            }
        )

    return StreamingResponse(stream(), media_type="text/event-stream")
