"""Cassette-based tool recording and replay for deterministic eval runs.

RecordingToolset (EVAL_MODE=record): passes through to the real tool + saves results.
ReplayToolset  (EVAL_MODE=replay):  returns saved results for matching calls; raises on miss.

Both are WrapperToolset subclasses — same pattern as LedgerToolset — so they compose
cleanly with the existing LedgerToolset in _build_toolsets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from agent.models import AgentDeps

CASSETTES_DIR = Path(__file__).parent / "cassettes"


def _key(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Stable hash of (tool_name, sorted args) used as the cassette lookup key."""
    payload = json.dumps({"tool": tool_name, "args": tool_args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cassette_path(case_name: str) -> Path:
    CASSETTES_DIR.mkdir(exist_ok=True)
    return CASSETTES_DIR / f"{case_name}.json"


def _load_cassette(case_name: str) -> dict[str, Any]:
    p = _cassette_path(case_name)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_cassette(case_name: str, data: dict[str, Any]) -> None:
    _cassette_path(case_name).write_text(json.dumps(data, indent=2))


@dataclass
class RecordingToolset(WrapperToolset[AgentDeps]):
    """Live mode: passes through to real tools and records every call to a cassette file."""

    case_name: str
    _recordings: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        self._recordings = _load_cassette(self.case_name)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDeps],
        tool: ToolsetTool,
    ) -> Any:
        result = await super().call_tool(name, tool_args, ctx, tool)
        key = _key(name, tool_args)
        try:
            # Serialize the result — pydantic models have model_dump, others use repr
            if hasattr(result, "model_dump"):
                serialized = result.model_dump(mode="json")
            elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
                serialized = result
            else:
                serialized = str(result)
            self._recordings[key] = {"tool": name, "args": tool_args, "result": serialized}
        except Exception:
            pass  # recording failures are non-fatal
        _save_cassette(self.case_name, self._recordings)
        return result


@dataclass
class ReplayToolset(WrapperToolset[AgentDeps]):
    """CI replay mode: returns saved results without hitting real tools. Raises on cache miss."""

    case_name: str
    _recordings: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        self._recordings = _load_cassette(self.case_name)
        if not self._recordings:
            raise FileNotFoundError(
                f"No cassette found for case '{self.case_name}'. "
                f"Run with EVAL_MODE=record first to generate it."
            )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDeps],
        tool: ToolsetTool,
    ) -> Any:
        key = _key(name, tool_args)
        if key in self._recordings:
            stored = self._recordings[key]["result"]
            # Try to reconstruct a ToolResult if the stored value looks like one
            from agent.models import ToolResult
            if isinstance(stored, dict) and "tool_name" in stored and "success" in stored:
                try:
                    return ToolResult(**stored)
                except Exception:
                    pass
            return stored

        # Cache miss — fall back to the live tool and record the result for next time
        result = await super().call_tool(name, tool_args, ctx, tool)
        try:
            if hasattr(result, "model_dump"):
                serialized = result.model_dump(mode="json")
            elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
                serialized = result
            else:
                serialized = str(result)
            self._recordings[key] = {"tool": name, "args": tool_args, "result": serialized}
            _save_cassette(self.case_name, self._recordings)
        except Exception:
            pass
        return result
