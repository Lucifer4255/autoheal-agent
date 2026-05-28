"""Main agent and result validation."""

from __future__ import annotations

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

import config
from agent.models import AgentDeps, HealResult
from agent.prompts import SYSTEM_PROMPT_BASE, build_dynamic_prompt


def _make_model() -> OpenRouterModel:
    """Build an OpenRouter model from config.

    MODEL_NAME may be 'openrouter/google/gemini-2.0-flash-001' (env format)
    or 'google/gemini-2.0-flash-001' (slug format). Strip the prefix if present.
    """
    slug = config.MODEL_NAME.removeprefix("openrouter/")
    return OpenRouterModel(
        slug,
        provider=OpenRouterProvider(api_key=config.OPENROUTER_API_KEY or "no-key-set"),
    )

# ---------------------------------------------------------------------------
# Main investigation agent
# ---------------------------------------------------------------------------
# Contract for capabilities:
#   - Each capability implements AbstractCapability[AgentDeps]
#   - get_toolset() returns an AgentToolset or None (if not configured)
#   - Tool functions take ctx: RunContext[AgentDeps] as first arg
#   - All tool functions return ToolResult — never raise
#
# Contract for the loop:
#   - Call agent.run(prompt, deps=deps, toolsets=capabilities, usage_limits=...)
#   - Access result.output → HealResult
# ---------------------------------------------------------------------------

agent: Agent[AgentDeps, HealResult] = Agent(
    _make_model(),
    deps_type=AgentDeps,
    output_type=HealResult,
    instructions=SYSTEM_PROMPT_BASE,
)


@agent.instructions
def dynamic_instructions(ctx: RunContext[AgentDeps]) -> str:
    """Appended each run — tells the agent exactly which tools are available."""
    return build_dynamic_prompt(ctx.deps)


@agent.output_validator
async def validate_confidence(ctx: RunContext[AgentDeps], result: HealResult) -> HealResult:
    """Reject results whose confidence is too low — forces the agent to keep investigating."""
    if result.root_cause.confidence < config.RETRY_CONFIDENCE:
        raise ModelRetry(
            f"Root cause confidence {result.root_cause.confidence:.2f} is below the minimum "
            f"{config.RETRY_CONFIDENCE}. Collect more evidence before returning a result."
        )
    return result
