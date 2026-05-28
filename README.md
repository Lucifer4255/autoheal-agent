# AutoHeal AI

Autonomous debugging agent built with Pydantic AI. Input: a production issue description plus session config. Output: structured root cause analysis with a suggested fix.

See `CONTEXT.md`, `ARCHITECTURE.md`, and `EXECUTION.md` for design and build plan.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Verify

```bash
ruff check .
ruff format --check .
python -m pytest
```
