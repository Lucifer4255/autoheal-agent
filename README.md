# AutoHeal Agent

Python application package for the AutoHeal autonomous debugging agent.

For project overview, setup, and usage, see the [root README](../README.md).

## Quick commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set OPENROUTER_API_KEY at minimum

uvicorn server:app --reload --port 8000
```

## Verify

```bash
ruff check .
ruff format --check .
python -m pytest
```

## Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — design and data models
- [CONTEXT.md](../CONTEXT.md) — build rules
- [EXECUTION.md](../EXECUTION.md) — build checklist
- [evals/README.md](evals/README.md) — evaluation suites
