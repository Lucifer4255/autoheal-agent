# AutoHeal Evals

Three eval suites built with `pydantic_evals` + Logfire.

## Suites

| Suite | What it tests | Headline metric |
|---|---|---|
| Suite 2 — Verifier | Receipt-reading downgrade-only LLM | false-keep rate (target: 0%) |
| Suite 3 — Sandbox | Sandbox sub-agent orchestration + reproduction | reproduced=True on fixture bugs; zero false repros on OTel flags |
| Suite 1 — Investigation | Whole agent end-to-end (cassettes or live) | precision@HIGH (target: 100%), overconfidence_rate (target: 0%) |

## Running

### Suite 2 — Verifier (cheapest, fully deterministic, no live services)
```bash
uv run python -m evals.run_verifier
```

### Suite 3 — Sandbox
```bash
# Tier A: mock E2B, checks orchestration only (no credentials needed)
uv run python -m evals.run_sandbox --mock

# Tier B: real E2B, deterministic fixture bugs (needs E2B_API_KEY)
uv run python -m evals.run_sandbox --fixture

# Tier C: real E2B, OTel flags — checks honest judgment (needs E2B_API_KEY + OTel stack)
uv run python -m evals.run_sandbox --otel
```

### Suite 1 — End-to-end investigation
```bash
# Step 1: record cassettes (live run against the real stack; writes to evals/cassettes/)
EVAL_MODE=record uv run python -m evals.run_investigation

# Step 2: replay cassettes (CI; only the LLM varies)
EVAL_MODE=replay uv run python -m evals.run_investigation
```

## Prompt experimentation

To compare two prompt variants:
1. Edit `agent/prompts.py` (or add a `PROMPT_VARIANT` env switch)
2. Run `EVAL_MODE=replay uv run python -m evals.run_investigation` for each variant
3. Compare the printed calibration rollup — precision@HIGH, overconfidence, coverage@HIGH
4. Logfire holds every run's prompt + tool calls + tokens + output as the experiment ledger

Hold model + temperature + (replayed) tool inputs fixed. The prompt is the only variable.

## Promoting metrics to a hard CI gate

Once `precision@HIGH` and `overconfidence_rate` are stable across replay runs, add to CI:

```bash
# Example CI step
result=$(EVAL_MODE=replay uv run python -m evals.run_investigation 2>&1)
echo "$result"
echo "$result" | grep -q "ZERO overconfidence" || (echo "OVERCONFIDENCE DETECTED" && exit 1)
```

## Environment variables

| Var | Default | Used by |
|---|---|---|
| `EVAL_MODE` | `replay` | Suite 1 — `record` for live run, `replay` for CI |
| `EVAL_REPO` | `Lucifer4255/opentelemetry-demo` | Suite 1 / 3 GitHub target |
| `GITHUB_TOKEN` | — | Suite 1 / 3 |
| `E2B_API_KEY` | — | Suite 3 Tier B / C |
| `JAEGER_URL` | `http://localhost:8080/jaeger/ui` | Suite 1 record mode |
| `LOKI_URL` | `http://localhost:3100` | Suite 1 record mode |
