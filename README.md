# AutoHeal Agent

> Autonomous debugging agent for production incidents. Describe a failure in plain language; AutoHeal correlates traces, logs, and source code, then returns a structured root cause analysis with a recommended fix — without ever touching your code or infrastructure.

Built with [Pydantic AI](https://ai.pydantic.dev), FastAPI, and a strictly read-only investigation model.

**Demo:** [Watch AutoHeal investigate a live failure (YouTube)](https://youtu.be/h22gYUet604)

---

## 1. What is AutoHeal?

When a production service starts misbehaving, an on-call engineer does roughly this:

1. Reads the alert / vague bug report ("ad service is throwing errors since 10am").
2. Opens the tracing UI, finds the failing spans, reads the error message.
3. Cross-references the logs around that timestamp for the service.
4. Jumps to the source file/line the trace points at and reads the code.
5. Forms a hypothesis, sometimes reproduces it locally, then proposes a fix.

AutoHeal automates that loop. You give it an issue description (and whatever credentials you have wired up), and it returns a **`HealResult`** — a structured object containing:

- **Root cause** — description, file/line, error classification, and a confidence band (HIGH / MEDIUM / LOW) backed by evidence.
- **Recommended fix** — prose plus an optional code snippet.
- **Investigation timeline** — every tool it called, what each found, and the confidence after each step.
- **Capability accounting** — `tools_used` vs `tools_unavailable`, so you always know what the agent could and couldn't see.

The reference target is the [OpenTelemetry "Astronomy Shop" demo](https://github.com/open-telemetry/opentelemetry-demo) (a realistic polyglot microservices app with built-in fault-injection flags), but every integration is gated on session config — the same agent runs against a local demo or a production observability stack.

### Design principles

| Principle | What it means in practice |
|---|---|
| **Trace-anchored** | Investigation starts from observability data. Source is read only at the line numbers traces point to — never a blind codebase search. |
| **Graceful degradation** | Every capability is optional and self-checks its config. One capability still yields a coherent `HealResult`; missing ones land in `tools_unavailable`. |
| **Evidence-gated confidence** | The model cannot *assert* HIGH confidence. A governor computes the maximum band the collected tool receipts support and clamps the claim to it. |
| **Read-only** | No commits, PRs, or migrations. The only place code executes is a throwaway E2B sandbox, purely to reproduce a bug. |

---

## 2. Architecture

### 2.1 Big picture

```
                         ┌──────────────────────────────────────┐
   Browser chat UI  ───► │  FastAPI server (server.py)          │
   (SSE stream)     ◄─── │  POST /chat — parses config + issue  │
                         └───────────────┬──────────────────────┘
                                         │ AgentDeps (session config)
                                         ▼
                         ┌──────────────────────────────────────┐
                         │  Investigation loop (loop.py)         │
                         │  • resets per-run evidence ledger     │
                         │  • wraps toolsets in LedgerToolset    │
                         └───────────────┬──────────────────────┘
                                         ▼
                         ┌──────────────────────────────────────┐
                         │  Main agent (core.py)                 │
                         │  Pydantic AI Agent                    │
                         │  output_type = HealResult             │
                         │  output_validator = govern_confidence │
                         └──┬─────────┬─────────┬─────────┬──────┘
                            ▼         ▼         ▼         ▼
                        Jaeger     Loki     GitHub     Source     ── capabilities
                       (traces)   (logs)   (MCP read)  (slice)       (registry.py)
                            │                              │
                            └──────────────┐   ┌───────────┘
                                           ▼   ▼
                                  ┌────────────────────────┐
                                  │ Sandbox sub-agent      │  ── delegated, conditional
                                  │ (subagents/sandbox.py) │     output_type = SandboxResult
                                  │ E2B container          │     clone + run repro
                                  └────────────────────────┘
```

It is deliberately a **single agent with one delegated sub-agent**, not a multi-agent graph. The sandbox is the only legitimate delegation boundary: it has a different goal (write the smallest repro), a different output type (`SandboxResult`), and a different environment (an isolated E2B container).

### 2.2 Components

**Web layer (`server.py`)** — A FastAPI app with a single `POST /chat` endpoint that streams Server-Sent Events. Each incoming message is parsed for `key: value` config pairs (credentials, URLs, target service) and free-form issue text. Config mutates the session's `AgentDeps`; issue text triggers an investigation. Both can appear in one message. Sessions are in-memory, keyed by an `X-Session-Id` header that the browser persists in `localStorage`.

**Investigation loop (`agent/loop.py`)** — The thin wrapper around `agent.run`/`agent.iter`. Before each run it resets the per-run `RunEvidence` ledger and wraps each capability toolset in a `LedgerToolset` so every tool call is recorded as a *receipt* (which tool family ran, with what result). Those receipts are what the confidence governor reads — confidence is earned from observed tool calls, not from the model's prose.

**Main agent (`agent/core.py`)** — One Pydantic AI `Agent`:
- `deps_type = AgentDeps` — session config injected via `RunContext`.
- `output_type = HealResult` — the structured final answer; the agent literally cannot return without both a `root_cause` and a `recommended_fix`.
- `@agent.output_validator govern_confidence` — the evidence-gated confidence stage (§2.4).
- `@agent.instructions` (dynamic) — the system prompt is rebuilt per session to tell the model exactly which capabilities are wired up, so it never calls a tool that isn't there.

**Capabilities (`agent/capabilities/`, assembled by `registry.py`)** — Each subclasses Pydantic AI's `AbstractCapability` and implements `get_toolset() -> AgentToolset | None`, returning `None` when its config is absent. `build_capabilities(deps)` assembles the active set before the run:

| Capability | Requires | Provides |
|---|---|---|
| **Jaeger** | `jaeger_url` | `query_traces`, `get_trace`, `extract_error_spans` — finds failing spans, error messages, and trace-anchored file hints. |
| **Loki** | `loki_url` | `query_logs`, `get_log_context` — LogQL over container logs (`{compose_service="<svc>"} \|= "<query>"`). |
| **GitHub** | `github_token` + `repo` | Read-only source via the GitHub MCP server (read file, list, search). Never writes. |
| **Source** | `github_token` + `repo` | `get_file_slice` — reads just the window around a known line via the Contents API (cheaper than fetching a whole file once a trace has anchored the location). |
| **Sandbox** | `e2b_api_key` + `github_token` + `repo` + a file anchor | `reproduce_in_sandbox` — validates trigger conditions, then delegates to the sandbox sub-agent. |

**Sandbox sub-agent (`agent/subagents/sandbox.py`)** — A second `Agent` (`output_type = SandboxResult`) following Pydantic AI's agent-delegation pattern. Its toolset is E2B-only (`create_sandbox`, `clone_repo`, `run_command`, `read_output`, `terminate`), built at run time from `deps`. It has **no GitHub MCP** — it is a code runner, not an investigator: the parent already located the suspect file, so the sub-agent clones the repo into the sandbox and executes that known path directly. `clone_repo` injects the auth token server-side and redacts it from all output, so the token never enters the model's context.

**Verification stack (`agent/verification/`)** — `confidence.py` (band policy), `ledger.py` (the `LedgerToolset` and tool-family normalizers), and `verifier.py` (an optional downgrade-only verifier sub-agent that reads receipts, not prose).

**Fingerprint (`agent/fingerprint.py`)** — A pre-investigation pattern matcher against known error signatures (`OOMKilled`, `ECONNREFUSED`, `nil pointer`, `TLS expired`, runtime panic, …). A strong match (≥ 0.9) becomes a *hint* that narrows the first evidence round — it is never returned as a final answer without verification.

### 2.3 Investigation lifecycle

```
Issue description
       │
       ▼
Fingerprint pre-check  ── strong match ──► focused first-round hint (still verified)
       │
       ▼
Resolve session capabilities (AgentDeps via RunContext)
       │
       ▼
Round 1 — Jaeger traces + Loki logs (parallel)
       │
       ▼
govern_confidence  (the "governor": reads receipts → max supported band)
       │
       ├── band granted AND confidence ≥ 0.85 ──► synthesize HealResult
       │
       ▼  below threshold / overclaim
Round 2 — GitHub / Source read at the trace-anchored file path
       │   (skipped if no GitHub or no file anchor)
       ▼
govern_confidence (runs again on the new result)
       │
       ├── band granted AND confidence ≥ 0.85 ──► synthesize HealResult
       │
       ▼  still ambiguous, all trigger conditions met
Sandbox sub-agent — reproduce in E2B  (reproduced=True → authoritative HIGH)
       │
       ▼
HealResult
```

The agent stops when it is *confident enough*, not after a fixed number of steps. A clear hit finishes in two tool calls; an ambiguous case may take three rounds plus a sandbox repro. Soft target: 3–5 tool calls per investigation.

### 2.4 The confidence governor

This is the heart of the design: **confidence is earned by evidence, not asserted by the model.** After each run, the `govern_confidence` output validator:

1. **Floor check** — if the model's raw confidence is below `RETRY_CONFIDENCE`, raise `ModelRetry` ("collect more evidence").
2. **Ceiling check** — compute the maximum band the *collected receipts* support and compare to the model's claim:
   - First overclaim → `ModelRetry` naming exactly what evidence is missing (the agent gets another round and may run the sandbox, read source, etc.).
   - Still overclaiming on the second pass → clamp the confidence to the band ceiling and write the missing-evidence message into `confidence_note`.
3. Set `root_cause.confidence_level` from the final (possibly clamped) float.

**Band policy:**

| Band | Requirements |
|---|---|
| **HIGH** | `error_type != "unknown"` AND `file_path` set AND a GitHub tool used AND (sandbox reproduced OR both a Jaeger *and* a Loki tool used) |
| **MEDIUM** | `error_type != "unknown"` AND (trace+log corroboration OR source anchored) |
| **LOW** | everything else (default) |

The band drives a colored badge in the UI (HIGH green / MEDIUM amber / LOW grey); `confidence_note` tells the user what additional evidence would raise it. Full numeric thresholds (`BAND_*`, `STOP_CONFIDENCE`, `RETRY_CONFIDENCE`, `OUTPUT_RETRIES`) live in `config.py` and are env-tunable.

### 2.5 Key data models

All models live in `agent/models.py`.

- **`AgentDeps`** — the session: observability URLs/auth, GitHub token + repo, E2B key, target `service_name`, a shared `httpx` client, and the per-run `RunEvidence` ledger.
- **`IssueContext`** — one investigation: description, optional `service_name` override, `trace_id`, time window.
- **`RootCause`** — description, `file_path`/`line_number`, raw `confidence` float, governor-granted `confidence_level`, `evidence`, and an `error_type` enum (`null_pointer`, `runtime_error`, `race_condition`, `oom`, `tls`, `network`, …).
- **`HealResult`** — the top-level output: summary, investigation steps, root cause, recommended fix (+ optional snippet), `action_taken`, `tools_used`, `tools_unavailable`, `confidence_note`.
- **`SandboxResult`** — `reproduced`, confirmed file/line, stdout/stderr, exit code, the actual repro script, attempts, and a `skip_reason` when trigger conditions fail.

---

## 3. Repository layout

```
autoheal-agent/
├── agent/
│   ├── core.py          # main Agent + output validators (govern_confidence)
│   ├── loop.py          # investigation loop; resets RunEvidence, wraps LedgerToolset
│   ├── registry.py      # assembles active capabilities from AgentDeps
│   ├── fingerprint.py   # pre-tool error-pattern matcher
│   ├── prompts.py       # static + dynamic (capability-aware) system prompts
│   ├── models.py        # all Pydantic + dataclass models
│   ├── aimodel.py       # make_model() factory (OpenRouter)
│   ├── capabilities/    # jaeger, loki, github, source, sandbox
│   ├── verification/    # confidence policy, ledger, downgrade-only verifier
│   └── subagents/       # sandbox Agent + E2B tools
├── server.py            # FastAPI entrypoint (POST /chat, SSE)
├── config.py            # env loading + thresholds
├── docker-compose.override.yml  # Loki + Promtail sidecar for the OTel demo
├── promtail-config.yaml
├── tests/               # pytest suites
└── evals/               # verifier / sandbox / investigation eval suites
```

---

## 4. Prerequisites

- **Python 3.11+**
- **Docker** — for the OpenTelemetry demo and the Loki/Promtail sidecar
- **Node.js / npx** — for the GitHub MCP server (only when GitHub access is enabled)
- **OpenRouter API key** — [openrouter.ai/keys](https://openrouter.ai/keys)

Optional, for full capability coverage:

- GitHub personal access token (read-only)
- E2B API key (sandbox reproduction)

---

## 5. Quick start

### 5.1 Install and configure the agent

```bash
git clone git@github.com:Lucifer4255/autoheal-agent.git
cd autoheal-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env — at minimum set OPENROUTER_API_KEY
```

### 5.2 (Optional) Start the local observability stack

To run against the OpenTelemetry "Astronomy Shop" demo, clone it as a sibling of this repo, then bring it up together with the Loki/Promtail sidecar shipped here:

```bash
# from the directory that contains autoheal-agent/
git clone https://github.com/open-telemetry/opentelemetry-demo.git
cd opentelemetry-demo

docker compose \
  -f compose.yaml \
  -f compose.full.yaml \
  -f compose.observability.yaml \
  -f ../autoheal-agent/docker-compose.override.yml \
  up --build -d
```

> The Promtail volume mount in `docker-compose.override.yml` points at an absolute path to `promtail-config.yaml`. Adjust that path to your clone location before running.

Verify:

- Demo UI: [http://localhost:8080](http://localhost:8080)
- Jaeger: [http://localhost:8080/jaeger/ui/](http://localhost:8080/jaeger/ui/)
- Loki ready: `curl http://localhost:3100/ready`

The Loki/Promtail override ships Docker container logs with the `compose_service` label, which AutoHeal uses for LogQL queries.

### 5.3 Run the server

```bash
uvicorn server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) for the chat UI.

---

## 6. Using the chat UI

The interface is a single conversational stream — no separate setup phase. Paste credentials and describe the issue in the same message box.

**Configure and investigate in one message:**

```
repo: open-telemetry/opentelemetry-demo
service_name: ad
ad service is throwing runtime errors since 10am
```

**Provide credentials mid-investigation:**

```
github_token: ghp_xxx
repo: open-telemetry/opentelemetry-demo
```

The agent may *elicit* missing credentials during an investigation. Paste them in a follow-up message; the session picks up where it left off.

### Session config keys

| Key | Env default | Notes |
|---|---|---|
| `github_token` | `GITHUB_TOKEN` | Read-only PAT for GitHub MCP |
| `repo` | — | `owner/repo` or a GitHub URL |
| `service_name` | — | Target microservice (e.g. `ad`, `cart`) |
| `jaeger_url` | `JAEGER_URL` | Use `http://localhost:8080/jaeger/ui` for the OTel demo |
| `loki_url` | `LOKI_URL` | Default `http://localhost:3100` |
| `e2b_key` | `E2B_API_KEY` | Enables sandbox reproduction |

---

## 7. API

**`POST /chat`** — single endpoint for config updates and investigations. Returns Server-Sent Events.

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "service_name: ad\nad service is panicking"}'
```

SSE event types:

| Type | Meaning |
|---|---|
| `setup` | Capability configuration confirmed |
| `step` | Investigation progress (a tool call) |
| `elicit` | Agent requesting missing credentials |
| `final` | Completed `HealResult` |
| `error` | Unhandled failure |

Session ID is returned in the `X-Session-Id` response header and persisted in browser `localStorage`.

**`GET /health`** — liveness check.

---

## 8. Integration testing with the OTel demo

The demo exposes feature flags that inject known failures. Toggle a flag, then ask AutoHeal to investigate.

| Flag | Service | Expected error type |
|---|---|---|
| `adFailure` | `ad` | `runtime_error` |
| `cartFailure` | `cart` | `runtime_error` |
| `paymentFailure` | `payment` | `runtime_error` |
| `intlShippingSlowdown` | `shipping` | `network` |

List flags:

```bash
curl http://localhost:8080/feature/api/read
```

Example investigation after enabling `adFailure`:

```
repo: open-telemetry/opentelemetry-demo
service_name: ad
ad service is throwing runtime errors since the flag was enabled
```

---

## 9. Development

### Run tests

```bash
ruff check .
ruff format --check .
python -m pytest
```

### Run evals

Three eval suites live under `evals/`. See [evals/README.md](evals/README.md) for details.

```bash
# Verifier suite (no live services)
uv run python -m evals.run_verifier

# Sandbox suite (mock mode — no credentials)
uv run python -m evals.run_sandbox --mock

# End-to-end investigation (replay recorded cassettes)
EVAL_MODE=replay uv run python -m evals.run_investigation
```

---

## 10. Configuration reference

Key environment variables (see [.env.example](.env.example) for the full list):

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | LLM provider (required) |
| `MODEL_NAME` | `openrouter/google/gemini-2.0-flash-001` | Model slug |
| `JAEGER_URL` | `http://localhost:8080/jaeger/ui` | Trace backend |
| `LOKI_URL` | `http://localhost:3100` | Log backend |
| `GITHUB_TOKEN` | — | GitHub MCP auth |
| `E2B_API_KEY` | — | Sandbox reproduction |
| `LOGFIRE_TOKEN` | — | Optional observability ([logfire.pydantic.dev](https://logfire.pydantic.dev)) |

Confidence thresholds (`STOP_CONFIDENCE`, `RETRY_CONFIDENCE`, `BAND_*`, `OUTPUT_RETRIES`, …) are tunable via env — defaults live in `config.py`.

---

## 11. Tech stack

- **Agent framework:** Pydantic AI (`AbstractCapability`, MCP toolsets, agent delegation)
- **Web:** FastAPI + SSE streaming + vanilla HTML/JS chat UI
- **LLM:** OpenRouter (configurable model)
- **Observability:** Jaeger (traces), Loki + Promtail (logs)
- **Code access:** GitHub MCP (read-only) + a targeted Contents-API slice reader
- **Sandbox:** E2B (optional reproduction sub-agent)
- **Tests:** pytest, pytest-asyncio, ruff

---

## License

The OpenTelemetry demo referenced for integration testing is upstream at [open-telemetry/opentelemetry-demo](https://github.com/open-telemetry/opentelemetry-demo).
