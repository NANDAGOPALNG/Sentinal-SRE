# SentinalSRE

**Autonomous Local-First SRE Engine — Zero Cloud, Zero Rate Limits, Zero Manual Triage**

A production-grade, multi-agent DevOps orchestration system that detects incidents from live logs, performs Root Cause Analysis, self-corrects its own patches through a cyclic verification loop, and opens a GitHub Pull Request all running entirely on a local NVIDIA GPU via Ollama.

[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-blue)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Inference-Ollama%20%2F%20Llama3-green)](https://ollama.com)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange)](https://smith.langchain.com)
[![MCP](https://img.shields.io/badge/Tools-FastMCP%20%2B%20GitHub%20MCP-purple)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What Problem Does This Solve?

When a production service crashes at 2 AM, the on-call engineer faces a well-known, repeatable process: read the logs, find the root cause, write a fix, open a PR, get it reviewed, deploy. This process — entirely manual — is the primary driver of high Mean Time to Recovery (MTTR).

SentinalSRE automates every step of that loop. It runs entirely locally on commodity GPU hardware, meaning your production logs never leave your machine, you never hit a cloud API rate limit, and you never pay per token.

---

## Key Features

**100% Local Inference**
Every agent — log ingestion, root cause analysis, patch generation, syntax verification, semantic review — runs on Ollama with Llama 3 via the local NVIDIA GPU. No Gemini, no OpenAI, no cloud dependency, no rate limits.

**PII Redaction Before Any LLM Call**
Raw production logs are scrubbed of IP addresses, email addresses, UUIDs, session tokens, and precise timestamps using deterministic regex patterns before any text is passed to the language model. Your sensitive operational data stays clean.

**Cyclic Self-Correction Loop**
The system does not blindly accept the first patch it generates. A dedicated Syntax Checker agent runs a hard-fail pattern scan followed by an LLM structural review. If the patch fails, the pipeline automatically loops back to the Analyst for re-diagnosis. This is capped at 3 iterations to prevent infinite loops.

**Human-in-the-Loop (HITL) Safety Gate**
No code ever reaches GitHub without explicit operator approval. The LangGraph pipeline pauses at a `interrupt_before` breakpoint, presents the full patch preview and manager directive to the terminal, and waits for a `y` / `n` confirmation before proceeding.

**GitHub MCP Integration**
PR creation is handled through the official `@modelcontextprotocol/server-github` MCP server via stdio, not a raw REST call. The system executes a 4-step flow: verify repo access, create the head branch, commit the patched file, then open the pull request — all within a single MCP session.

**LangSmith Trace Visibility**
Every agent invocation is captured in a LangSmith trace, providing full thought-stream transparency for debugging, auditing, and demo purposes. Environment variables are set before any LangChain import to ensure all local Ollama calls appear in the trace.

**Structured Latency Metrics**
The terminal dashboard prints a LaTeX-style `DT = T_now - T_start` metric after every node and a full GPU Processing Time Breakdown at the HITL gate:

```
Latency_total = T_pipeline_end - T_pipeline_start
```

---

## Architecture

SentinalSRE is built as a stateful, cyclic LangGraph `StateGraph`. All nodes share a single `SREState` TypedDict that flows through the entire pipeline.

```
START
  |
  v
LOG_MONITOR  -->  ANALYST  -->  MANAGER  -->  ENGINEER
                     ^                            |
                     |                            v
                     |                     SYNTAX_CHECKER
                     |                       |         |
                     |                     FAIL       PASS
                     |                       |         |
                     +<----------------------+      REVIEWER
                     |                              |       |
                     |                          REJECTED  APPROVED
                     +<-----------------------------+         |
                                                        [interrupt]
                                                             |
                                                          DEPLOY  <-- HITL breakpoint
                                                             |
                                               +-------------+-------------+
                                            approved                   rejected
                                               |                           |
                                           OPEN_PR                        END
                                           (GitHub MCP)
                                               |
                                              END
```

### Node Responsibilities

| Node | Tier | Responsibility |
|---|---|---|
| `log_monitor` | LOCAL | Reads logs via FastMCP stdio, applies PII redaction, produces a ≤150-word sanitized summary |
| `analyst` | LOCAL | Correlates sanitized summary with `./app/main.py` source to produce a structured RCA |
| `manager` | LOCAL | Validates the RCA, classifies severity (CRITICAL/HIGH/MEDIUM/LOW), issues engineer directive |
| `engineer` | LOCAL | Generates a full GitHub PR body including `PR_TITLE`, `PR_BRANCH`, `PR_DESCRIPTION`, and the complete patched Python file |
| `syntax_checker` | LOCAL | Hard-fail regex scan + LLM structural review; routes FAIL back to analyst, PASS forward to reviewer |
| `reviewer` | LOCAL | Semantic review: correctness, safety, coherence, hallucination detection |
| `deploy` | HUMAN | LangGraph `interrupt_before` breakpoint; pauses for operator approval |
| `open_pr` | TOOL | 4-step GitHub MCP flow: verify → branch → commit → PR |

### MCP Tool Binding

SentinalSRE uses two MCP servers connected via stdio, not HTTP:

- **Custom FastMCP server** (`mcp_server/server.py`): Provides `read_incident_logs` and `check_log_health` tools. Spawned as a Python subprocess, session initialized, tool called, subprocess exits.
- **Official GitHub MCP server** (`@modelcontextprotocol/server-github`): Provides the full GitHub API surface. Spawned via `npx`, token injected through subprocess environment, not the command line.

---

## The Self-Correction Loop in Action

The screenshots below capture a real pipeline run on an Asus TUF Gaming F15 (RTX 4060 GPU).

**Iteration 1 — Syntax Checker Hard Fail:**
The Engineer generated a patch that blanked the `trigger_crash` function with only a `pass` statement. The Syntax Checker detected this immediately with the regex pattern `def\s+trigger_crash.*?pass` and returned `HARD FAIL -- patch blanks trigger_crash with pass only`. No LLM call was made for the review — the failure was caught in microseconds and the pipeline looped back to the Analyst.

**Iteration 2 — Clean Patch, Full Approval:**
The Analyst re-diagnosed with the retry context injected into its prompt (the previous patch and the manager directive from iteration 1). The Engineer produced a complete, valid Python file. The Syntax Checker returned `PASS: The code block contains syntactically valid Python, imports only modules already present in the original source`. The Reviewer returned `APPROVED: The patch correctly modifies the trigger_crash function to prevent raising an HTTPException and logging a critical error`. The pipeline proceeded to HITL.

---

## Performance Metrics

The following was recorded on a live run (Incident `INC-20260501-105148`):

| Phase | Node | Time |
|---|---|---|
| Log ingestion + summarisation | `log_monitor` | 16.81 s |
| Root cause analysis (Iter 1) | `analyst` | 26.46 s |
| Incident triage | `manager` | 12.65 s |
| Patch generation (Iter 1) | `engineer` | 87.61 s |
| Syntax check (Iter 1 — FAIL) | `syntax_checker` | ~0.1 s |
| Root cause analysis (Iter 2) | `analyst` | 25.62 s |
| Incident triage (Iter 2) | `manager` | 11.65 s |
| Patch generation (Iter 2) | `engineer` | 90.56 s |
| Syntax check (Iter 2 — PASS) | `syntax_checker` | 8.06 s |
| Semantic review | `reviewer` | 9.38 s |
| **Total GPU inference** | | **288.79 s** |
| **Wall-clock total** | | **322.73 s** |

GPU utilisation was 99.8% across the run — idle time was only 0.459 s (I/O and subprocess startup).

### Agentic MTTR Model

$$MTTR_{Agentic} = T_{Ingestion} + T_{RCA} + T_{Patching} + T_{Verification}$$

$$Efficiency = \frac{T_{Manual\_MTTR}}{T_{Agentic\_MTTR}}$$

For a typical P1 incident where manual MTTR is 45-90 minutes, SentinalSRE achieves the same outcome in under 6 minutes on consumer hardware — without an on-call engineer being paged.

---

## Project Structure

```
SentinalSRE/
  agents/
    __init__.py       -- exports SREState and sre_graph
    state.py          -- SREState TypedDict (shared memory)
    nodes.py          -- all 8 agent node implementations
    graph.py          -- LangGraph StateGraph + routing logic
  app/
    main.py           -- FastAPI sandbox app (the "production" service)
    Dockerfile        -- container definition for the sandbox
  mcp_server/
    server.py         -- FastMCP log-reader server (stdio)
  logs/
    app.log           -- shared volume between Docker and host
  main.py             -- entry point: LangSmith init, streaming, HITL, dashboard
  docker-compose.yaml -- spins up the sandbox FastAPI app
  pyproject.toml      -- uv-managed dependencies
  .env.example        -- environment variable template
```

---

## Prerequisites

Install the following before running:

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Runtime |
| uv | latest | Package and venv management |
| Node.js | LTS | Required by GitHub MCP server |
| Docker Desktop | latest | Runs the sandbox FastAPI app |
| Ollama | latest | Local LLM inference server |

---

## Setup

### 1. Install the GitHub MCP Server

```powershell
npm install -g @modelcontextprotocol/server-github
```

### 2. Pull the Llama 3 Model

```powershell
ollama pull llama3
```

This downloads approximately 4.7 GB. Only required once.

### 3. Clone the Repository

```powershell
git clone https://github.com/NANDAGOPALNG/Sentinal-SRE.git
cd Sentinal-SRE
```

### 4. Configure Environment Variables

```powershell
copy .env.example .env
notepad .env
```

Fill in the following:

```env
# GitHub — required for real PR creation
GITHUB_PERSONAL_ACCESS_TOKEN="github_pat_..."
GITHUB_OWNER="your-github-username"
GITHUB_REPO="your-repository-name"

# Ollama — local inference
OLLAMA_MODEL="llama3"
OLLAMA_BASE_URL="http://localhost:11434"

# LangSmith — optional, enables trace visibility
LANGCHAIN_API_KEY="lsv2_pt_..."
```

### 5. Install Python Dependencies

```powershell
uv venv
uv sync
```

If no lockfile exists:

```powershell
uv add langchain-core langchain-ollama langgraph langsmith mcp python-dotenv fastapi uvicorn
```

---

## Running the System

Three terminal windows are required simultaneously.

**Window 1 — Ollama inference server:**
```powershell
ollama serve
```

**Window 2 — Docker sandbox (the "production" app):**
```powershell
docker compose up --build
```

Once running, trigger an incident by visiting:
```
http://localhost:8000/crash
```

This writes the `CRITICAL: Database connection pool exhausted` error to `./logs/app.log`.

**Window 3 — SentinalSRE:**
```powershell
uv run python main.py
```

The pipeline will stream all node completions to the terminal. When the Reviewer approves the patch, the pipeline pauses and prompts:

```
Approve opening a GitHub Pull Request?
y = Approve & submit PR   |   n = Abort & escalate
->
```

Type `y` to open the PR via GitHub MCP, or `n` to escalate.

---

## GitHub Token Permissions

This is the most common configuration error. The GitHub Personal Access Token **must** have the `repo` scope (full repository access). Fine-grained tokens with restricted permissions will cause the following error at Step 2/4 of the MCP flow:

```
[GitHub MCP] Branch creation note: Permission Denied:
Resource not accessible by personal access token
```

and subsequently:

```
[GitHub MCP ERROR] Failed to commit file:
Permission Denied: Resource not accessible by personal access token
```

To fix this: go to **github.com/settings/tokens**, generate a new classic token, and check the top-level `repo` scope checkbox. Replace the token in `.env` and re-run.

Additionally, `GITHUB_OWNER` must match your exact GitHub username or organisation name (case-sensitive), and `GITHUB_REPO` must match the exact repository name. The repository must contain `app/main.py` at the root level — this is the file the Engineer patches and the MCP server commits.

---

## Observability

If `LANGCHAIN_API_KEY` is set, all agent invocations are captured at `https://smith.langchain.com` under the project `SentinalSRE-Local`. Each run appears as a named trace with full input/output visibility for every node, making it straightforward to inspect the exact prompts and responses that drove each decision.

LangSmith environment variables are set at the very top of `main.py`, before any LangChain or LangGraph import, to ensure local Ollama calls are captured correctly.

---

## Troubleshooting

**`npx not found` when opening PR**
Node.js is not installed, or the npm global bin directory is not on your PATH. Install Node.js from nodejs.org, restart your terminal, and verify with `npx --version`.

**`Log file not found`**
Docker is not running, or you have not triggered the incident yet. Start Docker Compose in Window 2, then visit `http://localhost:8000/crash`.

**Pipeline loops 3 times then escalates**
Llama 3 8B is generating patches that fail the Syntax Checker or Reviewer. This is normal behaviour on first runs with small models. The system escalates to HITL after the iteration cap and you can still approve manually. To reduce iteration count, use a larger model: set `OLLAMA_MODEL=llama3:70b` in `.env` (requires ~40 GB VRAM).

**`ModuleNotFoundError: langchain_ollama`**
You are running `python main.py` directly instead of through the uv virtual environment. Use `uv run python main.py`.

**`Permission Denied` from GitHub MCP**
See the GitHub Token Permissions section above. The token must have the `repo` scope, not a fine-grained permission subset.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph 0.2+ (StateGraph, MemorySaver, interrupt_before) |
| Local Inference | Ollama / Llama 3 via langchain-ollama |
| Log Ingestion Tool | FastMCP (custom stdio server) |
| GitHub Integration | @modelcontextprotocol/server-github (official MCP, stdio) |
| Observability | LangSmith |
| Sandbox App | FastAPI + Uvicorn in Docker |
| Package Management | uv (Astral) |
| Environment | Windows 11, Asus TUF Gaming F15, NVIDIA RTX 4060 |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact & Collaboration

**Nanda Gopal D**

AI Engineer | Local-First AI & DevOps Automation

| | |
|---|---|
| Email | [nandagopalng2004@gmail.com](mailto:nandagopalng2004@gmail.com) |
| LinkedIn | [linkedin.com/in/nanda-gopal-d-1b623229b](https://www.linkedin.com/in/nanda-gopal-d-1b623229b/) |
| GitHub | [github.com/NANDAGOPALNG](https://github.com/NANDAGOPALNG) |

Feel free to reach out if you are working on anything in the space of local-first AI, autonomous DevOps tooling, or agentic systems — always open to a conversation.

### Contributing

Contributions are welcome. If you have an idea for a new agent node, a better prompt strategy, or a fix for an edge case you hit during setup:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit them with a clear message
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request against `main` — describe what you changed and why

Please keep PRs focused on a single change. If you are planning something large, open an issue first so we can discuss the approach before you invest the time.

*Built as a demonstration of production-grade, privacy-first autonomous SRE tooling on consumer hardware.*
