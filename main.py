"""
main.py
=======
SentinalSRE Local -- Observability Layer & CLI Dashboard

100% Local Architecture -- Zero Cloud Dependency
-------------------------------------------------
  All inference: Ollama / Llama3 / NVIDIA GPU.
  GitHub actions: official @modelcontextprotocol/server-github via MCP stdio.

  Every node completion prints:
    - Agent name + LOCAL/TOOL/HUMAN badge
    - DT metric:  DT = T_now - T_start

  At HITL breakpoint:
    - GPU Processing Time Breakdown:
          Latency_total = T_pipeline_end - T_pipeline_start

Execution Flow
--------------
  Phase 1 : Stream all local nodes until HITL interrupt_before="deploy"
  Phase 2 : Terminal HITL prompt -- human approval
  Phase 3 : Resume graph -- open_pr (GitHub MCP) if approved

LangSmith tracing is initialised BEFORE any LangChain/LangGraph import
so that all local Ollama inference calls appear in the trace.

Usage
-----
    uv run python main.py
"""

import os
import asyncio
import time
import uuid
from datetime import datetime

from dotenv import load_dotenv

# =============================================================================
# LangSmith tracing -- MUST be set before any LangChain/LangGraph import
# =============================================================================
load_dotenv()

_lc_key = os.getenv("LANGCHAIN_API_KEY", "")
if _lc_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = _lc_key
    os.environ["LANGCHAIN_PROJECT"]     = "SentinalSRE-Local"
    os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
    _trace_status = "LangSmith tracing ACTIVE  ->  https://smith.langchain.com"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    _trace_status = "LangSmith DISABLED (set LANGCHAIN_API_KEY to enable)"

# =============================================================================
# Graph import (after env vars are set)
# =============================================================================
from agents.graph import sre_graph
from agents.state import SREState


# =============================================================================
# ANSI colour palette
# =============================================================================
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    PURPLE = "\033[95m"
    BLUE   = "\033[94m"
    ORANGE = "\033[38;5;208m"


# Per-node styling: (colour, tier_label)
NODE_META = {
    "log_monitor":    (C.CYAN,   "LOCAL  [Ollama/Llama3]"),
    "analyst":        (C.BLUE,   "LOCAL  [Ollama/Llama3]"),
    "manager":        (C.GREEN,  "LOCAL  [Ollama/Llama3]"),
    "engineer":       (C.YELLOW, "LOCAL  [Ollama/Llama3]"),
    "syntax_checker": (C.ORANGE, "LOCAL  [Ollama/Llama3]"),
    "reviewer":       (C.PURPLE, "LOCAL  [Ollama/Llama3]"),
    "deploy":         (C.RED,    "HUMAN  [HITL]"),
    "open_pr":        (C.GREEN,  "TOOL   [GitHub MCP]"),
}

LOCAL_NODES = {"log_monitor", "analyst", "manager", "engineer", "syntax_checker", "reviewer"}


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _banner():
    w = 66
    print(C.BOLD + C.CYAN)
    print("+" + "=" * (w - 2) + "+")
    print("|" + "  SentinalSRE  -  100% Local Multi-Agent SRE Engine  ".center(w - 2) + "|")
    print("|" + "  Ollama / Llama3 / NVIDIA GPU  +  LangGraph  ".center(w - 2) + "|")
    print("+" + "=" * (w - 2) + "+")
    print(C.RESET)


def _tier_badge(node: str) -> str:
    colour, tier = NODE_META.get(node, (C.DIM, "UNKNOWN"))
    if "LOCAL" in tier:
        return f"{C.CYAN}[ LOCAL - Ollama/Llama3 ]{C.RESET}"
    elif "HUMAN" in tier:
        return f"{C.RED}[ HUMAN - HITL ]{C.RESET}"
    elif "TOOL" in tier:
        return f"{C.GREEN}[ TOOL  - GitHub MCP ]{C.RESET}"
    return f"{C.DIM}[ {tier} ]{C.RESET}"


def _print_node_header(node: str, t_now: float, t_start: float):
    """Print node completion banner with DT metric."""
    colour, _ = NODE_META.get(node, (C.DIM, ""))
    delta = t_now - t_start

    print(f"\n{colour}{C.BOLD}{'-' * 66}{C.RESET}")
    print(
        f"{colour}{C.BOLD}  AGENT COMPLETED: {node.upper():<22}{C.RESET}"
        f"  {_tier_badge(node)}"
    )
    print(f"{colour}{'-' * 66}{C.RESET}")

    # LaTeX-style DT metric block (matches screenshots format)
    print(
        f"{colour}"
        f"  +-- Metric  DT = T_now - T_start\n"
        f"  |   Node    : {node.upper()}\n"
        f"  |   T_start : {datetime.fromtimestamp(t_start).strftime('%H:%M:%S.%f')[:-3]}\n"
        f"  |   T_now   : {datetime.fromtimestamp(t_now).strftime('%H:%M:%S.%f')[:-3]}\n"
        f"  +-- DT      = {delta:.3f} s"
        f"{C.RESET}"
    )


def _print_field(label: str, value: str, colour: str, max_chars: int = 350):
    snippet = (value or "")[:max_chars]
    if len(value or "") > max_chars:
        snippet += " ..."
    print(f"\n  {colour}{C.BOLD}{label}:{C.RESET}")
    for line in snippet.splitlines():
        print(f"    {C.DIM}{line}{C.RESET}")


def _gpu_latency_summary(t_pipeline_start: float, t_pipeline_end: float,
                         wall_elapsed: float, gpu_elapsed: float):
    """
    Print GPU processing time breakdown at HITL breakpoint.

        Latency_total = T_pipeline_end - T_pipeline_start
    """
    gpu_time  = (
        max(0.0, t_pipeline_end - t_pipeline_start)
        if (t_pipeline_start and t_pipeline_end)
        else gpu_elapsed
    )
    idle_time = max(0.0, wall_elapsed - gpu_time)
    pct_gpu   = (gpu_time  / wall_elapsed * 100) if wall_elapsed else 0.0
    pct_idle  = (idle_time / wall_elapsed * 100) if wall_elapsed else 0.0

    bar_width = 40
    gpu_fill  = int(bar_width * pct_gpu / 100)
    idle_fill = bar_width - gpu_fill
    bar = C.CYAN + "#" * gpu_fill + C.DIM + "-" * idle_fill + C.RESET

    print(f"\n{C.BOLD}  +-- GPU Processing Time Breakdown {'=' * 24}+{C.RESET}")
    print(f"  |   Formula : Latency_total = T_pipeline_end - T_pipeline_start")
    print(f"  |")
    print(f"  |   {C.CYAN}[LOCAL PROCESSING - NVIDIA GPU ACTIVE]{C.RESET}")
    print(f"  |")
    print(f"  |   T_gpu  (Ollama inference) = {gpu_time:7.3f} s  ({pct_gpu:5.1f}%)")
    print(f"  |   T_idle (I/O + overhead)   = {idle_time:7.3f} s  ({pct_idle:5.1f}%)")
    print(f"  |   {'-' * 44}")
    print(f"  |   Latency_total             = {wall_elapsed:7.3f} s")
    print(f"  |")
    print(f"  |   [{bar}]")
    print(f"  |    {C.CYAN}GPU inference{C.RESET}       {C.DIM}idle / I/O{C.RESET}")
    print(f"  +{'=' * 58}+")


def _hitl_prompt(state: dict, wall_elapsed: float, gpu_elapsed: float) -> bool:
    """
    Terminal HITL prompt. Shows GPU breakdown then waits for y/n.
    Returns True if approved.
    """
    t_ps = state.get("t_pipeline_start", 0.0)
    t_pe = state.get("t_pipeline_end",   0.0)
    _gpu_latency_summary(t_ps, t_pe, wall_elapsed, gpu_elapsed)

    print("\n" + "=" * 66)
    print(f"{C.RED}{C.BOLD}  HUMAN-IN-THE-LOOP  -  OPERATOR APPROVAL REQUIRED{C.RESET}")
    print("=" * 66)
    print(f"\n  Incident    : {state.get('incident_id', 'N/A')}")
    print(f"  Iterations  : {state.get('iteration_count', 0)}")
    print(f"  Verified    : {state.get('is_verified', False)}")

    print(f"\n{C.YELLOW}  -- Manager Directive --{C.RESET}")
    for ln in (state.get("manager_directive") or "")[:400].splitlines():
        print(f"    {C.DIM}{ln}{C.RESET}")

    print(f"\n{C.YELLOW}  -- Proposed Patch (preview) --{C.RESET}")
    for ln in (state.get("proposed_patch") or "")[:600].splitlines():
        print(f"    {C.DIM}{ln}{C.RESET}")

    # Show GitHub config status
    owner = os.getenv("GITHUB_OWNER", "").strip()
    repo  = os.getenv("GITHUB_REPO",  "").strip()
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
    print(f"\n{C.YELLOW}  -- GitHub MCP Config --{C.RESET}")
    print(f"    Token  : {'SET (' + token[:8] + '...)' if token else C.RED + 'NOT SET -- will dry-run' + C.RESET}")
    print(f"    Owner  : {owner or C.RED + 'NOT SET -- add GITHUB_OWNER to .env' + C.RESET}")
    print(f"    Repo   : {repo  or C.RED + 'NOT SET -- add GITHUB_REPO to .env'  + C.RESET}")

    print()
    print(f"{C.BOLD}  Approve opening a GitHub Pull Request?{C.RESET}")
    print(f"  {C.GREEN}y{C.RESET} = Approve & submit PR   |   {C.RED}n{C.RESET} = Abort & escalate")
    print(f"  -> ", end="", flush=True)

    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted.")
        return False

    return answer in ("y", "yes")


# =============================================================================
# PER-NODE OUTPUT RENDERING
# =============================================================================

def _render_node_output(node: str, output: dict):
    """Print the key state fields produced by each node."""
    colour, _ = NODE_META.get(node, (C.DIM, ""))

    if node == "log_monitor":
        _print_field("Incident ID",       output.get("incident_id", ""),       colour, 60)
        _print_field("Sanitized Summary", output.get("sanitized_summary", ""), colour, 300)

    elif node == "analyst":
        _print_field("RCA Hypothesis",    output.get("rca_hypothesis", ""),    colour, 400)
        print(f"  {colour}  Iteration : #{output.get('iteration_count', '?')}{C.RESET}")

    elif node == "manager":
        _print_field("Manager Directive", output.get("manager_directive", ""), colour, 400)

    elif node == "engineer":
        _print_field("Proposed Patch",    output.get("proposed_patch", ""),    colour, 400)

    elif node == "syntax_checker":
        verdict = "PASS" if output.get("is_verified") else "FAIL"
        print(f"  {colour}  Syntax Verdict : {verdict}{C.RESET}")

    elif node == "reviewer":
        verdict = "APPROVED" if output.get("is_verified") else "REJECTED"
        print(f"  {colour}  Review Verdict : {verdict}{C.RESET}")

    elif node == "deploy":
        pass  # deploy prints its own output inside deploy_node

    elif node == "open_pr":
        pr_result = output.get("pr_result", "")
        msgs      = output.get("messages", [])
        content   = pr_result or (getattr(msgs[-1], "content", "") if msgs else "")
        _print_field("PR Result", content, colour, 400)


# =============================================================================
# MAIN ASYNC RUNNER
# =============================================================================

async def run_sentinal():
    _banner()

    print(f"  {_trace_status}")
    print(f"  {C.DIM}Run ID    : {uuid.uuid4()}{C.RESET}")
    print(f"  {C.DIM}Started   : {datetime.now().isoformat()}{C.RESET}")
    print(f"  {C.CYAN}[LOCAL PROCESSING - NVIDIA GPU ACTIVE]{C.RESET}")
    print()

    print(f"  {C.CYAN}LOCAL nodes{C.RESET}  ->  log_monitor, analyst, manager,")
    print(f"               engineer, syntax_checker, reviewer  (Ollama/Llama3)")
    print(f"  {C.RED}HITL{C.RESET}         ->  deploy                           (You)")
    print(f"  {C.GREEN}TOOL{C.RESET}         ->  open_pr                          (GitHub MCP)")
    print()

    # ── Initial state ─────────────────────────────────────────────────────────
    initial_state: SREState = {
        "messages":          [],
        "incident_id":       "",
        "raw_logs":          "",
        "log_data":          "",
        "sanitized_summary": "",
        "rca_hypothesis":    "",
        "manager_directive": "",
        "proposed_patch":    "",
        "pr_result":         "",
        "iteration_count":   0,
        "is_verified":       False,
        "user_approval":     False,
        "t_pipeline_start":  0.0,
        "t_pipeline_end":    0.0,
    }

    config = {
        "configurable": {
            "thread_id": f"local-{int(time.time())}",
        },
        "metadata": {
            "project":   "SentinalSRE-Local",
            "inference": "Ollama/Llama3",
        },
    }

    t_start     = time.time()
    gpu_elapsed = 0.0
    prev_t      = t_start

    print(f"{C.CYAN}{C.BOLD}  >> Autonomous local SRE pipeline starting ...{C.RESET}\n")

    # ── Phase 1: Stream nodes until HITL interrupt ────────────────────────────
    current_state = None

    async for chunk in sre_graph.astream(initial_state, config=config):
        for node_name, node_output in chunk.items():
            if node_name == "__interrupt__":
                print(f"\n{C.RED}  Pipeline paused -- awaiting HITL input.{C.RESET}")
                break

            t_now   = time.time()
            node_dt = t_now - prev_t
            prev_t  = t_now

            if node_name in LOCAL_NODES:
                gpu_elapsed += node_dt

            _print_node_header(node_name, t_now, t_start)
            _render_node_output(node_name, node_output)
            print()

            current_state = sre_graph.get_state(config)

    # ── Phase 2: HITL prompt ──────────────────────────────────────────────────
    if current_state is None:
        current_state = sre_graph.get_state(config)

    state_vals = current_state.values if current_state else {}
    next_nodes  = current_state.next   if current_state else []

    if "deploy" in next_nodes or not next_nodes:
        wall_so_far = time.time() - t_start
        approved = _hitl_prompt(state_vals, wall_so_far, gpu_elapsed)

        sre_graph.update_state(
            config,
            {"user_approval": approved},
            as_node="deploy",
        )

        if approved:
            print(f"\n{C.GREEN}  Resuming pipeline -- opening GitHub PR ...{C.RESET}\n")
            async for chunk in sre_graph.astream(None, config=config):
                for node_name, node_output in chunk.items():
                    if node_name == "__interrupt__":
                        break
                    t_now = time.time()
                    _print_node_header(node_name, t_now, t_start)
                    _render_node_output(node_name, node_output)
                    print()
        else:
            print(f"\n{C.RED}  PR aborted by operator. Incident escalated to on-call team.{C.RESET}")

    # ── Final summary ─────────────────────────────────────────────────────────
    t_total_wall = time.time() - t_start
    final_state  = (
        sre_graph.get_state(config).values
        if sre_graph.get_state(config)
        else state_vals
    )

    t_ps = final_state.get("t_pipeline_start", 0.0)
    t_pe = final_state.get("t_pipeline_end",   0.0)
    gpu_total = (t_pe - t_ps) if (t_ps and t_pe) else gpu_elapsed

    pr_result = final_state.get("pr_result", "")

    print("\n" + "=" * 66)
    print(f"{C.BOLD}{C.GREEN}  SentinalSRE Local  -  Run Complete{C.RESET}")
    print(f"  Incident          : {final_state.get('incident_id', 'N/A')}")
    print(f"  Final Verified    : {final_state.get('is_verified', False)}")
    print(f"  User Approved     : {final_state.get('user_approval', False)}")
    print(f"  Iterations        : {final_state.get('iteration_count', 0)}")
    if pr_result:
        print(f"  PR Result         : {pr_result[:120]}")
    print()
    print(f"  {C.CYAN}[LOCAL PROCESSING - NVIDIA GPU ACTIVE]{C.RESET}")
    print(f"  Model             : Ollama / {os.getenv('OLLAMA_MODEL', 'llama3')}")
    print(f"  GPU inference     : {gpu_total:.2f} s")
    print(f"  Latency_total     : {t_total_wall:.2f} s  (wall-clock)")
    if _lc_key:
        print(f"  Trace             : https://smith.langchain.com")
    print("=" * 66 + "\n")


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    asyncio.run(run_sentinal())