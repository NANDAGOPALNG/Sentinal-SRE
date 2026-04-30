"""
agents/state.py
===============
SentinalSRE Local -- Shared Memory Layer (100% Local Architecture)

All inference is handled by Ollama/Llama3 running on the local NVIDIA GPU.
No cloud fields, no data gates -- the full log pipeline stays on-device.

Pipeline flow tracked by this state:
  raw_logs          -> ingested by log_monitor via FastMCP stdio tool
  sanitized_summary -> PII-redacted summary (local LLM, <=200 tokens)
  rca_hypothesis    -> structured RCA from analyst (local LLM)
  manager_directive -> severity classification + fix strategy (local LLM)
  proposed_patch    -> full PR-formatted code fix + unified diff (local LLM)
  is_verified       -> True when reviewer approves patch (local LLM)
  user_approval     -> True after human operator confirms via HITL terminal
  pr_result         -> URL or result string returned by GitHub MCP open_pr_node

Latency telemetry:
  t_pipeline_start  -> set by log_monitor at pipeline entry
  t_pipeline_end    -> set by reviewer on approval
  Used in main.py to display:
      Latency_total = T_pipeline_end - T_pipeline_start
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SREState(TypedDict):
    """
    Central State Schema -- SentinalSRE 100% Local Multi-Agent Workflow.

    Message history
    ---------------
    messages        : Append-only LangChain message list (add_messages).

    Incident identity
    -----------------
    incident_id     : Auto-generated ID, e.g. "INC-20260426-143201".

    Log pipeline (all local)
    ------------------------
    raw_logs          : Full unredacted log content from MCP tool.
    log_data          : Backward-compat alias -- set equal to raw_logs.
    sanitized_summary : PII-free compressed summary (local LLM, <=200 tokens).

    Agent outputs (all local LLM)
    -----------------------------
    rca_hypothesis    : Structured Root Cause Analysis from analyst node.
    manager_directive : Severity classification + fix strategy from manager.
    proposed_patch    : Full GitHub PR body + unified diff from engineer.

    GitHub action output
    --------------------
    pr_result         : Result string from open_pr_node (URL or dry-run msg).

    Control flow
    ------------
    iteration_count : How many analyst->engineer->reviewer cycles completed.
    is_verified     : True when reviewer approves patch; False triggers loop.
    user_approval   : Set by human at HITL terminal prompt.

    GPU latency telemetry
    ---------------------
    t_pipeline_start : Unix timestamp at pipeline start (set by log_monitor).
    t_pipeline_end   : Unix timestamp after reviewer approves (set by reviewer).
    """

    # Message history
    messages: Annotated[List, add_messages]

    # Incident identity
    incident_id: str

    # Log pipeline
    raw_logs: str
    log_data: str              # backward-compat alias
    sanitized_summary: str

    # Agent outputs
    rca_hypothesis: str
    manager_directive: str
    proposed_patch: str

    # GitHub action output
    pr_result: str

    # Control flow
    iteration_count: int
    is_verified: bool
    user_approval: bool

    # Latency telemetry
    t_pipeline_start: float
    t_pipeline_end: float