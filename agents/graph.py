"""
agents/graph.py
===============
SentinalSRE Local -- Central Nervous System (LangGraph StateGraph)

100% Local Workflow -- All nodes run on Ollama/Llama3
------------------------------------------------------

  START
    |
    v
  log_monitor  ->  analyst  ->  manager  ->  engineer
                      ^                          |
                      |                          v
                      |                   syntax_checker
                      |                     |       |
                      |                  FAIL      PASS
                      |                   |         |
                      +<------------------+      reviewer
                      |                          |       |
                      |                      REJECTED  APPROVED
                      |                         |         |
                      +<------------------------+    [interrupt]
                                                         |
                                                       deploy  <- HITL breakpoint
                                                         |
                                              +----------+-----------+
                                           approved             rejected
                                              |                     |
                                           open_pr                 END
                                              |
                                             END

Node Tier Summary
-----------------
  ALL nodes: LOCAL (Ollama/Llama3 on NVIDIA GPU)
  deploy   : HUMAN (HITL -- no model)
  open_pr  : TOOL  (GitHub MCP -- no LLM)

Key Design Decisions
--------------------
  - syntax_checker runs BEFORE reviewer to catch cheap failures without
    spending an extra Ollama call on a bad patch.
  - Cyclic loop cap: MAX_ITERATIONS=3 prevents infinite loops.
  - MemorySaver checkpoint persists full state across the HITL pause,
    so the graph can be resumed after human input without re-running nodes.
  - interrupt_before=["deploy"] is the LangGraph v0.2+ HITL mechanism.
    main.py calls sre_graph.update_state() then sre_graph.astream(None)
    to resume from the checkpoint.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import SREState
from .nodes import (
    log_monitor_node,
    analyst_node,
    manager_node,
    engineer_node,
    syntax_checker_node,
    reviewer_node,
    deploy_node,
    open_pr_node,
)

MAX_ITERATIONS = 3


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_syntax_check(state: SREState) -> str:
    """
    After syntax_checker_node:
      PASS (is_verified=True)      -> reviewer (semantic check)
      FAIL + iter < MAX            -> analyst  (retry loop)
      FAIL + iter >= MAX           -> deploy   (escalate to HITL)
    """
    iteration   = state.get("iteration_count", 0)
    is_verified = state.get("is_verified", False)

    if is_verified:
        return "reviewer"

    if iteration >= MAX_ITERATIONS:
        print(f"\n  Max iterations ({MAX_ITERATIONS}) hit at syntax check -- escalating to HITL.")
        return "deploy"

    print(f"\n  Syntax FAIL -- retrying analyst (iteration {iteration}/{MAX_ITERATIONS})")
    return "analyst"


def route_after_reviewer(state: SREState) -> str:
    """
    After reviewer_node:
      APPROVED (is_verified=True)  -> deploy  (HITL breakpoint)
      REJECTED + iter < MAX        -> analyst (retry loop)
      REJECTED + iter >= MAX       -> deploy  (escalate to HITL)
    """
    iteration   = state.get("iteration_count", 0)
    is_verified = state.get("is_verified", False)

    if is_verified:
        return "deploy"

    if iteration >= MAX_ITERATIONS:
        print(f"\n  Max iterations ({MAX_ITERATIONS}) reached -- escalating to HITL.")
        return "deploy"

    print(f"\n  Reviewer REJECTED -- retrying analyst (iteration {iteration}/{MAX_ITERATIONS})")
    return "analyst"


def route_after_deploy(state: SREState) -> str:
    """
    After human operator provides input via main.py:
      user_approval=True  -> open_pr (GitHub MCP)
      user_approval=False -> END (incident escalated)
    """
    if state.get("user_approval", False):
        return "open_pr"
    return END


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_graph():
    """
    Construct and compile the SentinalSRE Local StateGraph.

    All nodes are registered, edges wired, conditional routing attached,
    then compiled with MemorySaver checkpoint and interrupt_before=["deploy"].
    """
    builder = StateGraph(SREState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("log_monitor",    log_monitor_node)
    builder.add_node("analyst",        analyst_node)
    builder.add_node("manager",        manager_node)
    builder.add_node("engineer",       engineer_node)
    builder.add_node("syntax_checker", syntax_checker_node)
    builder.add_node("reviewer",       reviewer_node)
    builder.add_node("deploy",         deploy_node)
    builder.add_node("open_pr",        open_pr_node)

    # ── Linear edges (forward pipeline) ──────────────────────────────────────
    builder.add_edge(START,          "log_monitor")
    builder.add_edge("log_monitor",  "analyst")
    builder.add_edge("analyst",      "manager")
    builder.add_edge("manager",      "engineer")
    builder.add_edge("engineer",     "syntax_checker")

    # ── Conditional: after syntax_checker ────────────────────────────────────
    builder.add_conditional_edges(
        "syntax_checker",
        route_after_syntax_check,
        {
            "reviewer": "reviewer",
            "analyst":  "analyst",
            "deploy":   "deploy",
        },
    )

    # ── Conditional: after reviewer ───────────────────────────────────────────
    builder.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "deploy":  "deploy",
            "analyst": "analyst",
        },
    )

    # ── Conditional: after HITL deploy ────────────────────────────────────────
    builder.add_conditional_edges(
        "deploy",
        route_after_deploy,
        {
            "open_pr": "open_pr",
            END:       END,
        },
    )

    builder.add_edge("open_pr", END)

    # ── Compile with HITL interrupt + memory checkpoint ───────────────────────
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["deploy"],
    )

    return graph


# Module-level compiled graph -- imported by main.py
sre_graph = build_graph()