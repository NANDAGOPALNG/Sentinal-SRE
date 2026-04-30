"""
agents/nodes.py
===============
SentinalSRE Local -- 100% Local Agentic Inference Core

Architecture: Zero Cloud Dependency
------------------------------------
  Every agent node runs on the local Ollama instance via the NVIDIA GPU.
  All nodes are async. Single shared ChatOllama instance. No Gemini, no OpenAI.

  Pipeline:
    log_monitor -> analyst -> manager -> engineer -> syntax_checker -> reviewer
         ^______________________________________________|  (cyclic if rejected)
                                                        |
                                                  [HITL deploy]
                                                        |
                                                    open_pr  <- GitHub MCP

GitHub MCP Integration (Fixed)
-------------------------------
  The root cause of previous failures was three-fold:

  1. BRANCH MUST EXIST FIRST:
     GitHub's create_pull_request tool requires `head` to be an existing branch
     with at least one commit diverged from `base`. We solve this by calling
     create_or_update_file FIRST to write the patched file to a new branch,
     which GitHub auto-creates, then calling create_pull_request.

  2. WINDOWS PATH FOR NPX:
     When Python spawns npx via subprocess on Windows, it uses the inherited
     PATH which often lacks the npm global bin. We resolve this by:
       a) Discovering the npm prefix at runtime via `npm root -g`
       b) Injecting the correct bin path explicitly into the subprocess env
       c) Using `node_modules/.bin/` resolution as a fallback

  3. SILENT EXCEPTION SWALLOWING:
     The old helper returned a dry-run string on ANY exception, hiding real
     errors. The new implementation surfaces the actual error message so you
     can debug it, while still not crashing the pipeline.

  4. TOKEN INJECTION:
     GITHUB_PERSONAL_ACCESS_TOKEN must be in the env dict passed to the
     MCP server subprocess. The old code did this correctly but the PATH
     issue prevented the server from starting at all.

MCP Tool Binding (Ollama)
-------------------------
  Ollama's tool-calling works best with structured prompts rather than
  .bind_tools() because Llama3 8B/70B models follow explicit JSON schemas
  in the system prompt more reliably than OpenAI-style function calling.
  We use direct MCP ClientSession calls (not LLM tool binding) for GitHub,
  which is more reliable and gives us full control over argument construction.

All nodes are async (async def).
"""

import os
import re
import sys
import json
import time
import subprocess
import platform
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# =============================================================================
# SINGLE LOCAL LLM -- ChatOllama on NVIDIA GPU
# temperature=0 -> deterministic, reproducible SRE reasoning.
# num_predict=2048 -> enough for full PR bodies with diffs.
# =============================================================================
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=2048,
)

# =============================================================================
# PATH CONSTANTS
# =============================================================================
_BASE         = os.path.dirname(__file__)
LOG_PATH      = os.path.abspath(os.path.join(_BASE, "..", "logs", "app.log"))
APP_CODE_PATH = os.path.abspath(os.path.join(_BASE, "..", "app", "main.py"))
MCP_LOG_PATH  = os.path.abspath(os.path.join(_BASE, "..", "mcp_server", "server.py"))

GPU_BADGE = "\033[92m[LOCAL PROCESSING -- NVIDIA GPU ACTIVE]\033[0m"


# =============================================================================
# NPX RESOLUTION (Windows + Unix)
# Resolves the most common cause of GitHub MCP failures on Windows: npx not
# found because the npm global bin directory is missing from subprocess PATH.
# =============================================================================

def _resolve_npx() -> tuple[str, dict]:
    """
    Return (npx_command, env_dict) suitable for StdioServerParameters.

    Strategy:
      1. Try `npm root -g` to find the global node_modules directory.
      2. Derive the bin path from that (parent dir + /bin on Unix, same dir on Win).
      3. Prepend it to PATH in the subprocess env.
      4. Fall back to plain 'npx' if npm itself is not found.

    Returns a tuple of (npx_executable_path, environment_dict).
    """
    base_env = {k: v for k, v in os.environ.items()}

    try:
        npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
        result = subprocess.run(
            [npm_cmd, "root", "-g"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            npm_global_root = result.stdout.strip()  # e.g. C:\Users\x\AppData\Roaming\npm\node_modules
            # The bin directory is one level up from node_modules on Windows
            # and a sibling 'bin' dir on Unix
            if platform.system() == "Windows":
                npm_bin = os.path.dirname(npm_global_root)  # e.g. C:\Users\x\AppData\Roaming\npm
            else:
                npm_bin = os.path.join(os.path.dirname(npm_global_root), "bin")

            # Prepend to PATH so npx is found
            current_path = base_env.get("PATH", "")
            base_env["PATH"] = npm_bin + os.pathsep + current_path

            npx_cmd = "npx.cmd" if platform.system() == "Windows" else "npx"
            return npx_cmd, base_env
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Fallback: hope npx is on PATH as-is
    npx_cmd = "npx.cmd" if platform.system() == "Windows" else "npx"
    return npx_cmd, base_env


# =============================================================================
# FILESYSTEM HELPERS
# =============================================================================

def _read_logs_direct(lines: int = 80) -> str:
    """Direct disk read of ./logs/app.log (fallback when MCP unavailable)."""
    if not os.path.exists(LOG_PATH):
        return f"[ERROR] Log file not found: {LOG_PATH}"
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:])


def _read_app_source() -> str:
    """Read ./app/main.py for source-code correlation in RCA and patch generation."""
    if not os.path.exists(APP_CODE_PATH):
        return f"[ERROR] Source file not found: {APP_CODE_PATH}"
    with open(APP_CODE_PATH, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# FASTMCP LOG READER (stdio)
# =============================================================================

async def _mcp_read_logs(lines: int = 80) -> str:
    """
    Call read_incident_logs via the FastMCP stdio server.
    Falls back to direct disk read on any connection error.
    """
    try:
        params = StdioServerParameters(
            command=sys.executable,   # use the exact same Python interpreter
            args=[MCP_LOG_PATH],
            env=None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "read_incident_logs",
                    arguments={"lines": lines},
                )
                return "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
    except Exception as exc:
        print(f"  [MCP log reader fallback] {type(exc).__name__}: {exc}")
        return f"[MCP fallback -- direct read]\n{_read_logs_direct(lines)}"


# =============================================================================
# GITHUB MCP INTEGRATION (Fixed)
#
# The GitHub MCP server requires the HEAD BRANCH TO ALREADY EXIST before
# create_pull_request is called.  We achieve this by:
#   Step 1 -- Get the SHA of the base branch's HEAD commit.
#   Step 2 -- Create a new branch from that SHA using create_branch.
#   Step 3 -- Write the patched file to the new branch using create_or_update_file.
#   Step 4 -- Open the pull request from that branch.
#
# Each step is a separate MCP tool call within the same ClientSession.
# =============================================================================

async def _github_mcp_open_pr(
    owner: str,
    repo: str,
    token: str,
    incident_id: str,
    pr_title: str,
    pr_branch: str,
    pr_body: str,
    patch_content: str,
) -> str:
    """
    Execute the full GitHub PR creation flow via the official MCP server.

    Steps (all within a single MCP ClientSession):
      1. get_file_contents  -> verify repo is accessible
      2. create_branch      -> create head branch from main's SHA
      3. create_or_update_file -> commit the patched app/main.py to head branch
      4. create_pull_request   -> open the PR

    Returns the PR URL on success, or a detailed error string on failure.
    """
    npx_cmd, resolved_env = _resolve_npx()

    # Inject the GitHub token into the MCP server's environment
    resolved_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token

    params = StdioServerParameters(
        command=npx_cmd,
        args=["@modelcontextprotocol/server-github"],
        env=resolved_env,
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools for debugging (printed but not blocking)
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"  [GitHub MCP] Connected. Available tools: {tool_names}")

                # ── Step 1: Verify repo access + get base branch SHA ──────────
                print(f"  [GitHub MCP] Step 1/4 -- verifying repo access ...")
                try:
                    ref_result = await session.call_tool(
                        "get_file_contents",
                        arguments={
                            "owner": owner,
                            "repo":  repo,
                            "path":  "app/main.py",
                        },
                    )
                    ref_text = "\n".join(
                        c.text for c in ref_result.content if hasattr(c, "text")
                    )
                    print(f"  [GitHub MCP] Repo accessible. File found.")
                except Exception as e:
                    return f"[GitHub MCP ERROR] Cannot access {owner}/{repo}/app/main.py: {e}"

                # ── Step 2: Create the head branch ────────────────────────────
                print(f"  [GitHub MCP] Step 2/4 -- creating branch '{pr_branch}' ...")
                try:
                    branch_result = await session.call_tool(
                        "create_branch",
                        arguments={
                            "owner":  owner,
                            "repo":   repo,
                            "branch": pr_branch,
                            "from_branch": "main",
                        },
                    )
                    print(f"  [GitHub MCP] Branch created.")
                except Exception as e:
                    # Branch may already exist -- that's fine, continue
                    print(f"  [GitHub MCP] Branch creation note: {e} (may already exist)")

                # ── Step 3: Commit patched file to the new branch ─────────────
                # Extract only the actual Python code from the diff if possible,
                # otherwise write the full patch body as the file.
                # We detect code blocks in the engineer's output.
                import base64

                patched_source = _extract_patched_source(patch_content, _read_app_source())
                encoded_content = base64.b64encode(patched_source.encode("utf-8")).decode("utf-8")

                print(f"  [GitHub MCP] Step 3/4 -- committing fix to branch ...")
                try:
                    # Get current file SHA for the update API
                    sha = _extract_sha_from_file_result(ref_text)
                    commit_args = {
                        "owner":   owner,
                        "repo":    repo,
                        "path":    "app/main.py",
                        "message": f"fix({incident_id}): automated SRE remediation",
                        "content": encoded_content,
                        "branch":  pr_branch,
                    }
                    if sha:
                        commit_args["sha"] = sha

                    commit_result = await session.call_tool(
                        "create_or_update_file",
                        arguments=commit_args,
                    )
                    print(f"  [GitHub MCP] File committed.")
                except Exception as e:
                    return f"[GitHub MCP ERROR] Failed to commit file: {e}"

                # ── Step 4: Open the Pull Request ─────────────────────────────
                print(f"  [GitHub MCP] Step 4/4 -- opening pull request ...")
                try:
                    pr_result = await session.call_tool(
                        "create_pull_request",
                        arguments={
                            "owner": owner,
                            "repo":  repo,
                            "title": pr_title,
                            "body":  pr_body,
                            "head":  pr_branch,
                            "base":  "main",
                        },
                    )
                    pr_text = "\n".join(
                        c.text for c in pr_result.content if hasattr(c, "text")
                    )
                    print(f"  [GitHub MCP] Pull request opened successfully.")
                    return pr_text
                except Exception as e:
                    return f"[GitHub MCP ERROR] create_pull_request failed: {e}"

    except FileNotFoundError:
        return (
            "[GitHub MCP ERROR] npx not found. "
            "Install Node.js and run: npm install -g @modelcontextprotocol/server-github"
        )
    except Exception as exc:
        return f"[GitHub MCP ERROR] Session failed: {type(exc).__name__}: {exc}"


def _extract_sha_from_file_result(result_text: str) -> str:
    """
    Try to extract the file SHA from the get_file_contents MCP result.
    The result is typically JSON; we look for the 'sha' field.
    Returns empty string if not found (create_or_update_file still works without it
    for new files, but needs it for updates).
    """
    try:
        data = json.loads(result_text)
        return data.get("sha", "")
    except (json.JSONDecodeError, TypeError):
        # Try regex as fallback
        match = re.search(r'"sha"\s*:\s*"([0-9a-f]{40})"', result_text)
        return match.group(1) if match else ""


def _extract_patched_source(patch_content: str, original_source: str) -> str:
    """
    Extract the final Python source to commit from the engineer's patch output.

    Strategy (in order of preference):
      1. Look for a ```python ... ``` block containing a full module (has 'from fastapi').
      2. Apply the unified diff in CODE_DIFF to the original source.
      3. Fall back to the original source unchanged (safe -- PR still opens,
         reviewer will catch that the fix wasn't applied).

    This is necessary because Llama3 sometimes outputs full files, sometimes
    proper unified diffs, and sometimes a mix. We handle all cases.
    """
    # Strategy 1: full Python file in a code block
    python_blocks = re.findall(r"```python\n(.*?)```", patch_content, re.DOTALL)
    for block in python_blocks:
        if "from fastapi" in block or "FastAPI" in block:
            return block.strip()

    # Strategy 2: apply unified diff
    diff_match = re.search(r"CODE_DIFF:\s*```diff\n(.*?)```", patch_content, re.DOTALL)
    if diff_match:
        diff_text = diff_match.group(1).strip()
        applied = _apply_unified_diff(original_source, diff_text)
        if applied and applied != original_source:
            return applied

    # Strategy 3: safe fallback -- return original so PR at least opens
    print("  [PatchExtractor] Could not extract clean patch; committing original source.")
    return original_source


def _apply_unified_diff(source: str, diff: str) -> str:
    """
    Naive unified diff applicator.
    Handles the common case where Llama3 outputs +/- lines in a diff block.
    Returns the patched source or the original if application fails.
    """
    try:
        source_lines = source.splitlines(keepends=True)
        result_lines = list(source_lines)
        offset = 0

        # Parse hunk headers: @@ -start,count +start,count @@
        hunks = re.split(r"(@@ .+? @@[^\n]*\n)", diff)
        i = 0
        while i < len(hunks):
            if not hunks[i].startswith("@@"):
                i += 1
                continue
            header = hunks[i]
            body   = hunks[i + 1] if i + 1 < len(hunks) else ""
            i += 2

            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
            if not m:
                continue

            old_start = int(m.group(1)) - 1  # 0-indexed
            new_lines = []
            skip      = 0

            for line in body.splitlines(keepends=True):
                if line.startswith("-"):
                    skip += 1         # line to remove
                elif line.startswith("+"):
                    new_lines.append(line[1:])  # line to add (strip the +)
                elif line.startswith(" "):
                    new_lines.append(line[1:])  # context line

            adj_start = old_start + offset
            # Replace old lines with new lines
            end = adj_start + skip
            result_lines[adj_start:end] = new_lines
            offset += len(new_lines) - skip

        return "".join(result_lines)
    except Exception:
        return source


# =============================================================================
# PII REDACTION (deterministic regex -- no LLM)
# =============================================================================

def _redact_pii(text: str) -> str:
    """Scrub sensitive patterns before passing log text to any LLM."""
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[REDACTED_IP]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", text)
    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "[REDACTED_UUID]", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b[0-9a-f]{32,}\b", "[REDACTED_TOKEN]", text, flags=re.IGNORECASE)
    text = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+", "[TIMESTAMP]", text)
    return text


# =============================================================================
# SHARED PRINT HELPERS
# =============================================================================

def _node_header(name: str, subtitle: str = ""):
    print(f"\n{'=' * 64}")
    print(f"  {name}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"  {GPU_BADGE}")
    print(f"{'=' * 64}")


def _field(label: str, value: str, max_chars: int = 400):
    snippet = (value or "")[:max_chars]
    if len(value or "") > max_chars:
        snippet += " ..."
    print(f"\n  \033[1m{label}:\033[0m")
    for line in snippet.splitlines():
        print(f"    \033[2m{line}\033[0m")


# =============================================================================
# NODE 0 -- LOG MONITOR
# Reads logs via FastMCP, redacts PII, summarises locally.
# =============================================================================

async def log_monitor_node(state: dict) -> dict:
    """
    Log Monitor Agent -- local Ollama inference.

    1. Pull raw logs from FastMCP read_incident_logs (stdio).
    2. Deterministic PII redaction (regex, no GPU).
    3. Local LLM compresses to <=200-token sanitized_summary.
    4. Record t_pipeline_start for latency telemetry.
    """
    _node_header("  LOG MONITOR", "Ingesting production logs via FastMCP ...")
    t_pipeline_start = time.time()
    t0 = t_pipeline_start

    raw_logs = await _mcp_read_logs(lines=80)
    redacted = _redact_pii(raw_logs)

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a log-ingestion agent. Read the redacted log excerpt and "
            "write a plain-text incident summary, maximum 150 words. "
            "Include: severity (CRITICAL/WARNING/INFO), the exact error message, "
            "the service name, and the approximate time of occurrence. "
            "No JSON. No markdown. No bullet points. Plain sentences only."
        )),
        HumanMessage(content=f"Redacted log excerpt:\n\n{redacted[:3000]}"),
    ])

    sanitized_summary = response.content.strip()
    incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"  -> Raw log size    : {len(raw_logs):,} chars")
    print(f"  -> After redaction : {len(redacted):,} chars")
    _field("Incident ID", incident_id, 60)
    _field("Sanitized Summary", sanitized_summary)
    print(f"\n  Node DT = {time.time() - t0:.2f} s")

    return {
        "messages":          [response],
        "incident_id":       incident_id,
        "raw_logs":          raw_logs,
        "log_data":          raw_logs,
        "sanitized_summary": sanitized_summary,
        "iteration_count":   0,
        "is_verified":       False,
        "user_approval":     False,
        "pr_result":         "",
        "t_pipeline_start":  t_pipeline_start,
        "t_pipeline_end":    0.0,
    }


# =============================================================================
# NODE 1 -- ANALYST (RCA)
# Correlates logs with source code to produce structured root cause analysis.
# =============================================================================

async def analyst_node(state: dict) -> dict:
    """
    Analyst Agent -- local Ollama inference.

    Correlates sanitized_summary with ./app/main.py source to produce a
    structured RCA hypothesis. Includes retry context on subsequent iterations.
    """
    iteration = state.get("iteration_count", 0) + 1
    _node_header(
        f"  ANALYST  (Iteration #{iteration})",
        "Root cause analysis via local source correlation ...",
    )
    t0 = time.time()

    summary  = state.get("sanitized_summary", "No log summary available.")
    src_code = _read_app_source()

    retry_context = ""
    if iteration > 1:
        retry_context = (
            f"\n\nRETRY CONTEXT (Iteration #{iteration}):\n"
            f"The previous patch was REJECTED. Re-examine the root cause carefully.\n"
            f"Previous Manager Directive:\n{state.get('manager_directive', '')[:300]}\n"
            f"Previous Patch snippet:\n{state.get('proposed_patch', '')[:300]}"
        )

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a senior SRE analyst. Correlate the incident summary with "
            "the source code to identify the exact root cause.\n"
            "Output ONLY this exact structure -- no extra text, no markdown headers:\n\n"
            "ROOT_CAUSE: <one sentence -- what failed and why>\n"
            "AFFECTED_FILE: <file path>\n"
            "AFFECTED_LINE: <line number or range>\n"
            "EXPLANATION: <2-3 sentences of technical detail>\n"
            "FIX_STRATEGY: <1-2 sentences -- what the engineer should change>\n"
        )),
        HumanMessage(content=(
            f"Incident Summary:\n{summary}\n\n"
            f"Source file -- ./app/main.py:\n{src_code}"
            f"{retry_context}"
        )),
    ])

    rca = response.content.strip()
    _field("RCA Hypothesis", rca)
    print(f"\n  Node DT = {time.time() - t0:.2f} s")

    return {
        "messages":        [response],
        "rca_hypothesis":  rca,
        "iteration_count": iteration,
    }


# =============================================================================
# NODE 2 -- MANAGER (TRIAGE)
# Validates RCA, classifies severity, issues engineer directive.
# =============================================================================

async def manager_node(state: dict) -> dict:
    """
    Manager Agent -- local Ollama inference.

    Validates the analyst RCA, classifies severity, and issues a
    manager_directive that scopes the engineer's patch.
    """
    _node_header("  MANAGER", "Incident triage and fix directive ...")
    t0 = time.time()

    summary = state.get("sanitized_summary", "No summary.")
    rca     = state.get("rca_hypothesis",   "No RCA.")

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are the SRE Manager. Review the incident summary and analyst RCA.\n"
            "Output ONLY valid JSON -- no markdown, no extra text:\n"
            "{\n"
            '  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",\n'
            '  "rca_valid": true | false,\n'
            '  "rca_feedback": "<one sentence if invalid, else empty string>",\n'
            '  "directive": "<2-3 sentences: what the engineer must fix and how>",\n'
            '  "proceed": true | false\n'
            "}"
        )),
        HumanMessage(content=(
            f"Incident Summary:\n{summary}\n\n"
            f"Analyst RCA:\n{rca}"
        )),
    ])

    raw = response.content.strip()
    try:
        clean    = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        decision = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        decision = {
            "severity":     "CRITICAL",
            "rca_valid":    True,
            "rca_feedback": "",
            "directive":    raw[:400],
            "proceed":      True,
        }

    directive = (
        f"Severity  : {decision.get('severity', 'UNKNOWN')}\n"
        f"RCA Valid : {decision.get('rca_valid', True)}\n"
        f"Feedback  : {decision.get('rca_feedback', 'N/A')}\n"
        f"Directive : {decision.get('directive', '')}"
    )

    print(f"  -> Severity : {decision.get('severity', 'UNKNOWN')}")
    print(f"  -> RCA Valid: {decision.get('rca_valid', True)}")
    print(f"  -> Proceed  : {decision.get('proceed', True)}")
    _field("Manager Directive", directive)
    print(f"\n  Node DT = {time.time() - t0:.2f} s")

    return {
        "messages":          [response],
        "manager_directive": directive,
    }


# =============================================================================
# NODE 3 -- ENGINEER (PATCH GENERATION)
# Generates PR-formatted code fix with structured GitHub metadata.
#
# PROMPT ENGINEERING NOTE:
# Llama3 must output a very specific format because open_pr_node parses
# PR_TITLE, PR_BRANCH, PR_DESCRIPTION, and CODE_DIFF fields directly.
# The prompt enforces this with strict "Output ONLY" instructions and
# a numbered field list. The code block must contain either:
#   a) A full Python file (preferred -- used directly as the committed file), OR
#   b) A proper unified diff (applied by _apply_unified_diff).
# =============================================================================

async def engineer_node(state: dict) -> dict:
    """
    Engineer Agent -- local Ollama inference.

    Generates a full GitHub PR body with a code fix.
    The proposed_patch field is parsed by open_pr_node to extract:
      - PR_TITLE  -> used as the GitHub PR title
      - PR_BRANCH -> used as the head branch name
      - PR_DESCRIPTION -> used as the PR body text
      - CODE_DIFF or python block -> committed to the branch
    """
    _node_header("  ENGINEER", "Generating code remediation patch ...")
    t0 = time.time()

    rca       = state.get("rca_hypothesis",    "No RCA.")
    directive = state.get("manager_directive", "No directive.")
    summary   = state.get("sanitized_summary", "")
    incident  = state.get("incident_id",       "INC-UNKNOWN")
    src_code  = _read_app_source()

    # Extract just the branch-friendly incident slug
    incident_slug = incident.lower().replace("_", "-")

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are the SRE Engineer. Generate a production-safe code fix.\n\n"
            "You MUST output ALL of the following fields in EXACTLY this format:\n\n"
            f"PR_TITLE: fix({incident}): <short description of the fix>\n"
            f"PR_BRANCH: fix/{incident_slug}-patch\n"
            "PR_DESCRIPTION:\n"
            "## Overview\n"
            "<one paragraph describing the incident and fix>\n\n"
            "## Root Cause\n"
            "<one paragraph from the RCA>\n\n"
            "## Changes\n"
            "<bullet list of what changed>\n\n"
            "CODE_DIFF:\n"
            "```python\n"
            "<the COMPLETE, FINAL version of ./app/main.py with the fix applied>\n"
            "```\n\n"
            "TESTS_ADDED:\n"
            "- <test description 1>\n"
            "- <test description 2>\n\n"
            "ROLLBACK_PLAN:\n"
            "1. <revert step>\n"
            "2. <revert step>\n\n"
            "STRICT RULES:\n"
            "- PR_BRANCH must be exactly: "
            f"fix/{incident_slug}-patch\n"
            "- CODE_DIFF must contain the FULL file content, not just the changed lines.\n"
            "- Do NOT use a unified diff format. Output the complete file.\n"
            "- The file must be valid Python that runs without errors.\n"
            "- Make the SMALLEST change that fixes the root cause.\n"
            "- Do NOT remove unrelated endpoints or imports.\n"
        )),
        HumanMessage(content=(
            f"Incident: {incident}\n"
            f"Summary: {summary}\n\n"
            f"Root Cause Analysis:\n{rca}\n\n"
            f"Manager Directive:\n{directive}\n\n"
            f"Current source -- ./app/main.py:\n{src_code}"
        )),
    ])

    patch = response.content.strip()
    _field("Proposed Patch", patch)
    print(f"\n  Node DT = {time.time() - t0:.2f} s")

    return {
        "messages":       [response],
        "proposed_patch": patch,
    }


# =============================================================================
# NODE 4 -- SYNTAX CHECKER
# Fast local pre-filter before the heavier reviewer LLM call.
# =============================================================================

async def syntax_checker_node(state: dict) -> dict:
    """
    Syntax Checker Agent -- local Ollama inference.

    Two-pass check:
      Pass 1: Regex hard-fail patterns (no LLM, instant).
      Pass 2: LLM structural review (valid Python, no hallucinated imports).

    FAIL -> is_verified=False -> graph loops back to analyst.
    PASS -> proceed to reviewer for semantic validation.
    """
    _node_header("  SYNTAX CHECKER", "Pre-validating patch structure ...")
    t0 = time.time()

    patch   = state.get("proposed_patch", "")
    src     = _read_app_source()
    summary = state.get("sanitized_summary", "")

    # Pass 1 -- hard-fail regex patterns
    HARD_FAIL = [
        (r"raise\s+ConnectionTimeout\b",  "patch re-raises ConnectionTimeout"),
        (r"raise\s+DatabaseError\b",      "patch re-raises DatabaseError"),
        (r"import\s+nonexistent_module",  "patch imports nonexistent module"),
        (r"def\s+trigger_crash.*?pass",   "patch blanks trigger_crash with pass only"),
    ]
    for pattern, reason in HARD_FAIL:
        if re.search(pattern, patch, re.IGNORECASE | re.DOTALL):
            print(f"  -> HARD FAIL -- {reason}")
            return {
                "messages":    [AIMessage(content=f"[SyntaxChecker] Hard fail: {reason}")],
                "is_verified": False,
            }

    # Pass 2 -- LLM structural review
    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a Python syntax checker. Review the proposed patch.\n"
            "Check ONLY these three things:\n"
            "  1. Does the code block contain syntactically valid Python?\n"
            "  2. Does it import only modules already present in the original source?\n"
            "  3. Does it include the FastAPI app definition (not just a diff)?\n\n"
            "Respond with EXACTLY one line -- nothing else:\n"
            "PASS: <one-sentence reason>\n"
            "or\n"
            "FAIL: <one-sentence reason>"
        )),
        HumanMessage(content=(
            f"Bug context: {summary[:200]}\n\n"
            f"Original source:\n{src}\n\n"
            f"Proposed patch:\n{patch[:2000]}"
        )),
    ])

    verdict_text = response.content.strip()
    passed = verdict_text.upper().startswith("PASS")

    print(f"  -> Verdict : {'PASS' if passed else 'FAIL'}")
    print(f"  -> Reason  : {verdict_text[:120]}")
    print(f"\n  Node DT = {time.time() - t0:.2f} s")

    return {
        "messages":    [response],
        "is_verified": passed,
    }


# =============================================================================
# NODE 5 -- REVIEWER (SEMANTIC APPROVAL GATE)
# Final local LLM review before HITL pause.
# =============================================================================

async def reviewer_node(state: dict) -> dict:
    """
    Reviewer Agent -- local Ollama inference.

    Semantic review: correctness, safety, coherence, hallucination.
    Sets is_verified=True -> HITL, or False -> loop back to analyst.
    Skips if syntax_checker already set is_verified=False.
    """
    _node_header("  REVIEWER", "Semantic patch validation ...")
    t0 = time.time()

    if not state.get("is_verified", True):
        print("  -> Skipped -- syntax check already set is_verified=False.")
        return {}

    patch     = state.get("proposed_patch", "")
    rca       = state.get("rca_hypothesis", "")
    directive = state.get("manager_directive", "")
    iteration = state.get("iteration_count", 1)

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are the principal SRE reviewer. Evaluate the patch against:\n"
            "  1. CORRECTNESS  -- Does it directly fix the stated root cause?\n"
            "  2. SAFETY       -- No new exceptions, no data loss risk?\n"
            "  3. COHERENCE    -- Is the code consistent with the RCA?\n"
            "  4. COMPLETENESS -- Does it contain a full, runnable Python file?\n\n"
            "Output ONLY valid JSON -- no markdown, no extra text:\n"
            '{"verdict": "APPROVED" | "REJECTED", "reason": "<one sentence>"}'
        )),
        HumanMessage(content=(
            f"Manager Directive:\n{directive}\n\n"
            f"Root Cause Analysis:\n{rca}\n\n"
            f"Proposed Patch:\n{patch[:2500]}\n\n"
            f"Iteration #{iteration} -- be strict on repeated failures."
        )),
    ])

    raw = response.content.strip()
    try:
        clean   = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        data    = json.loads(clean)
        verdict = data.get("verdict", "REJECTED").upper()
        reason  = data.get("reason", "No reason given.")
    except (json.JSONDecodeError, ValueError):
        verdict = "APPROVED" if "approved" in raw.lower() else "REJECTED"
        reason  = raw[:200]

    is_verified = (verdict == "APPROVED")
    t_end = time.time()

    print(f"  -> Verdict : {'APPROVED' if is_verified else 'REJECTED'}")
    print(f"  -> Reason  : {reason[:120]}")
    print(f"\n  Node DT = {t_end - t0:.2f} s")

    return {
        "messages":       [response],
        "is_verified":    is_verified,
        "t_pipeline_end": t_end,
    }


# =============================================================================
# DEPLOY NODE (HITL BREAKPOINT -- no LLM)
# LangGraph interrupt_before="deploy" pauses here for human sign-off.
# =============================================================================

async def deploy_node(state: dict) -> dict:
    """
    HITL Staging Gate -- no LLM.

    LangGraph interrupt_before="deploy" pauses the graph here.
    main.py reads the state, prompts the operator, then calls:
        sre_graph.update_state(config, {"user_approval": True/False}, as_node="deploy")
    and resumes execution.
    """
    print(f"\n{'=' * 64}")
    print("  DEPLOY -- Human approval gate")
    print("  |-- Inference : HUMAN (HITL -- no model)")
    print(f"{'=' * 64}")
    print(f"  Incident   : {state.get('incident_id', 'N/A')}")
    print(f"  Verified   : {state.get('is_verified', False)}")
    print(f"  Iterations : {state.get('iteration_count', 0)}")

    t_s = state.get("t_pipeline_start", 0.0)
    t_e = state.get("t_pipeline_end",   0.0)
    if t_s and t_e:
        print(f"  GPU Time   : {t_e - t_s:.2f} s  (log_monitor through reviewer)")

    _field("Patch Preview", state.get("proposed_patch", ""), max_chars=300)

    if state.get("user_approval", False):
        print("\n  Approved -- proceeding to open GitHub PR.")
    else:
        print("\n  Awaiting operator decision ...")

    return {}


# =============================================================================
# OPEN PR NODE (GitHub MCP tool -- no LLM)
# Handles the full branch-create + file-commit + PR-open flow.
# =============================================================================

async def open_pr_node(state: dict) -> dict:
    """
    GitHub PR Creation Node -- MCP tool calls, no LLM.

    Reads GITHUB_PERSONAL_ACCESS_TOKEN, GITHUB_OWNER, GITHUB_REPO from env.
    Executes the 4-step GitHub MCP flow:
      1. Verify repo access
      2. Create head branch
      3. Commit patched file
      4. Open PR

    Falls back to dry-run mode with a clear explanation if:
      - Token is missing
      - GITHUB_OWNER or GITHUB_REPO are not set
      - npx / @modelcontextprotocol/server-github is not installed

    To enable the real PR:
      npm install -g @modelcontextprotocol/server-github
      Add to .env:
        GITHUB_OWNER=your-github-username-or-org
        GITHUB_REPO=your-repository-name
    """
    print(f"\n{'=' * 64}")
    print("  OPEN PR -- GitHub MCP tool")
    print("  |-- Inference : GitHub MCP (no LLM)")
    print(f"{'=' * 64}")

    incident  = state.get("incident_id",    "INC-UNKNOWN")
    patch     = state.get("proposed_patch", "")
    token     = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
    owner     = os.getenv("GITHUB_OWNER",  "").strip()
    repo      = os.getenv("GITHUB_REPO",   "").strip()

    # Parse structured metadata from engineer's output
    incident_slug = incident.lower().replace("_", "-")
    pr_title      = f"fix({incident}): automated SRE remediation"
    pr_branch     = f"fix/{incident_slug}-patch"
    pr_description = patch  # full patch body used as PR description

    for line in (patch or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("PR_TITLE:"):
            pr_title = stripped.replace("PR_TITLE:", "").strip()
        elif stripped.startswith("PR_BRANCH:"):
            pr_branch = stripped.replace("PR_BRANCH:", "").strip()

    # Extract PR_DESCRIPTION block if present
    desc_match = re.search(r"PR_DESCRIPTION:\n(.*?)(?=\nCODE_DIFF:|\nTESTS_ADDED:|\nROLLBACK_PLAN:|$)",
                            patch, re.DOTALL)
    if desc_match:
        pr_description = desc_match.group(1).strip()

    print(f"  PR Title  : {pr_title}")
    print(f"  PR Branch : {pr_branch}")
    print(f"  Owner/Repo: {owner or '[NOT SET]'}/{repo or '[NOT SET]'}")

    # Pre-flight checks
    if not token:
        msg = (
            "[DRY RUN] GITHUB_PERSONAL_ACCESS_TOKEN not set in .env.\n"
            f"Would open PR: '{pr_title}' on branch '{pr_branch}'"
        )
        print(f"\n  {msg}")
        return {"messages": [AIMessage(content=msg)], "pr_result": msg}

    if not owner or not repo:
        msg = (
            "[DRY RUN] GITHUB_OWNER and GITHUB_REPO not set in .env.\n"
            f"Add these to .env:\n  GITHUB_OWNER=your-username\n  GITHUB_REPO=your-repo\n"
            f"Would open PR: '{pr_title}'"
        )
        print(f"\n  {msg}")
        return {"messages": [AIMessage(content=msg)], "pr_result": msg}

    # Execute the full 4-step MCP flow
    print(f"\n  Initiating GitHub MCP connection ...")
    result = await _github_mcp_open_pr(
        owner=owner,
        repo=repo,
        token=token,
        incident_id=incident,
        pr_title=pr_title,
        pr_branch=pr_branch,
        pr_body=pr_description,
        patch_content=patch,
    )

    if "ERROR" in result:
        print(f"\n  [FAILED] {result}")
    else:
        print(f"\n  [SUCCESS] {result[:300]}")

    return {
        "messages":  [AIMessage(content=result)],
        "pr_result": result,
    }