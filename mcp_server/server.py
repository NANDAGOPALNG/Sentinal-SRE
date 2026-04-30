from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("SentinalSRE-Logs")

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "app.log"))

@mcp.tool()
def read_incident_logs(lines: int = 20) -> str:
    """Reads the most recent lines from the production app logs to triage incidents."""
    try:
        if not os.path.exists(LOG_PATH):
            return "Log file not found. Ensure the Sandbox is running."
        with open(LOG_PATH, "r") as f:
            content = f.readlines()[-lines:]
            return "".join(content)
    except Exception as e:
        return f"Error reading logs: {str(e)}"

@mcp.tool()
def check_log_health() -> str:
    """Returns the file size and last modified time of the log file."""
    if os.path.exists(LOG_PATH):
        stats = os.stat(LOG_PATH)
        return f"Size: {stats.st_size} bytes | Last Updated: {stats.st_mtime}"
    return "Log file missing."

if __name__ == "__main__":
    mcp.run()
