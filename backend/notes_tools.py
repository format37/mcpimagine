"""Simple file-backed tool-notes knowledge base (no heavy deps)."""

import logging
import re
from datetime import datetime
from pathlib import Path

from request_logger import log_request

logger = logging.getLogger(__name__)

MAX_TOOL_NAME = 200
MAX_NOTES = 100_000


def _safe_name(tool_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", tool_name).strip("._-") or "notes"


def register_notes_tools(mcp, notes_dir: Path, requests_dir: Path):
    notes_dir.mkdir(parents=True, exist_ok=True)

    @mcp.tool()
    def save_tool_notes(requester: str, tool_name: str, markdown_notes: str) -> str:
        """Append timestamped usage notes / lessons learned about an MCP tool.

        Parameters:
            requester (str): Who is calling, for request logging.
            tool_name (str): Tool to document (e.g. 'generate_image').
            markdown_notes (str): Markdown notes (gotchas, good prompts, etc.).

        Returns:
            str: Confirmation with the saved file path.
        """
        if len(tool_name) > MAX_TOOL_NAME:
            return f"✗ Error: tool_name too long (max {MAX_TOOL_NAME} chars)"
        if len(markdown_notes) > MAX_NOTES:
            return f"✗ Error: markdown_notes too long (max {MAX_NOTES} chars)"
        safe = _safe_name(tool_name)
        path = notes_dir / f"{safe}.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            mode = "a" if path.exists() else "w"
            with open(path, mode, encoding="utf-8") as f:
                if mode == "w":
                    f.write(f"# Tool Usage Notes: {tool_name}\n")
                f.write(f"\n\n---\n**Added:** {ts}\n\n{markdown_notes}\n")
            result = f"✓ Notes saved\n\nTool: {tool_name}\nFile: tool_notes/{safe}.md\nTimestamp: {ts}"
        except Exception as e:
            result = f"✗ Error saving notes: {e}"
        log_request(requests_dir, requester, "save_tool_notes",
                    {"tool_name": tool_name, "markdown_notes": markdown_notes[:500]}, result)
        return result

    @mcp.tool()
    def read_tool_notes(requester: str, tool_name: str) -> str:
        """Read all saved usage notes for a given MCP tool.

        Parameters:
            requester (str): Who is calling, for request logging.
            tool_name (str): Tool to read notes for (e.g. 'generate_image').

        Returns:
            str: Markdown notes, or a message if none exist.
        """
        safe = _safe_name(tool_name)
        path = notes_dir / f"{safe}.md"
        if not path.exists():
            result = f"No notes found for tool: {tool_name}."
        else:
            try:
                result = path.read_text(encoding="utf-8")
            except Exception as e:
                result = f"✗ Error reading notes: {e}"
        log_request(requests_dir, requester, "read_tool_notes", {"tool_name": tool_name}, result)
        return result
