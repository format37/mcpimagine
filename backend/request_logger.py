"""
Request logging utility for MCP tools.

Logs all tool invocations to JSON files for audit and analytics purposes.
Input params and outputs are truncated so logs never balloon (image bytes are
never written here — only metadata).
"""

import json
import logging
import time
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


def log_request(
    requests_dir: pathlib.Path,
    requester: str,
    tool_name: str,
    input_params: Dict[str, Any],
    output_result: Any,
) -> pathlib.Path:
    """Log a single tool request to a JSON file. Never raises."""
    timestamp_ms = int(time.time() * 1000)
    safe_requester = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(requester))
    filename = f"{timestamp_ms}-{tool_name}-{safe_requester}.json"
    filepath = requests_dir / filename

    record = {
        "timestamp_ms": timestamp_ms,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "requester": requester,
        "tool_name": tool_name,
        "input_params": _sanitize(input_params),
        "output_result": _sanitize(output_result),
    }

    try:
        requests_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str, ensure_ascii=False)
        logger.debug("Logged request to %s", filename)
    except Exception as e:  # logging must never break a tool
        logger.error("Failed to log request: %s", e)

    return filepath


def _sanitize(value: Any) -> Any:
    """Truncate large strings (e.g. base64 image input) so logs stay small."""
    max_length = 2000
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + f"... [TRUNCATED, total length: {len(value)}]"
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value
