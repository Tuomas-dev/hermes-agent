"""Task-scoped lifecycle fence shared by Codex and its Hermes MCP child."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


KANBAN_TERMINAL_FENCE_ENV = "HERMES_KANBAN_TERMINAL_FENCE"
TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})


def publish_terminal_callback(tool_name: str, result: str) -> bool:
    """Publish a successful terminal callback for the owning app-server.

    The callback has already committed task/run state before this function is
    called.  Errors deliberately do not publish: Codex must remain alive so it
    can correct the call and retry.
    """
    path = os.environ.get(KANBAN_TERMINAL_FENCE_ENV)
    task_id = os.environ.get("HERMES_KANBAN_TASK")
    if not path or not task_id or tool_name not in TERMINAL_KANBAN_TOOLS:
        return False
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return False

    marker = {
        "tool": tool_name,
        "task_id": task_id,
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
    }
    marker_path = Path(path)
    tmp_path = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, marker_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def read_terminal_callback(path: Optional[str]) -> Optional[dict[str, Any]]:
    """Read a valid terminal marker, returning ``None`` until it exists."""
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("tool") not in TERMINAL_KANBAN_TOOLS:
        return None
    if not payload.get("task_id"):
        return None
    return payload
