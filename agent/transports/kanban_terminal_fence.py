"""Trusted, board-backed lifecycle fence for Codex Kanban workers."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Optional


# Retained as an import-compatible constant for older callers. It is not
# exported to the worker process: a worker-writable pathname is not authority.
KANBAN_TERMINAL_FENCE_ENV = "HERMES_KANBAN_TERMINAL_FENCE"
TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})


def publish_terminal_callback(tool_name: str, result: str) -> bool:
    """Deprecated worker-side publisher; terminal authority lives in SQLite."""
    return False


def read_terminal_callback(path: Optional[str]) -> Optional[dict[str, Any]]:
    """Deprecated marker reader; worker-created files are never trusted."""
    return None


def read_authoritative_terminal_callback(
    db_path: Optional[str],
    task_id: Optional[str],
    tool_name: str,
    run_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return terminal state only when committed board rows prove this callback.

    The app-server parent calls this only for a completed Hermes MCP invocation.
    The query binds the expected task and current run to the same ended run and
    terminal event. A read-only connection observes committed transactions only.
    """
    if (
        not db_path
        or not task_id
        or not run_id
        or tool_name not in TERMINAL_KANBAN_TOOLS
    ):
        return None
    try:
        expected_run_id = int(run_id)
    except (TypeError, ValueError):
        return None

    if tool_name == "kanban_complete":
        expected_outcome = "completed"
        expected_events = ("completed",)
    else:
        expected_outcome = "blocked"
        expected_events = ("blocked", "dependency_wait", "block_loop_detected")

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(
            f"file:{os.path.abspath(db_path)}?mode=ro", uri=True
        )
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in expected_events)
        row = conn.execute(
            "SELECT e.run_id, e.kind, r.ended_at, r.outcome "
            "FROM task_runs r "
            "JOIN task_events e ON e.task_id=r.task_id AND e.run_id=r.id "
            f"WHERE r.task_id=? AND r.id=? AND e.kind IN ({placeholders}) "
            "ORDER BY e.id DESC LIMIT 1",
            (task_id, expected_run_id, *expected_events),
        ).fetchone()
    except (OSError, sqlite3.Error):
        return None
    finally:
        if conn is not None:
            conn.close()

    if (
        row is None
        or row["ended_at"] is None
        or int(row["run_id"]) != expected_run_id
        or row["outcome"] != expected_outcome
    ):
        return None
    return {
        "tool": tool_name,
        "task_id": task_id,
        "run_id": expected_run_id,
        "status": row["kind"],
    }
