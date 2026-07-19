"""Process-session fencing for terminal Kanban runs."""

import signal

import pytest

from hermes_cli import kanban_db as kb


pytestmark = pytest.mark.skipif(kb.os.name == "nt", reason="POSIX process groups")


def test_group_cleanup_escalates_only_recorded_group(monkeypatch):
    calls = []
    snapshots = [
        [(4101, 100), (4102, 101)],
        [(4101, 100), (4102, 101)],
        [(4102, 101)],
    ]
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(kb, "_linux_process_identity", lambda pid: (4101, 4101, 100))
    monkeypatch.setattr(
        kb, "_linux_group_members",
        lambda pgid: [
            (pid, 4101, started)
            for pid, started in (snapshots.pop(0) if snapshots else [])
        ],
    )

    result = kb._terminate_worker_process_group(
        4101, 4101, 4101, 100, grace_seconds=0,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == [(4101, signal.SIGTERM), (4101, signal.SIGKILL)]
    assert result["group_sigkill"] is True
    assert all(pgid == 4101 for pgid, _sig in calls)


def test_dead_leader_with_owned_descendants_is_cleaned(monkeypatch):
    calls = []
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(kb, "_linux_process_identity", lambda pid: None)
    snapshots = [[(4202, 4201, 101)], [(4202, 4201, 101)], []]
    monkeypatch.setattr(
        kb, "_linux_group_members",
        lambda pgid: snapshots.pop(0) if snapshots else [],
    )

    result = kb._terminate_worker_process_group(
        4201, 4201, 4201, 100, grace_seconds=0,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == [(4201, signal.SIGTERM), (4201, signal.SIGKILL)]
    assert result["group_terminated"] is True


@pytest.mark.parametrize(
    "identity,members",
    [
        ((4301, 4301, 999), [(4301, 4301, 999)]),
        (None, [(4302, 9999, 101)]),
        (None, None),
    ],
)
def test_reused_identity_is_never_signalled(monkeypatch, identity, members):
    calls = []
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(kb, "_linux_process_identity", lambda pid: identity)
    monkeypatch.setattr(kb, "_linux_group_members", lambda pgid: members)

    result = kb._terminate_worker_process_group(
        4301, 4301, 4301, 100,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == []
    assert result["group_termination_attempted"] is False


def test_dead_leader_with_no_visible_owned_members_is_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(kb, "_linux_process_identity", lambda pid: None)
    monkeypatch.setattr(kb, "_linux_group_members", lambda pgid: [])

    result = kb._terminate_worker_process_group(
        4351, 4351, 4351, 100,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == []
    assert result["group_termination_attempted"] is False
    assert result["group_terminated"] is False


def test_live_leader_missing_from_group_snapshot_is_never_signalled(monkeypatch):
    calls = []
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        kb, "_linux_process_identity", lambda pid: (4361, 4361, 100)
    )
    monkeypatch.setattr(
        kb, "_linux_group_members", lambda pgid: [(4362, 4361, 101)]
    )

    result = kb._terminate_worker_process_group(
        4361, 4361, 4361, 100,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == []
    assert result["group_termination_attempted"] is False


@pytest.mark.parametrize(
    "current_identity",
    [
        (4702, 4701, 100),  # unrelated/reused process group
        (4701, 9999, 100),  # session mismatch
        (4701, 4701, 999),  # process creation token mismatch
        None,  # dead leader or unavailable/partial native metadata
    ],
)
def test_nonlinux_unproved_identity_is_never_signalled(
    monkeypatch, current_identity
):
    calls = []
    monkeypatch.setattr(kb.sys, "platform", "darwin")
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        kb, "_portable_process_identity", lambda pid: current_identity
    )

    result = kb._terminate_worker_process_group(
        4701, 4701, 4701, 100,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == []
    assert result["group_termination_attempted"] is False
    assert result["group_terminated"] is False


def test_nonlinux_exact_live_identity_can_complete_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(kb.sys, "platform", "darwin")
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        kb, "_portable_process_identity", lambda pid: (4711, 4711, 100)
    )

    def signal_group(pgid, sig):
        calls.append((pgid, sig))
        raise ProcessLookupError

    result = kb._terminate_worker_process_group(
        4711, 4711, 4711, 100, killpg_fn=signal_group
    )

    assert calls == [(4711, signal.SIGTERM)]
    assert result["group_termination_attempted"] is True
    assert result["group_terminated"] is True


def test_nonlinux_identity_is_rechecked_before_escalation(monkeypatch):
    calls = []
    identities = iter([(4721, 4721, 100), (4721, 9999, 100)])
    monkeypatch.setattr(kb.sys, "platform", "darwin")
    monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        kb, "_portable_process_identity", lambda pid: next(identities)
    )

    result = kb._terminate_worker_process_group(
        4721, 4721, 4721, 100, grace_seconds=0,
        killpg_fn=lambda pgid, sig: calls.append((pgid, sig)),
    )

    assert calls == [(4721, signal.SIGTERM)]
    assert result["group_sigkill"] is False
    assert result["group_terminated"] is False


def test_nonlinux_spawn_persists_complete_portable_identity(tmp_path, monkeypatch):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="portable identity", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        monkeypatch.setattr(kb.sys, "platform", "darwin")
        monkeypatch.setattr(kb, "_linux_process_identity", lambda pid: None)
        monkeypatch.setattr(
            kb, "_portable_process_identity", lambda pid: (4731, 4731, 123456)
        )

        kb._set_worker_pid(conn, task_id, 4731)
        row = conn.execute(
            "SELECT worker_pgid, worker_sid, worker_start_ticks FROM task_runs "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (task_id,),
        ).fetchone()

        assert tuple(row) == (4731, 4731, 123456)
    finally:
        conn.close()


def test_cleanup_runs_only_after_terminal_transaction(tmp_path, monkeypatch):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="fenced", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        kb._set_worker_pid(conn, task_id, 4401)
        run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE task_runs SET worker_pgid=4401, worker_sid=4401, "
            "worker_start_ticks=100 "
            "WHERE id=?", (run_id,),
        )
        calls = []
        monkeypatch.setattr(
            kb, "_terminate_worker_process_group",
            lambda *args, **kwargs: calls.append(args) or {"group_terminated": True},
        )

        assert kb.cleanup_terminal_worker_groups(conn) == []
        assert calls == []
        assert kb.complete_task(conn, task_id, result="ok") is True
        assert kb.cleanup_terminal_worker_groups(conn) == [run_id]
        assert calls == [(4401, 4401, 4401, 100)]
        assert kb.cleanup_terminal_worker_groups(conn) == []
    finally:
        conn.close()


def test_failed_owned_group_cleanup_is_retried(tmp_path, monkeypatch):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="retry cleanup", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE task_runs SET worker_pid=4501, worker_pgid=4501, "
            "worker_sid=4501, worker_start_ticks=100 WHERE id=?", (run_id,),
        )
        assert kb.complete_task(conn, task_id, result="ok") is True
        results = iter(
            [
                {
                    "group_termination_attempted": True,
                    "group_terminated": False,
                },
                {
                    "group_termination_attempted": True,
                    "group_terminated": True,
                },
            ]
        )
        monkeypatch.setattr(
            kb, "_terminate_worker_process_group",
            lambda *args, **kwargs: next(results),
        )

        assert kb.cleanup_terminal_worker_groups(conn) == []
        assert conn.execute(
            "SELECT process_cleaned_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()[0] is None
        assert kb.cleanup_terminal_worker_groups(conn) == [run_id]
    finally:
        conn.close()


def test_unknown_dead_leader_membership_stays_pending_and_fenced(
    tmp_path, monkeypatch
):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="unknown ownership", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE task_runs SET worker_pid=4601, worker_pgid=4601, "
            "worker_sid=4601, worker_start_ticks=100 WHERE id=?", (run_id,),
        )
        assert kb.complete_task(conn, task_id, result="ok") is True
        monkeypatch.setattr(kb.os, "getpgrp", lambda: 999)
        monkeypatch.setattr(kb, "_linux_process_identity", lambda pid: None)
        monkeypatch.setattr(kb, "_linux_group_members", lambda pgid: None)

        assert kb.cleanup_terminal_worker_groups(conn) == []
        assert conn.execute(
            "SELECT process_cleaned_at FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()[0] is None
        assert kb.check_respawn_guard(conn, task_id) == "worker_cleanup_pending"
    finally:
        conn.close()
