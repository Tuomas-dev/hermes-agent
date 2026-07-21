"""A cleanly-exited worker group must resolve, not fence its task forever.

``check_respawn_guard`` defers a task while any ended run still has a
``worker_pgid`` and no ``process_cleaned_at`` ("worker_cleanup_pending").
``cleanup_terminal_worker_groups`` only stamps ``process_cleaned_at`` when
``_terminate_worker_process_group`` reports ``group_terminated``.

On Linux, a worker that exited normally produced neither: the leader was gone
and the owned-group scan was empty, and that combination returned "pending".
Nothing ever revisits it — the pgid cannot reappear — so the task was fenced
permanently and the dispatcher logged "ready queue non-empty for N consecutive
ticks but 0 workers spawned" indefinitely. Observed on a live board with four
runs wedged in exactly this state.

The distinction that makes empty safe: ``_linux_group_members`` returns ``None``
for any unreadable or unparseable /proc entry and skips only vanished-pid
races, so a returned list is a complete snapshot. Empty + dead leader is
verified absence. The portable branch already treated its equivalent
(``portable_state == "absent"``) as terminated.
"""

import sys

import pytest

from hermes_cli import kanban_db as kb


LEADER = 424242
IDENTITY = (LEADER, LEADER, 99_000)  # (pgid, sid, start_ticks)


@pytest.fixture
def linux_only():
    if sys.platform != "linux":
        pytest.skip("covers the Linux /proc branch")


class TestExitedGroupResolves:
    def test_long_dead_group_is_terminated(self, linux_only, monkeypatch):
        """The production stuck-state: worker ran, finished, exited cleanly,
        and the run has been terminal for well past the absence grace."""
        monkeypatch.setattr(kb, "_linux_group_members", lambda _pgid: [])
        monkeypatch.setattr(kb, "_linux_process_identity", lambda _pid: None)

        info = kb._terminate_worker_process_group(
            LEADER, LEADER, LEADER, 99_000, ended_at=1,  # epoch: ancient
        )

        assert info["group_terminated"] is True
        # Nothing was alive, so no signal should have been sent.
        assert info["group_termination_attempted"] is False

    def test_freshly_exited_group_is_still_retried(self, linux_only, monkeypatch):
        """Inside the grace window the empty-snapshot race is still possible,
        so the original fail-closed behaviour is preserved."""
        import time as _time

        monkeypatch.setattr(kb, "_linux_group_members", lambda _pgid: [])
        monkeypatch.setattr(kb, "_linux_process_identity", lambda _pid: None)

        info = kb._terminate_worker_process_group(
            LEADER, LEADER, LEADER, 99_000, ended_at=int(_time.time()),
        )
        assert info["group_terminated"] is False

    def test_absence_grace_is_bounded(self):
        """A permanent fence is worse than the race it guards against, so the
        window must actually elapse rather than being effectively infinite."""
        assert 0 < kb._WORKER_GROUP_ABSENCE_GRACE_SECONDS <= 300

    def test_terminated_group_clears_the_respawn_fence(self, linux_only, monkeypatch, tmp_path):
        """End-to-end: the run gets stamped, so the task is dispatchable again."""
        monkeypatch.setattr(kb, "_linux_group_members", lambda _pgid: [])
        monkeypatch.setattr(kb, "_linux_process_identity", lambda _pid: None)

        conn = kb.connect(tmp_path / "kanban.db")
        try:
            task_id = kb.create_task(conn, title="fenced task", assignee="w")
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO task_runs (task_id, status, started_at, ended_at, "
                    "worker_pid, worker_pgid, worker_sid, worker_start_ticks) "
                    "VALUES (?, 'crashed', 1, 2, ?, ?, ?, ?)",
                    (task_id, LEADER, LEADER, LEADER, 99_000),
                )

            assert kb.check_respawn_guard(conn, task_id) == "worker_cleanup_pending"

            cleaned = kb.cleanup_terminal_worker_groups(conn)
            assert len(cleaned) == 1
            assert kb.check_respawn_guard(conn, task_id) != "worker_cleanup_pending"
        finally:
            conn.close()


class TestUnprovenStatesStillFailClosed:
    def test_unreadable_scan_stays_pending(self, linux_only, monkeypatch):
        """None means the snapshot was not trustworthy — must not resolve."""
        monkeypatch.setattr(kb, "_linux_group_members", lambda _pgid: None)
        monkeypatch.setattr(kb, "_linux_process_identity", lambda _pid: None)

        info = kb._terminate_worker_process_group(
            LEADER, LEADER, LEADER, 99_000, ended_at=1,
        )
        assert info["group_terminated"] is False

    def test_live_leader_with_empty_group_stays_pending(self, linux_only, monkeypatch):
        """Probe and scan disagree — untrustworthy, so retry rather than resolve."""
        monkeypatch.setattr(kb, "_linux_group_members", lambda _pgid: [])
        monkeypatch.setattr(
            kb, "_linux_process_identity", lambda _pid: (LEADER, LEADER, 99_000)
        )

        info = kb._terminate_worker_process_group(
            LEADER, LEADER, LEADER, 99_000, ended_at=1,
        )
        assert info["group_terminated"] is False
