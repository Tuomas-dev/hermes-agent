"""The kanban stop-guard must also cover the codex app-server runtime.

Kanban workers have to end with ``kanban_complete`` / ``kanban_block``. Models
routinely narrate the finish instead ("Implemented task X. Key changes: ...")
and stop cleanly, which the dispatcher records as a ``protocol_violation``
(rc=0 with the task still ``running``) and retries until the budget is gone —
even though the work itself succeeded.

``conversation_loop`` guards against exactly this, but the guard lives *inside*
the tool-calling while-loop that the codex runtime bypasses via the early
return in ``run_conversation``. So codex-profile workers had no guard at all:
observed on a live board as repeated "crashed (worker exited)" notifications on
tasks whose work had actually completed.

These tests lock in that the codex path re-runs the turn with the shared
synthetic nudge, and — just as importantly — that it does NOT fire on turns
where nudging would be wrong or unsafe.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn


def _make_turn(**overrides):
    values = dict(
        interrupted=False,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[{"role": "assistant", "content": "narrated finish"}],
        tool_iterations=0,
        final_text="narrated finish",
        should_retire=False,
        kanban_terminal=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_agent(*turns):
    agent = MagicMock()
    agent._codex_session = MagicMock()
    if turns:
        agent._codex_session.run_turn.side_effect = list(turns)
    else:
        agent._codex_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-codex"
    return agent


def _run(agent, messages=None):
    return run_codex_app_server_turn(
        agent,
        user_message="work kanban task t_abc123",
        original_user_message="work kanban task t_abc123",
        messages=messages if messages is not None else [
            {"role": "user", "content": "work kanban task t_abc123"}
        ],
        effective_task_id="task-1",
    )


@pytest.fixture
def kanban_worker(monkeypatch):
    """Env of a dispatcher-spawned kanban worker."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)


class TestNudgeFires:
    def test_narrated_stop_is_nudged(self, kanban_worker):
        """The exact production failure: clean stop, no terminal board call."""
        agent = _make_agent(
            _make_turn(),
            _make_turn(kanban_terminal="kanban_complete", final_text=""),
        )
        _run(agent)

        assert agent._codex_session.run_turn.call_count == 2
        second = agent._codex_session.run_turn.call_args_list[1]
        assert "kanban_complete" in second.kwargs["user_input"]

    def test_nudge_is_spliced_into_the_transcript(self, kanban_worker):
        """The synthetic prompt must be recorded, and marked as synthetic so
        curator/compression can tell it apart from a real user turn."""
        agent = _make_agent(
            _make_turn(),
            _make_turn(kanban_terminal="kanban_complete", final_text=""),
        )
        messages = [{"role": "user", "content": "work kanban task t_abc123"}]
        _run(agent, messages)

        synthetic = [m for m in messages if m.get("_kanban_stop_synthetic")]
        assert len(synthetic) == 1
        assert synthetic[0]["role"] == "user"
        assert "kanban_complete" in synthetic[0]["content"]

    def test_nudge_budget_is_bounded(self, kanban_worker):
        """A model that never complies must not loop forever."""
        agent = _make_agent(*[_make_turn() for _ in range(6)])
        _run(agent)

        # 1 real turn + the shared 2-attempt nudge budget.
        assert agent._codex_session.run_turn.call_count == 3


class TestNudgeDoesNotFire:
    def test_terminal_callback_is_not_nudged(self, kanban_worker):
        agent = _make_agent(
            _make_turn(kanban_terminal="kanban_complete", final_text="")
        )
        _run(agent)
        assert agent._codex_session.run_turn.call_count == 1

    def test_terminal_tool_in_transcript_is_not_nudged(self, kanban_worker):
        """Guard keys off the conversation too, not just the fence flag."""
        agent = _make_agent(
            _make_turn(
                projected_messages=[
                    {"role": "tool", "name": "kanban_complete", "content": "{}"}
                ]
            )
        )
        _run(agent)
        assert agent._codex_session.run_turn.call_count == 1

    def test_non_kanban_session_is_not_nudged(self, monkeypatch):
        """Ordinary codex chat must be untouched by this guard."""
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        agent = _make_agent(_make_turn())
        _run(agent)
        assert agent._codex_session.run_turn.call_count == 1

    def test_opt_out_is_respected(self, kanban_worker, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
        agent = _make_agent(_make_turn())
        _run(agent)
        assert agent._codex_session.run_turn.call_count == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            {"error": "codex exploded"},
            {"interrupted": True},
            {"should_retire": True},
        ],
        ids=["errored", "interrupted", "retiring"],
    )
    def test_unhealthy_turns_are_left_alone(self, kanban_worker, overrides):
        """Re-running a dead or interrupted session would ride a broken
        process (or fight a Ctrl+C), so the guard must stay out of the way."""
        agent = _make_agent(_make_turn(**overrides))
        # A retiring turn clears agent._codex_session, so hold the reference.
        session = agent._codex_session
        _run(agent)
        assert session.run_turn.call_count == 1

    def test_nudge_turn_failure_does_not_crash_the_run(self, kanban_worker):
        """If the follow-up turn raises, the original result still returns."""
        agent = _make_agent()
        agent._codex_session.run_turn.side_effect = [
            _make_turn(),
            RuntimeError("session died mid-nudge"),
        ]
        result = _run(agent)
        assert result["completed"] is True
