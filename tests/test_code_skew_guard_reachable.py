"""The code-skew guard's early return must actually be reachable.

``tests/test_code_skew.py`` and ``tests/test_agent_code_skew.py`` cover the
*detection* helpers only. They all passed while the guard itself raised
``UnboundLocalError: cannot access local variable 'finalize_turn'`` on every
turn, because ``run_conversation`` imported ``finalize_turn`` in a
function-local ``from agent.turn_finalizer import ...`` near the end of the
body. A function-local import binds the name for the *whole* function, so the
guard's early ``return finalize_turn(...)`` at the top of the loop read an
unbound local and blew up — turning "please restart, the source changed" into
"Sorry, I encountered an unexpected error." for every user message.

These tests pin the invariant at the bytecode level: any name the guard's early
return depends on must resolve globally, not as a function local.
"""

import ast
import inspect
import textwrap

from agent import conversation_loop


class TestFinalizeTurnIsNotShadowed:
    def test_finalize_turn_resolves_at_module_scope(self):
        assert callable(getattr(conversation_loop, "finalize_turn", None)), (
            "finalize_turn must be importable at module scope so the "
            "code-skew guard's early return can reach it"
        )

    def test_finalize_turn_is_not_a_local_of_run_conversation(self):
        code = conversation_loop.run_conversation.__code__
        assert "finalize_turn" not in code.co_varnames, (
            "finalize_turn is a local of run_conversation — a function-local "
            "import re-introduced the UnboundLocalError in the code-skew guard"
        )
        assert "finalize_turn" in code.co_names, (
            "finalize_turn should be looked up as a global"
        )

    def test_no_function_local_import_of_turn_finalizer(self):
        src = inspect.getsource(conversation_loop.run_conversation)
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "agent.turn_finalizer":
                raise AssertionError(
                    "run_conversation must not import agent.turn_finalizer "
                    "locally; it shadows the module-level binding the "
                    "code-skew guard returns through"
                )
