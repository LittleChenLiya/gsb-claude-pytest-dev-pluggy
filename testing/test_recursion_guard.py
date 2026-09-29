"""
Tests for the optional hook recursion guard.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from pluggy import HookimplMarker
from pluggy import HookRecursionError
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy._recursion import MAX_CHAIN_ENTRIES


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Spec:
    @hookspec
    def hook_a(self, arg: int) -> int: ...

    @hookspec
    def hook_b(self, arg: int) -> int: ...


@pytest.fixture
def rpm() -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Spec)
    return pm


def make_loop_plugins(pm: PluginManager, reentrant: dict[str, bool]):
    """hook_a calls hook_b, hook_b calls hook_a: a re-entry loop."""

    class PluginA:
        @hookimpl
        def hook_a(self, arg: int) -> str:
            if reentrant["enabled"]:
                pm.hook.hook_b(arg=arg)
            return "a"

    class PluginB:
        @hookimpl
        def hook_b(self, arg: int) -> str:
            if reentrant["enabled"]:
                pm.hook.hook_a(arg=arg)
            return "b"

    pm.register(PluginA(), "plug_a")
    pm.register(PluginB(), "plug_b")


def test_guard_disabled_by_default_allows_bounded_recursion(rpm: PluginManager) -> None:
    # Without the guard, a (bounded) recursive hook chain works fine.
    class Rec:
        @hookimpl
        def hook_a(self, arg: int) -> int:
            if arg > 0:
                return sum(rpm.hook.hook_a(arg=arg - 1)) + 1
            return 1

    rpm.register(Rec(), "rec")
    assert rpm._recursion_guard is None
    assert rpm.hook.hook_a(arg=5) == [6]


def test_reentry_loop_blocked_and_stack_cleaned(rpm: PluginManager) -> None:
    reentrant = {"enabled": True}
    make_loop_plugins(rpm, reentrant)
    rpm.enable_recursion_guard(no_reentry={"hook_a"})

    with pytest.raises(HookRecursionError) as excinfo:
        rpm.hook.hook_a(arg=1)

    chain = excinfo.value.call_chain
    # The chain is a closed loop: hook_a -> hook_b -> hook_a.
    assert [frame.hook_name for frame in chain] == ["hook_a", "hook_b", "hook_a"]
    assert [frame.depth for frame in chain] == [1, 2, 3]
    assert chain[0].plugin_names == ("plug_a",)
    assert chain[1].plugin_names == ("plug_b",)
    assert "hook_a" in str(excinfo.value)
    assert "re-entered" in str(excinfo.value)

    # No stale stack remains after the exception was handled.
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()

    # A normal call afterwards works and leaves no residue either.
    reentrant["enabled"] = False
    assert rpm.hook.hook_a(arg=1) == ["a"]
    assert rpm._recursion_guard.current_chain() == ()


def test_guard_undo_function_disables(rpm: PluginManager) -> None:
    reentrant = {"enabled": True}
    make_loop_plugins(rpm, reentrant)
    undo = rpm.enable_recursion_guard(no_reentry={"hook_a"})
    undo()
    assert rpm._recursion_guard is None
    # Recursion is unguarded again (bounded here by the flag).
    reentrant["enabled"] = False
    assert rpm.hook.hook_a(arg=1) == ["a"]


def test_invalid_max_depth_rejected(rpm: PluginManager) -> None:
    with pytest.raises(ValueError, match="max_depth"):
        rpm.enable_recursion_guard(max_depth=0)


def test_max_depth_exact_limit(rpm: PluginManager) -> None:
    # hook_a and hook_b call each other while counting down.
    class PluginA:
        @hookimpl
        def hook_a(self, arg: int) -> int:
            if arg > 0:
                rpm.hook.hook_b(arg=arg - 1)
            return arg

    class PluginB:
        @hookimpl
        def hook_b(self, arg: int) -> int:
            if arg > 0:
                rpm.hook.hook_a(arg=arg - 1)
            return arg

    rpm.register(PluginA(), "plug_a")
    rpm.register(PluginB(), "plug_b")
    rpm.enable_recursion_guard(max_depth=4)

    # A chain of exactly max_depth=4 calls (a, b, a, b) is allowed.
    assert rpm.hook.hook_a(arg=3) == [3]
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()

    # One more nesting level exceeds the limit.
    with pytest.raises(HookRecursionError, match="maximum depth of 4") as excinfo:
        rpm.hook.hook_a(arg=4)
    assert excinfo.value.call_chain[-1].depth == 5
    # The stack is cleaned up after the depth error as well.
    assert rpm._recursion_guard.current_chain() == ()


def test_wrappers_unwind_in_entry_order_on_recursion_error(
    rpm: PluginManager,
) -> None:
    events: list[str] = []
    reentrant = {"enabled": True}
    make_loop_plugins(rpm, reentrant)

    class Wrapper1:
        @hookimpl(wrapper=True)
        def hook_a(self, arg: int):
            events.append("enter-w1")
            try:
                yield
            finally:
                events.append("exit-w1")

    class Wrapper2:
        @hookimpl(wrapper=True)
        def hook_a(self, arg: int):
            events.append("enter-w2")
            try:
                yield
            finally:
                events.append("exit-w2")

    rpm.register(Wrapper1(), "wrap1")
    rpm.register(Wrapper2(), "wrap2")
    rpm.enable_recursion_guard(no_reentry={"hook_a"})

    with pytest.raises(HookRecursionError):
        rpm.hook.hook_a(arg=1)

    # Both wrappers were entered and unwound, innermost first (LIFO).
    assert len(events) == 4
    assert {events[0], events[1]} == {"enter-w1", "enter-w2"}
    assert events[2] == events[1].replace("enter", "exit")
    assert events[3] == events[0].replace("enter", "exit")
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()


def test_wrapper_teardown_exception_still_cleans_stack(rpm: PluginManager) -> None:
    reentrant = {"enabled": True}
    make_loop_plugins(rpm, reentrant)

    class BadWrapper:
        @hookimpl(wrapper=True)
        def hook_a(self, arg: int):
            try:
                yield
            finally:
                raise RuntimeError("teardown blew up")

    rpm.register(BadWrapper(), "bad_wrap")
    rpm.enable_recursion_guard(no_reentry={"hook_a"})

    # The wrapper teardown exception replaces the recursion error.
    with pytest.raises(RuntimeError, match="teardown blew up"):
        rpm.hook.hook_a(arg=1)
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()


def test_firstresult_early_exit_cleans_stack(rpm: PluginManager) -> None:
    class FirstSpec:
        @hookspec(firstresult=True)
        def hook_fr(self, arg: int) -> int: ...

    rpm.add_hookspecs(FirstSpec)

    class P1:
        @hookimpl
        def hook_fr(self, arg: int) -> int:
            return 42

    class P2:
        @hookimpl(trylast=True)
        def hook_fr(self, arg: int) -> int:
            return 1

    rpm.register(P1(), "p1")
    rpm.register(P2(), "p2")
    rpm.enable_recursion_guard()

    assert rpm.hook.hook_fr(arg=1) == 42
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()


def test_historic_call_and_replay_clean_stack(rpm: PluginManager) -> None:
    class HistSpec:
        @hookspec(historic=True)
        def hook_h(self, arg: int) -> int: ...

    rpm.add_hookspecs(HistSpec)
    rpm.enable_recursion_guard()

    seen: list[int] = []

    class P1:
        @hookimpl
        def hook_h(self, arg: int) -> None:
            seen.append(arg)

    rpm.register(P1(), "p1")
    rpm.hook.hook_h.call_historic(kwargs=dict(arg=1))
    assert seen == [1]
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()

    # Registering a new plugin replays the historic call through the guard.
    class P2:
        @hookimpl
        def hook_h(self, arg: int) -> None:
            seen.append(arg * 10)

    rpm.register(P2(), "p2")
    assert seen == [1, 10]
    assert rpm._recursion_guard.current_chain() == ()


def test_concurrent_threads_have_independent_chains(rpm: PluginManager) -> None:
    barrier = threading.Barrier(2)
    depths: list[int] = []

    class P:
        @hookimpl
        def hook_a(self, arg: int) -> int:
            guard = rpm._recursion_guard
            assert guard is not None
            depths.append(len(guard.current_chain()))
            # Both threads sit inside hook_a at the same time; a shared
            # (non-context-local) stack would misjudge this as recursion.
            barrier.wait(timeout=10)
            return arg

    rpm.register(P(), "p")
    rpm.enable_recursion_guard(no_reentry={"hook_a"})

    results: list[list[int]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(rpm.hook.hook_a(arg=1))
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert results == [[1], [1]]
    # Each thread only ever saw its own chain of depth 1.
    assert sorted(depths) == [1, 1]
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()


def test_asyncio_tasks_have_independent_chains(rpm: PluginManager) -> None:
    depths: list[int] = []

    class P:
        @hookimpl
        def hook_a(self, arg: int) -> int:
            guard = rpm._recursion_guard
            assert guard is not None
            depths.append(len(guard.current_chain()))
            return arg

    rpm.register(P(), "p")
    rpm.enable_recursion_guard(no_reentry={"hook_a"})

    async def main() -> list[list[int]]:
        async def worker(arg: int) -> list[int]:
            await asyncio.sleep(0)
            return rpm.hook.hook_a(arg=arg)

        return await asyncio.gather(worker(1), worker(2))

    assert asyncio.run(main()) == [[1], [2]]
    assert sorted(depths) == [1, 1]
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()


def test_long_chain_trimmed_but_keeps_first_and_last(rpm: PluginManager) -> None:
    # hook_a -> hook_b -> hook_b -> ... -> hook_a: the re-entry of hook_a
    # is only detected deep in the chain, beyond the chain entry cap.
    depth_arg = MAX_CHAIN_ENTRIES * 2

    class PluginA:
        @hookimpl
        def hook_a(self, arg: int) -> int:
            if arg > 0:
                rpm.hook.hook_b(arg=arg)
            return arg

    class PluginB:
        @hookimpl
        def hook_b(self, arg: int) -> int:
            if arg > 0:
                rpm.hook.hook_b(arg=arg - 1)
            else:
                rpm.hook.hook_a(arg=0)
            return arg

    rpm.register(PluginA(), "plug_a")
    rpm.register(PluginB(), "plug_b")
    rpm.enable_recursion_guard(max_depth=100, no_reentry={"hook_a"})

    with pytest.raises(HookRecursionError) as excinfo:
        rpm.hook.hook_a(arg=depth_arg)

    chain = excinfo.value.call_chain
    assert len(chain) <= MAX_CHAIN_ENTRIES
    # The original re-entry point (the outermost hook_a) is preserved.
    assert chain[0].hook_name == "hook_a"
    assert chain[0].depth == 1
    # The last triggering frame is preserved.
    assert chain[-1].hook_name == "hook_a"
    assert chain[-1].depth == depth_arg + 3
    # Depths are strictly increasing and the message marks the gap.
    depths = [frame.depth for frame in chain]
    assert depths == sorted(depths)
    assert len(set(depths)) == len(depths)
    assert "..." in str(excinfo.value)
    assert rpm._recursion_guard is not None
    assert rpm._recursion_guard.current_chain() == ()


def test_print_legal_nesting_and_blocked_ring(
    rpm: PluginManager, capsys: pytest.CaptureFixture[str]
) -> None:
    started = time.monotonic()
    reentrant = {"enabled": True}
    make_loop_plugins(rpm, reentrant)
    rpm.enable_recursion_guard(no_reentry={"hook_a"})

    # A legal nested call: hook_a -> hook_b without re-entering hook_a.
    reentrant["enabled"] = False

    class Probe:
        @hookimpl(wrapper=True)
        def hook_b(self, arg: int):
            guard = rpm._recursion_guard
            assert guard is not None
            print("legal nesting:", " -> ".join(f._format() for f in guard.current_chain()))
            return (yield)

    rpm.register(Probe(), "probe")
    assert rpm.hook.hook_a(arg=1) == ["a"]

    # A blocked call ring: hook_a -> hook_b -> hook_a.
    reentrant["enabled"] = True
    with pytest.raises(HookRecursionError) as excinfo:
        rpm.hook.hook_a(arg=1)
    print("blocked ring:", " -> ".join(f._format() for f in excinfo.value.call_chain))

    out = capsys.readouterr().out
    assert "legal nesting: #1 hook_a (plug_a) -> #2 hook_b (plug_b, probe)" in out
    assert "blocked ring: #1 hook_a (plug_a) -> #2 hook_b (plug_b, probe) -> #3 hook_a (plug_a)" in out
    # Echo to the real stdout so the chains are visible with pytest -s.
    with capsys.disabled():
        print(out, end="")
    # Well within the 90 second budget.
    assert time.monotonic() - started < 90
