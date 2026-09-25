"""Tests for the optional hook recursion protection."""

from __future__ import annotations

import asyncio
import threading

import pytest

from pluggy import HookimplMarker
from pluggy import HookRecursionError
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy._recursion import _call_stack


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Spec:
    @hookspec
    def hook_a(self):
        pass

    @hookspec
    def hook_b(self):
        pass

    @hookspec(firstresult=True)
    def hook_fr(self):
        pass

    @hookspec(historic=True)
    def hook_hist(self):
        pass


def make_pm() -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Spec)
    return pm


def test_protection_disabled_by_default() -> None:
    """Without opt-in, re-entrant calls are allowed (as before)."""
    pm = make_pm()
    count = 0

    class Plugin:
        @hookimpl
        def hook_a(self):
            nonlocal count
            count += 1
            if count < 3:
                pm.hook.hook_a()

    pm.register(Plugin())
    pm.hook.hook_a()
    assert count == 3
    assert _call_stack.get() == ()


def test_reentry_loop_detected_and_stack_cleaned() -> None:
    """hook_a -> hook_b -> hook_a is rejected as a closed loop, and after
    handling the error a fresh call starts from a clean stack."""
    pm = make_pm()
    recurse = True

    class PluginA:
        @hookimpl
        def hook_a(self):
            pm.hook.hook_b()

    class PluginB:
        @hookimpl
        def hook_b(self):
            if recurse:
                pm.hook.hook_a()

    pm.register(PluginA(), name="plugin_a")
    pm.register(PluginB(), name="plugin_b")
    pm.enable_recursion_protection(no_reentry={"hook_a"})

    with pytest.raises(HookRecursionError) as excinfo:
        pm.hook.hook_a()

    err = excinfo.value
    assert [frame.hook_name for frame in err.call_ring] == [
        "hook_a",
        "hook_b",
        "hook_a",
    ]
    assert [frame.depth for frame in err.call_ring] == [1, 2, 3]
    assert err.call_ring[0].plugin_names == ("plugin_a",)
    assert "plugin_b" in err.call_ring[1].plugin_names
    assert "non-reentrant" in str(err)
    assert "hook_a" in str(err)
    # The stack is cleaned up once the error is handled.
    assert _call_stack.get() == ()

    # A fresh call after the error does not see any leftover stack.
    recurse = False
    pm.hook.hook_a()
    assert _call_stack.get() == ()


def test_direct_self_reentry_detected() -> None:
    pm = make_pm()

    class Plugin:
        @hookimpl
        def hook_a(self):
            pm.hook.hook_a()

    pm.register(Plugin())
    pm.enable_recursion_protection(no_reentry={"hook_a"})
    with pytest.raises(HookRecursionError) as excinfo:
        pm.hook.hook_a()
    assert [frame.hook_name for frame in excinfo.value.call_ring] == [
        "hook_a",
        "hook_a",
    ]
    assert _call_stack.get() == ()


def test_max_depth_exact_boundary() -> None:
    """A chain exactly as deep as max_depth passes; one more level fails."""
    pm = make_pm()
    entered = 0

    class Plugin:
        @hookimpl
        def hook_a(self):
            nonlocal entered
            entered += 1
            pm.hook.hook_a()

    pm.register(Plugin())
    undo = pm.enable_recursion_protection(max_depth=3)
    with pytest.raises(HookRecursionError) as excinfo:
        pm.hook.hook_a()
    assert entered == 3
    assert "maximum depth of 3" in str(excinfo.value)
    assert _call_stack.get() == ()

    # Exactly at the limit is fine.
    undo()
    entered = 0
    limit = 3
    pm2 = make_pm()

    class Plugin2:
        @hookimpl
        def hook_a(self):
            nonlocal entered
            entered += 1
            if entered < limit:
                pm2.hook.hook_a()

    pm2.register(Plugin2())
    pm2.enable_recursion_protection(max_depth=3)
    pm2.hook.hook_a()
    assert entered == 3
    assert _call_stack.get() == ()


def test_enable_validation() -> None:
    pm = make_pm()
    with pytest.raises(ValueError, match="max_depth"):
        pm.enable_recursion_protection(max_depth=0)
    with pytest.raises(ValueError, match="max_ring_entries"):
        pm.enable_recursion_protection(max_ring_entries=1)


def test_undo_restores_unprotected_behavior() -> None:
    pm = make_pm()
    count = 0

    class Plugin:
        @hookimpl
        def hook_a(self):
            nonlocal count
            count += 1
            if count < 3:
                pm.hook.hook_a()

    pm.register(Plugin())
    undo = pm.enable_recursion_protection(no_reentry={"hook_a"})
    undo()
    pm.hook.hook_a()
    assert count == 3


def test_firstresult_early_exit_cleans_stack() -> None:
    """firstresult hooks halt the call loop early; the guard stack must be
    cleaned up exactly like for a full call."""
    pm = make_pm()

    class Plugin1:
        @hookimpl
        def hook_fr(self):
            return "early"

    class Plugin2:
        @hookimpl(tryfirst=True)
        def hook_fr(self):
            return "first"

    pm.register(Plugin1())
    pm.register(Plugin2())
    pm.enable_recursion_protection(max_depth=10, no_reentry={"hook_fr"})
    assert pm.hook.hook_fr() == "first"
    assert _call_stack.get() == ()

    class Plugin3:
        @hookimpl(tryfirst=True)
        def hook_fr(self):
            raise RuntimeError("boom")

    pm.register(Plugin3())
    with pytest.raises(RuntimeError, match="boom"):
        pm.hook.hook_fr()
    assert _call_stack.get() == ()


def test_historic_replay_cleans_stack() -> None:
    """Historic calls and their replay on newly registered plugins use the
    same stack cleanup rules."""
    pm = make_pm()
    seen = []

    class Plugin1:
        @hookimpl
        def hook_hist(self):
            seen.append("p1")

    pm.register(Plugin1())
    pm.enable_recursion_protection(max_depth=5, no_reentry={"hook_hist"})
    pm.hook.hook_hist.call_historic()
    assert seen == ["p1"]
    assert _call_stack.get() == ()

    class Plugin2:
        @hookimpl
        def hook_hist(self):
            seen.append("p2")

    # Registration replays the historic call on the new plugin.
    pm.register(Plugin2())
    assert seen == ["p1", "p2"]
    assert _call_stack.get() == ()


def test_wrapper_teardown_exception_cleans_stack() -> None:
    """An exception raised by a wrapper teardown must not leave stale stack
    entries behind."""
    pm = make_pm()

    class Plugin:
        @hookimpl
        def hook_a(self):
            return None

    class Wrapper:
        @hookimpl(wrapper=True)
        def hook_a(self):
            yield
            raise RuntimeError("teardown boom")

    pm.register(Plugin())
    pm.register(Wrapper())
    pm.enable_recursion_protection(max_depth=5)
    with pytest.raises(RuntimeError, match="teardown boom"):
        pm.hook.hook_a()
    assert _call_stack.get() == ()


def test_wrappers_unwind_in_entered_order_on_recursion_error() -> None:
    """When a nested call is rejected, the wrappers of the outer calls unwind
    innermost-first, matching the order they were entered."""
    pm = make_pm()
    events = []
    recurse = True

    class Wrapper1:
        @hookimpl(wrapper=True, tryfirst=True)
        def hook_a(self):
            events.append("enter w1")
            try:
                yield
            except HookRecursionError:
                events.append("exit w1")
                raise

    class Wrapper2:
        @hookimpl(wrapper=True)
        def hook_a(self):
            events.append("enter w2")
            try:
                yield
            except HookRecursionError:
                events.append("exit w2")
                raise

    class PluginA:
        @hookimpl
        def hook_a(self):
            pm.hook.hook_b()

    class PluginB:
        @hookimpl
        def hook_b(self):
            if recurse:
                pm.hook.hook_a()

    pm.register(Wrapper1())
    pm.register(Wrapper2())
    pm.register(PluginA())
    pm.register(PluginB())
    pm.enable_recursion_protection(no_reentry={"hook_a"})

    with pytest.raises(HookRecursionError):
        pm.hook.hook_a()
    assert events == ["enter w1", "enter w2", "exit w2", "exit w1"]
    assert _call_stack.get() == ()


def test_concurrent_threads_have_independent_stacks() -> None:
    """The same hook called concurrently in two threads is two independent
    call chains, not recursion."""
    pm = make_pm()
    entered = [threading.Event(), threading.Event()]
    index_of = {}

    class Plugin:
        @hookimpl
        def hook_a(self):
            i = index_of[threading.current_thread().ident]
            entered[i].set()
            # Wait until the other thread is inside hook_a as well.
            assert entered[1 - i].wait(timeout=10)

    pm.register(Plugin())
    pm.enable_recursion_protection(no_reentry={"hook_a"})

    errors = []

    def worker(i: int) -> None:
        index_of[threading.current_thread().ident] = i
        try:
            pm.hook.hook_a()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    assert all(event.is_set() for event in entered)
    assert _call_stack.get() == ()


def test_asyncio_tasks_have_independent_stacks() -> None:
    """Asyncio tasks each see their own call chain."""
    pm = make_pm()
    depths = []

    class Plugin:
        @hookimpl
        def hook_a(self):
            depths.append(len(_call_stack.get()))

    pm.register(Plugin())
    pm.enable_recursion_protection(no_reentry={"hook_a"})

    async def worker() -> None:
        pm.hook.hook_a()

    async def main() -> None:
        await asyncio.gather(worker(), worker())

    asyncio.run(main())
    # Each task observed only its own hook_a call on the stack.
    assert depths == [1, 1]
    assert _call_stack.get() == ()


def test_call_ring_trimmed_but_keeps_endpoints() -> None:
    """Long call rings are trimmed to max_ring_entries, but always keep the
    first re-entry point and the final triggering call."""
    pm = PluginManager("example")
    names = ["hook_a"] + [f"hook_{i}" for i in range(30)]

    def make_impl(i: int):
        next_name = names[i + 1] if i + 1 < len(names) else "hook_a"

        @hookimpl(specname=names[i])
        def impl(self):
            getattr(pm.hook, next_name)()

        return impl

    plugin_cls = type("Chain", (), {f"impl_{i}": make_impl(i) for i in range(len(names))})
    pm.register(plugin_cls(), name="chain")
    pm.enable_recursion_protection(no_reentry={"hook_a"}, max_ring_entries=6)

    with pytest.raises(HookRecursionError) as excinfo:
        pm.hook.hook_a()

    err = excinfo.value
    ring = err.call_ring
    assert len(ring) == 6
    # The first re-entry point (the outer hook_a call) is preserved...
    assert ring[0].hook_name == "hook_a"
    assert ring[0].depth == 1
    # ...and so is the final triggering call.
    assert ring[-1].hook_name == "hook_a"
    assert ring[-1].depth == len(names) + 1
    # Depths are strictly increasing and reflect the original chain.
    depths = [frame.depth for frame in ring]
    assert depths == sorted(depths)
    assert len(set(depths)) == len(depths)
    # The rendered message marks the trimmed gap.
    assert "..." in str(err)
    assert _call_stack.get() == ()


def test_max_depth_ring_keeps_first_and_last_frames() -> None:
    pm = make_pm()
    count = 0

    class Plugin:
        @hookimpl
        def hook_a(self):
            nonlocal count
            count += 1
            pm.hook.hook_a()

    pm.register(Plugin())
    pm.enable_recursion_protection(max_depth=50, max_ring_entries=5)
    with pytest.raises(HookRecursionError) as excinfo:
        pm.hook.hook_a()
    ring = excinfo.value.call_ring
    assert len(ring) == 5
    assert ring[0].depth == 1
    assert ring[-1].depth == 51
    assert _call_stack.get() == ()


def test_print_legal_nesting_and_blocked_ring(capsys) -> None:
    """Print a legal nested call chain and a blocked call ring."""
    pm = make_pm()
    recurse = True

    class PluginA:
        @hookimpl
        def hook_a(self):
            pm.hook.hook_b()

    class PluginB:
        @hookimpl
        def hook_b(self):
            stack = _call_stack.get()
            print("legal nesting:")
            for frame in stack:
                print(
                    f"  depth={frame.depth} hook={frame.hook_name} "
                    f"plugins={frame.plugin_names}"
                )
            if recurse:
                pm.hook.hook_a()

    pm.register(PluginA(), name="pa")
    pm.register(PluginB(), name="pb")
    pm.enable_recursion_protection(no_reentry={"hook_a"})

    # Legal nesting: hook_a -> hook_b is fine, hook_b is not re-entered.
    recurse = False
    pm.hook.hook_a()

    # Blocked ring: hook_a -> hook_b -> hook_a.
    recurse = True
    with pytest.raises(HookRecursionError) as excinfo:
        pm.hook.hook_a()
    print("blocked call ring:")
    print(excinfo.value)

    out = capsys.readouterr().out
    assert "legal nesting:" in out
    assert "hook=hook_a" in out
    assert "hook=hook_b" in out
    assert "blocked call ring:" in out
    assert "non-reentrant" in out
    assert _call_stack.get() == ()
