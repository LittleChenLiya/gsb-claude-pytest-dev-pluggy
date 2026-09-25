import asyncio
import threading

import pytest

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Hooks:
    @hookspec
    def h1(self, arg: int) -> int:
        raise NotImplementedError

    @hookspec
    def h2(self, arg: int) -> str:
        raise NotImplementedError

    @hookspec(firstresult=True)
    def hfirst(self, arg: int) -> int:
        raise NotImplementedError


@pytest.fixture
def calls() -> "list[str]":
    return []


@pytest.fixture
def pm(calls: "list[str]") -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Hooks)

    class A:
        @hookimpl
        def h1(self, arg: int) -> int:
            calls.append("A")
            return 1

        @hookimpl
        def h2(self, arg: int) -> str:
            calls.append("A2")
            return "a2"

        @hookimpl
        def hfirst(self, arg: int) -> None:
            calls.append("A")
            return None

    class B:
        @hookimpl
        def h1(self, arg: int) -> int:
            calls.append("B")
            return 2

        @hookimpl
        def h2(self, arg: int) -> str:
            calls.append("B2")
            return "b2"

        @hookimpl
        def hfirst(self, arg: int) -> int:
            calls.append("B")
            return 10

    class W:
        @hookimpl(wrapper=True)
        def h1(self, arg: int):
            calls.append("W-setup")
            try:
                result = yield
            finally:
                calls.append("W-teardown")
            return result

    pm.register(A(), "A")
    pm.register(B(), "B")
    pm.register(W(), "W")
    return pm


def collect(pm: PluginManager, calls: "list[str]") -> "list[str]":
    calls.clear()
    pm.hook.h1(arg=0)
    return list(calls)


FULL = ["W-setup", "B", "A", "W-teardown"]


def test_block_and_restore(pm: PluginManager, calls: "list[str]") -> None:
    assert collect(pm, calls) == FULL
    with pm.without_plugins({"B"}):
        assert collect(pm, calls) == ["W-setup", "A", "W-teardown"]
    # Restored after exit.
    assert collect(pm, calls) == FULL
    # Re-enter and restore once more.
    with pm.without_plugins({"A", "B"}):
        assert collect(pm, calls) == ["W-setup", "W-teardown"]
    assert collect(pm, calls) == FULL


def test_blocking_wrapper_plugin_skips_it_entirely(
    pm: PluginManager, calls: "list[str]"
) -> None:
    with pm.without_plugins({"W"}):
        assert collect(pm, calls) == ["B", "A"]
    assert collect(pm, calls) == FULL


def test_registry_not_modified(pm: PluginManager, calls: "list[str]") -> None:
    impl_names_before = [impl.plugin_name for impl in pm.hook.h1.get_hookimpls()]
    with pm.without_plugins({"B"}):
        assert pm.has_plugin("B")
        assert pm.get_plugin("B") is not None
        assert [impl.plugin_name for impl in pm.hook.h1.get_hookimpls()] == (
            impl_names_before
        )
        assert collect(pm, calls) == ["W-setup", "A", "W-teardown"]
    assert pm.has_plugin("B")


def test_nested_scopes_union(pm: PluginManager, calls: "list[str]") -> None:
    with pm.without_plugins({"A"}):
        assert collect(pm, calls) == ["W-setup", "B", "W-teardown"]
        with pm.without_plugins({"B"}):
            assert collect(pm, calls) == ["W-setup", "W-teardown"]
        # Inner scope exited, outer one still applies.
        assert collect(pm, calls) == ["W-setup", "B", "W-teardown"]
    assert collect(pm, calls) == FULL


def test_accepts_single_name(pm: PluginManager, calls: "list[str]") -> None:
    with pm.without_plugins("B"):
        assert collect(pm, calls) == ["W-setup", "A", "W-teardown"]
    assert collect(pm, calls) == FULL


def test_unknown_plugin_name_raises(pm: PluginManager, calls: "list[str]") -> None:
    with pytest.raises(ValueError, match="unknown plugin name.*'nope'"):
        with pm.without_plugins({"nope"}):
            pass  # pragma: no cover
    with pytest.raises(ValueError, match="unknown plugin name"):
        with pm.without_plugins({"A", "nope"}):
            pass  # pragma: no cover
    # Failed enters did not corrupt the state.
    assert collect(pm, calls) == FULL


def test_hook_exception_unwinds_through_enabled_wrappers(
    pm: PluginManager, calls: "list[str]"
) -> None:
    class Boom:
        @hookimpl
        def h1(self, arg: int) -> None:
            calls.append("Boom")
            raise RuntimeError("boom")

    pm.register(Boom(), "boom")
    with pm.without_plugins({"B"}):
        with pytest.raises(RuntimeError, match="boom"):
            pm.hook.h1(arg=0)
    # Boom aborted the impl loop, but the still-enabled wrapper teardown ran.
    assert calls == ["W-setup", "Boom", "W-teardown"]
    pm.unregister(name="boom")
    assert collect(pm, calls) == FULL


def test_wrapper_exception_propagates(pm: PluginManager, calls: "list[str]") -> None:
    class WBad:
        @hookimpl(wrapper=True)
        def h1(self, arg: int):
            calls.append("WBad-setup")
            yield
            calls.append("WBad-teardown")
            raise ValueError("wrapper boom")

    pm.register(WBad(), "wbad")
    with pm.without_plugins({"B"}):
        with pytest.raises(ValueError, match="wrapper boom"):
            pm.hook.h1(arg=0)
    assert calls == ["WBad-setup", "W-setup", "A", "W-teardown", "WBad-teardown"]
    pm.unregister(name="wbad")
    assert collect(pm, calls) == FULL


def test_old_style_hookwrapper_teardown_not_skipped(
    pm: PluginManager, calls: "list[str]"
) -> None:
    class OldW:
        @hookimpl(hookwrapper=True)
        def h1(self, arg: int):
            calls.append("Old-setup")
            yield
            calls.append("Old-teardown")

    pm.register(OldW(), "oldw")
    with pm.without_plugins({"B"}):
        result = pm.hook.h1(arg=0)
    assert result == [1]
    assert calls == ["Old-setup", "W-setup", "A", "W-teardown", "Old-teardown"]
    calls.clear()
    pm.hook.h1(arg=0)
    assert calls == ["Old-setup", "W-setup", "B", "A", "W-teardown", "Old-teardown"]


def test_firstresult_order_and_filtering(
    pm: PluginManager, calls: "list[str]"
) -> None:
    assert pm.hook.hfirst(arg=0) == 10
    assert calls == ["B"]
    with pm.without_plugins({"B"}):
        assert pm.hook.hfirst(arg=0) is None
        assert calls[-1] == "A"
    assert pm.hook.hfirst(arg=0) == 10


def test_nested_hook_call_sees_union_snapshot(
    pm: PluginManager, calls: "list[str]"
) -> None:
    seen = {}

    class Trigger:
        @hookimpl
        def h1(self, arg: int):
            with pm.without_plugins({"B"}):
                seen["inner"] = pm.hook.h2(arg=0)
            seen["after"] = pm.hook.h2(arg=0)
            return None

    pm.register(Trigger(), "trigger")
    with pm.without_plugins({"A"}):
        pm.hook.h1(arg=0)
    # Union of outer {"A"} and inner {"B"} filtered the nested h2 call.
    assert seen["inner"] == []
    # Inner scope exited; the outer scope's snapshot still applies.
    assert seen["after"] == ["b2"]
    # Everything restored afterwards.
    assert pm.hook.h2(arg=0) == ["b2", "a2"]


def test_threads_are_isolated(pm: PluginManager) -> None:
    barrier = threading.Barrier(2)
    outcomes: "dict[str, object]" = {}

    def worker_blocked() -> None:
        try:
            with pm.without_plugins({"B"}):
                barrier.wait(timeout=10)
                outcomes["blocked"] = pm.hook.h1(arg=0)
                # Keep the scope open until the other thread called too.
                barrier.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001
            outcomes["blocked-error"] = exc

    def worker_open() -> None:
        try:
            barrier.wait(timeout=10)
            outcomes["open"] = pm.hook.h1(arg=0)
            barrier.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001
            outcomes["open-error"] = exc

    t_blocked = threading.Thread(target=worker_blocked)
    t_open = threading.Thread(target=worker_open)
    t_blocked.start()
    t_open.start()
    t_blocked.join(timeout=30)
    t_open.join(timeout=30)
    assert not t_blocked.is_alive()
    assert not t_open.is_alive()
    assert outcomes.get("blocked-error") is None
    assert outcomes.get("open-error") is None
    # Both contexts ran concurrently; only the blocking one filtered B.
    assert outcomes["blocked"] == [1]
    assert outcomes["open"] == [2, 1]


def test_asyncio_task_inherits_context_at_creation(pm: PluginManager) -> None:
    async def main() -> None:
        child_ready = asyncio.Event()
        resume = asyncio.Event()
        outcomes: "dict[str, object]" = {}

        async def child() -> None:
            child_ready.set()
            await resume.wait()
            outcomes["child"] = pm.hook.h1(arg=0)

        with pm.without_plugins({"B"}):
            task = asyncio.create_task(child())
            await child_ready.wait()
        # The parent exited the scope, but the child task captured the
        # blocked context at creation and is not retroactively modified.
        outcomes["parent"] = pm.hook.h1(arg=0)
        resume.set()
        await task
        assert outcomes["parent"] == [2, 1]
        assert outcomes["child"] == [1]

    asyncio.run(main())
