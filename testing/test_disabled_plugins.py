"""Tests for PluginManager.disabled_plugins()."""

import asyncio
import threading

import pytest

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Spec:
    @hookspec
    def myhook(self, arg):
        pass

    @hookspec
    def otherhook(self, arg):
        pass

    @hookspec(firstresult=True)
    def firsthook(self, arg):
        pass


class PluginA:
    def __init__(self, events):
        self.events = events

    @hookimpl
    def myhook(self, arg):
        self.events.append("A")
        return "A"

    @hookimpl
    def otherhook(self, arg):
        self.events.append("A-other")
        return "A"

    @hookimpl
    def firsthook(self, arg):
        self.events.append("A-first")
        return "A"


class PluginB:
    def __init__(self, events):
        self.events = events

    @hookimpl
    def myhook(self, arg):
        self.events.append("B")
        return "B"

    @hookimpl
    def otherhook(self, arg):
        self.events.append("B-other")
        return "B"

    @hookimpl
    def firsthook(self, arg):
        self.events.append("B-first")
        return "B"


class NewStyleWrapper:
    def __init__(self, events):
        self.events = events

    @hookimpl(wrapper=True)
    def myhook(self, arg):
        self.events.append("wrapper-setup")
        try:
            result = yield
        finally:
            self.events.append("wrapper-teardown")
        return result


class OldStyleWrapper:
    def __init__(self, events):
        self.events = events

    @hookimpl(hookwrapper=True)
    def myhook(self, arg):
        self.events.append("oldwrapper-setup")
        outcome = yield
        self.events.append("oldwrapper-teardown")


@pytest.fixture
def events():
    return []


@pytest.fixture
def pm(events):
    pm = PluginManager("example")
    pm.add_hookspecs(Spec)
    pm.register(PluginA(events), name="A")
    pm.register(PluginB(events), name="B")
    pm.register(NewStyleWrapper(events), name="newwrapper")
    pm.register(OldStyleWrapper(events), name="oldwrapper")
    return pm


def test_basic_disable_and_restore(pm, events):
    assert pm.hook.myhook(arg=1) == ["B", "A"]
    events.clear()
    with pm.disabled_plugins({"B"}):
        assert pm.hook.myhook(arg=1) == ["A"]
        assert events == [
            "oldwrapper-setup",
            "wrapper-setup",
            "A",
            "wrapper-teardown",
            "oldwrapper-teardown",
        ]
    # restored after exit
    assert pm.hook.myhook(arg=1) == ["B", "A"]
    # enter and exit once more, restore must work repeatedly
    with pm.disabled_plugins({"A"}):
        assert pm.hook.myhook(arg=1) == ["B"]
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_registry_is_not_modified(pm):
    plugin_b = pm.get_plugin("B")
    with pm.disabled_plugins({"B"}):
        assert pm.is_registered(plugin_b)
        assert pm.get_plugin("B") is plugin_b
        assert pm.has_plugin("B")
    assert pm.get_plugin("B") is plugin_b


def test_unknown_plugin_name_raises_immediately(pm):
    with pytest.raises(ValueError, match="unknown or blocked plugin"):
        with pm.disabled_plugins({"does-not-exist"}):
            pass  # pragma: no cover
    # nothing was disabled by the failed attempt
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_blocked_plugin_name_raises(pm):
    pm.set_blocked("blocked-plugin")
    with pytest.raises(ValueError, match="unknown or blocked plugin"):
        with pm.disabled_plugins({"blocked-plugin"}):
            pass  # pragma: no cover


def test_nested_scopes_form_union(pm):
    with pm.disabled_plugins({"A"}):
        assert pm.hook.myhook(arg=1) == ["B"]
        with pm.disabled_plugins({"B"}):
            assert pm.hook.myhook(arg=1) == []
        # inner scope exited, outer one still active
        assert pm.hook.myhook(arg=1) == ["B"]
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_disable_multiple_names_at_once(pm):
    with pm.disabled_plugins({"A", "B"}):
        assert pm.hook.myhook(arg=1) == []
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_disabled_wrapper_is_skipped_entirely(pm, events):
    with pm.disabled_plugins({"newwrapper"}):
        assert pm.hook.myhook(arg=1) == ["B", "A"]
        assert events == [
            "oldwrapper-setup",
            "B",
            "A",
            "oldwrapper-teardown",
        ]
    events.clear()
    with pm.disabled_plugins({"oldwrapper"}):
        assert pm.hook.myhook(arg=1) == ["B", "A"]
        assert events == [
            "wrapper-setup",
            "B",
            "A",
            "wrapper-teardown",
        ]


def test_enabled_wrapper_teardown_not_skipped(pm, events):
    """A plugin disabling a wrapper mid-call must not skip the teardown
    of a wrapper which was enabled when the call was entered."""

    class MidCallDisabler:
        @hookimpl(tryfirst=True)
        def myhook(self, arg):
            with pm.disabled_plugins({"newwrapper", "oldwrapper"}):
                pass
            return None

    pm.register(MidCallDisabler(), name="midcall")
    assert pm.hook.myhook(arg=1) == ["B", "A"]
    assert events == [
        "oldwrapper-setup",
        "wrapper-setup",
        "B",
        "A",
        "wrapper-teardown",
        "oldwrapper-teardown",
    ]


def test_hook_exception_unwinds_through_enabled_wrappers(pm, events):
    class Raiser:
        @hookimpl(tryfirst=True)
        def myhook(self, arg):
            raise RuntimeError("boom")

    pm.register(Raiser(), name="raiser")
    with pm.disabled_plugins({"B"}):
        with pytest.raises(RuntimeError, match="boom"):
            pm.hook.myhook(arg=1)
    # the enabled wrappers still ran their teardown phase, B never ran
    assert events == [
        "oldwrapper-setup",
        "wrapper-setup",
        "wrapper-teardown",
        "oldwrapper-teardown",
    ]
    pm.unregister(name="raiser")
    # recovery after the exception
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_wrapper_exception_propagates(pm, events):
    class RaisingWrapper:
        @hookimpl(wrapper=True)
        def myhook(self, arg):
            yield
            raise ValueError("wrapper boom")

    pm.register(RaisingWrapper(), name="raisingwrapper")
    with pm.disabled_plugins({"A"}):
        with pytest.raises(ValueError, match="wrapper boom"):
            pm.hook.myhook(arg=1)
    assert "B" in events
    assert "A" not in events
    # recovery after the exception
    events.clear()
    with pytest.raises(ValueError, match="wrapper boom"):
        pm.hook.myhook(arg=1)
    assert events[:3] == ["oldwrapper-setup", "wrapper-setup", "B"]


def test_firstresult_order_preserved(pm, events):
    assert pm.hook.firsthook(arg=1) == "B"
    with pm.disabled_plugins({"B"}):
        assert pm.hook.firsthook(arg=1) == "A"
    assert pm.hook.firsthook(arg=1) == "B"
    assert events == ["B-first", "A-first", "B-first"]


def test_nested_hook_calls_use_their_own_snapshot(pm, events):
    class NestedCaller:
        @hookimpl(tryfirst=True)
        def myhook(self, arg):
            with pm.disabled_plugins({"B"}):
                pm.hook.otherhook(arg=1)
            return None

    pm.register(NestedCaller(), name="nestedcaller")
    pm.hook.myhook(arg=1)
    assert events == [
        "oldwrapper-setup",
        "wrapper-setup",
        # nested call to otherhook with B disabled
        "A-other",
        # remaining myhook impls, B still enabled in this call
        "B",
        "A",
        "wrapper-teardown",
        "oldwrapper-teardown",
    ]


def test_thread_isolation(pm):
    started = threading.Event()
    finish = threading.Event()
    thread_results = {}

    def worker():
        started.wait(timeout=10)
        thread_results["result"] = pm.hook.myhook(arg=1)
        finish.set()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        with pm.disabled_plugins({"B"}):
            assert pm.hook.myhook(arg=1) == ["A"]
            started.set()
            assert finish.wait(timeout=10)
        # the other thread was not affected by the main thread scope
        assert thread_results["result"] == ["B", "A"]
        assert pm.hook.myhook(arg=1) == ["B", "A"]
    finally:
        thread.join(timeout=10)


def test_asyncio_concurrent_contexts(pm):
    async def main():
        entered = asyncio.Event()
        proceed = asyncio.Event()
        results = {}

        async def blocker():
            with pm.disabled_plugins({"B"}):
                results["blocked"] = pm.hook.myhook(arg=1)
                entered.set()
                await proceed.wait()
                results["blocked_again"] = pm.hook.myhook(arg=1)

        async def observer():
            await entered.wait()
            # concurrent with the blocker scope, but in its own task context
            results["observer"] = pm.hook.myhook(arg=1)
            proceed.set()

        await asyncio.gather(blocker(), observer())
        return results

    results = asyncio.run(main())
    assert results["blocked"] == ["A"]
    assert results["observer"] == ["B", "A"]
    assert results["blocked_again"] == ["A"]
    # restored in the main context as well
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_asyncio_child_task_inherits_creation_context(pm):
    async def main():
        start = asyncio.Event()

        async def child():
            await start.wait()
            return pm.hook.myhook(arg=1)

        with pm.disabled_plugins({"B"}):
            # the child task snapshots the context at creation time
            task = asyncio.create_task(child())
        # the parent exited the scope before the child ran; this must not
        # retroactively modify the context the child already obtained
        assert pm.hook.myhook(arg=1) == ["B", "A"]
        start.set()
        return await task

    assert asyncio.run(main()) == ["A"]
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_managers_are_isolated(pm, events):
    other = PluginManager("example")
    other.add_hookspecs(Spec)
    other.register(PluginA(events), name="A")
    other.register(PluginB(events), name="B")
    with other.disabled_plugins({"B"}):
        assert other.hook.myhook(arg=1) == ["A"]
        # a different manager with the same plugin names is unaffected
        assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_empty_names_is_noop(pm):
    with pm.disabled_plugins(set()):
        assert pm.hook.myhook(arg=1) == ["B", "A"]
    with pm.disabled_plugins([]):
        assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_exception_inside_scope_still_restores(pm):
    with pytest.raises(RuntimeError, match="inside"):
        with pm.disabled_plugins({"B"}):
            assert pm.hook.myhook(arg=1) == ["A"]
            raise RuntimeError("inside")
    assert pm.hook.myhook(arg=1) == ["B", "A"]


def test_historic_call_respects_disabled_plugins(events):
    class HistoricSpec:
        @hookspec(historic=True)
        def historichook(self, arg):
            pass

    class HistoricA:
        @hookimpl
        def historichook(self, arg):
            events.append("A")

    class HistoricB:
        @hookimpl
        def historichook(self, arg):
            events.append("B")

    pm = PluginManager("example")
    pm.add_hookspecs(HistoricSpec)
    pm.register(HistoricA(), name="A")
    pm.register(HistoricB(), name="B")
    with pm.disabled_plugins({"B"}):
        pm.hook.historichook.call_historic(kwargs={"arg": 1})
    assert events == ["A"]
    events.clear()
    pm.hook.historichook.call_historic(kwargs={"arg": 1})
    assert events == ["B", "A"]


def test_call_extra_respects_disabled_plugins(pm):
    def extra(arg):
        return "extra"

    with pm.disabled_plugins({"B"}):
        assert pm.hook.myhook.call_extra([extra], {"arg": 1}) == ["extra", "A"]
    assert pm.hook.myhook.call_extra([extra], {"arg": 1}) == ["extra", "B", "A"]


def test_subset_hook_caller_respects_disabled_plugins(pm):
    caller = pm.subset_hook_caller("myhook", [pm.get_plugin("A")])
    with pm.disabled_plugins({"B"}):
        assert caller(arg=1) == []
    assert caller(arg=1) == ["B"]
