"""Tests for bounded historic call history windows and strict registration."""

from __future__ import annotations

import pytest

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy import PluginValidationError


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Hooks:
    @hookspec(historic=True)
    def he_method1(self, arg): ...


@pytest.fixture
def pm() -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Hooks)
    return pm


def call_n(pm: PluginManager, n: int, **kwargs) -> None:
    """Make ``n`` historic calls with arg=1..n."""
    for i in range(1, n + 1):
        pm.hook.he_method1.call_historic(kwargs={"arg": i}, **kwargs)


class Recorder:
    def __init__(self, out: list) -> None:
        self.out = out

    @hookimpl
    def he_method1(self, arg):
        self.out.append(arg)


def test_window_replays_only_retained_records(pm: PluginManager) -> None:
    """Capacity 2, three historic calls: a new plugin only receives
    records 2 and 3, in order, and the dropped range is reported."""
    pm.hook.he_method1.set_history_limit(2)
    call_n(pm, 3)

    assert pm.hook.he_method1.history_dropped_range == (1, 1)
    assert not pm.hook.he_method1.has_complete_history()

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == [2, 3]


def test_strict_registration_refused_on_gap(pm: PluginManager) -> None:
    """Strict registration requiring the complete history refuses up front
    when records were dropped; the plugin is never called nor registered."""
    pm.hook.he_method1.set_history_limit(2)
    call_n(pm, 3)

    out: list[int] = []
    plugin = Recorder(out)
    with pytest.raises(PluginValidationError) as excinfo:
        pm.register(plugin, require_complete_history=True)
    assert "1..1" in str(excinfo.value)
    # The plugin was rejected before any hookimpl was called.
    assert out == []
    assert not pm.is_registered(plugin)


def test_strict_registration_ok_without_gap(pm: PluginManager) -> None:
    """Strict registration succeeds when the history is complete."""
    pm.hook.he_method1.set_history_limit(2)
    call_n(pm, 2)
    assert pm.hook.he_method1.has_complete_history()
    assert pm.hook.he_method1.history_dropped_range is None

    out: list[int] = []
    pm.register(Recorder(out), require_complete_history=True)
    assert out == [1, 2]


def test_strict_registration_ok_unbounded(pm: PluginManager) -> None:
    """Without a history limit the default behavior is unchanged."""
    call_n(pm, 3)
    out: list[int] = []
    pm.register(Recorder(out), require_complete_history=True)
    assert out == [1, 2, 3]


def test_snapshot_immune_to_kwargs_mutation(pm: PluginManager) -> None:
    """With an explicit snapshot function, mutating the original kwargs
    dict after the call does not change the replayed values."""
    pm.hook.he_method1.set_history_limit(2)
    kwargs1 = {"arg": 1}
    kwargs2 = {"arg": 2}
    pm.hook.he_method1.call_historic(kwargs=kwargs1, snapshot=dict)
    pm.hook.he_method1.call_historic(kwargs=kwargs2, snapshot=dict)
    kwargs1["arg"] = 100
    kwargs2["arg"] = 200

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == [1, 2]


def test_no_snapshot_keeps_reference_default(pm: PluginManager) -> None:
    """Without a snapshot function the old default behavior is kept: the
    kwargs mapping itself is stored, so later mutations are replayed."""
    kwargs1 = {"arg": 1}
    pm.hook.he_method1.call_historic(kwargs=kwargs1)
    kwargs1["arg"] = 100

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == [100]


def test_trim_does_not_recall_result_callbacks(pm: PluginManager) -> None:
    """Trimming a record never re-invokes its result_callback."""
    pm.hook.he_method1.set_history_limit(2)
    calls = {1: 0, 2: 0, 3: 0}

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            return arg * 10

    pm.register(Plugin1())
    for i in (1, 2, 3):
        pm.hook.he_method1.call_historic(
            result_callback=lambda res, i=i: calls.__setitem__(i, calls[i] + 1),
            kwargs={"arg": i},
        )
    # Each callback ran exactly once, for its own call; the trimming of
    # record 1 during the third call did not add any callback invocations.
    assert calls == {1: 1, 2: 1, 3: 1}

    # Replaying the retained window to a new plugin only triggers the
    # callbacks of the retained records 2 and 3.
    pm.register(Plugin1())
    assert calls == {1: 1, 2: 2, 3: 2}


def test_nested_registration_replays_window(pm: PluginManager) -> None:
    """A plugin registered from within a history replay (nested
    registration) also only receives the retained window."""
    pm.hook.he_method1.set_history_limit(2)
    call_n(pm, 3)

    out_a: list[int] = []
    out_b: list[int] = []

    class PluginB:
        @hookimpl
        def he_method1(self, arg):
            out_b.append(arg)

    plugin_b = PluginB()

    class PluginA:
        @hookimpl
        def he_method1(self, arg):
            out_a.append(arg)
            if not pm.is_registered(plugin_b):
                pm.register(plugin_b)

    pm.register(PluginA())
    assert out_a == [2, 3]
    assert out_b == [2, 3]


def test_limit_zero_retains_nothing(pm: PluginManager) -> None:
    pm.hook.he_method1.set_history_limit(0)
    call_n(pm, 3)
    assert pm.hook.he_method1.history_dropped_range == (1, 3)

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == []

    with pytest.raises(PluginValidationError, match=r"1\.\.3"):
        pm.register(Recorder([]), require_complete_history=True)


def test_limit_boundary_no_trim_at_capacity(pm: PluginManager) -> None:
    """Exactly ``limit`` calls do not trim anything."""
    pm.hook.he_method1.set_history_limit(3)
    call_n(pm, 3)
    assert pm.hook.he_method1.history_dropped_range is None
    assert pm.hook.he_method1.has_complete_history()

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == [1, 2, 3]

    # The next call trims exactly the oldest record.
    pm.hook.he_method1.call_historic(kwargs={"arg": 4})
    assert pm.hook.he_method1.history_dropped_range == (1, 1)
    assert out == [1, 2, 3, 4]

    out2: list[int] = []
    pm.register(Recorder(out2))
    assert out2 == [2, 3, 4]


def test_set_limit_trims_retroactively(pm: PluginManager) -> None:
    call_n(pm, 3)
    pm.hook.he_method1.set_history_limit(1)
    assert pm.hook.he_method1.history_dropped_range == (1, 2)

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == [3]


def test_set_limit_none_restores_unbounded(pm: PluginManager) -> None:
    pm.hook.he_method1.set_history_limit(1)
    call_n(pm, 2)
    assert pm.hook.he_method1.history_dropped_range == (1, 1)
    pm.hook.he_method1.set_history_limit(None)
    assert pm.hook.he_method1.history_limit is None
    pm.hook.he_method1.call_historic(kwargs={"arg": 3})
    # No further trimming, but previously dropped records stay dropped.
    assert pm.hook.he_method1.history_dropped_range == (1, 1)

    out: list[int] = []
    pm.register(Recorder(out))
    assert out == [2, 3]


def test_set_history_limit_validation(pm: PluginManager) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pm.hook.he_method1.set_history_limit(-1)

    class PlainHooks:
        @hookspec
        def he_method2(self, arg): ...

    pm.add_hookspecs(PlainHooks)
    with pytest.raises(ValueError, match="non-historic"):
        pm.hook.he_method2.set_history_limit(2)


def test_subset_hook_caller_shares_window(pm: PluginManager) -> None:
    """A subset hook caller shares the history window with its origin."""
    out: list[int] = []

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            pass

    plugin1 = Plugin1()
    pm.register(plugin1)
    shc = pm.subset_hook_caller("he_method1", remove_plugins=[plugin1])
    shc.set_history_limit(2)
    assert pm.hook.he_method1.history_limit == 2
    call_n(pm, 3)
    assert pm.hook.he_method1.history_dropped_range == (1, 1)

    pm.register(Recorder(out))
    assert out == [2, 3]
