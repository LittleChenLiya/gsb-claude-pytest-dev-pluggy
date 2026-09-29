from __future__ import annotations

from collections.abc import Generator

import pytest

from pluggy import HookimplMarker
from pluggy import HookOrderingError
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Spec:
    @hookspec
    def myhook(self) -> None:
        pass


class FirstResultSpec:
    @hookspec(firstresult=True)
    def myhook(self) -> object:
        pass


def register_impl(
    pm: PluginManager,
    name: str,
    calls: list[str],
    **opts: object,
) -> None:
    """Register a plugin called ``name`` with a regular myhook impl."""

    class Plugin:
        @hookimpl(**opts)  # type: ignore[arg-type]
        def myhook(self) -> None:
            calls.append(name)

    pm.register(Plugin(), name=name)


@pytest.fixture
def ordered_pm(pm: PluginManager) -> PluginManager:
    pm.add_hookspecs(Spec)
    return pm


def test_basic_ordering(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)

    # Without ordering, implementations are called in LIFO registration order.
    ordered_pm.hook.myhook()
    assert calls == ["c", "b", "a"]
    calls.clear()

    ordered_pm.set_hookimpl_order("myhook", before={"c": "a", "a": "b"})
    ordered_pm.hook.myhook()
    assert calls == ["c", "a", "b"]


def test_after_option(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)

    ordered_pm.set_hookimpl_order("myhook", after={"b": "a"})
    ordered_pm.hook.myhook()
    assert calls == ["a", "b"]


def test_partial_constraints_keep_default_relative_order(
    ordered_pm: PluginManager,
) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)

    # Only "a" must run before "c"; unconstrained implementations keep
    # their default relative order (c, b, a -> b, a, c).
    ordered_pm.set_hookimpl_order("myhook", before={"a": "c"})
    ordered_pm.hook.myhook()
    assert calls == ["b", "a", "c"]


def test_multiple_targets(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)

    ordered_pm.set_hookimpl_order("myhook", before={"a": ["b", "c"]})
    ordered_pm.hook.myhook()
    assert calls == ["a", "c", "b"]


def test_unknown_hook(ordered_pm: PluginManager) -> None:
    with pytest.raises(HookOrderingError, match="unknown hook 'nosuch'"):
        ordered_pm.set_hookimpl_order("nosuch", before={"a": "b"})


def test_unknown_plugin_keeps_previous_config(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    ordered_pm.set_hookimpl_order("myhook", before={"a": "b"})

    with pytest.raises(HookOrderingError) as excinfo:
        ordered_pm.set_hookimpl_order("myhook", before={"a": "zzz"})
    assert "'zzz'" in str(excinfo.value)
    assert "not registered" in str(excinfo.value)

    # The failed configuration did not replace the previous working one.
    ordered_pm.hook.myhook()
    assert calls == ["a", "b"]


def test_plugin_not_implementing_hook(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)

    class Other:
        pass

    ordered_pm.register(Other(), name="other")
    with pytest.raises(HookOrderingError) as excinfo:
        ordered_pm.set_hookimpl_order("myhook", before={"a": "other"})
    assert "'other'" in str(excinfo.value)
    assert "does not implement" in str(excinfo.value)


def test_cycle_keeps_previous_config(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)
    ordered_pm.set_hookimpl_order("myhook", before={"c": "a", "a": "b"})
    ordered_pm.hook.myhook()
    assert calls == ["c", "a", "b"]
    calls.clear()

    with pytest.raises(HookOrderingError) as excinfo:
        ordered_pm.set_hookimpl_order(
            "myhook", before={"a": "b", "b": "c", "c": "a"}
        )
    message = str(excinfo.value)
    assert "cycle" in message
    for name in ("'a'", "'b'", "'c'"):
        assert name in message

    # The previous working configuration is still in effect.
    ordered_pm.hook.myhook()
    assert calls == ["c", "a", "b"]


def test_self_cycle(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    with pytest.raises(HookOrderingError, match="cycle"):
        ordered_pm.set_hookimpl_order("myhook", before={"a": "a"})


@pytest.mark.parametrize(
    "opts", [{"tryfirst": True}, {"trylast": True}, {"wrapper": True}]
)
def test_cross_tier_conflict(ordered_pm: PluginManager, opts: dict) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)

    if opts.get("wrapper"):

        class WrapperPlugin:
            @hookimpl(wrapper=True)
            def myhook(self) -> Generator[None, None, None]:
                yield

        ordered_pm.register(WrapperPlugin(), name="special")
    else:
        register_impl(ordered_pm, "special", calls, **opts)

    with pytest.raises(HookOrderingError) as excinfo:
        ordered_pm.set_hookimpl_order("myhook", before={"a": "special"})
    message = str(excinfo.value)
    assert "'special'" in message
    assert "tryfirst/trylast or a wrapper" in message

    # The hook still works with the default order.
    ordered_pm.hook.myhook()


def test_tiers_and_wrapper_surround_ordered_regulars(
    ordered_pm: PluginManager,
) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)
    register_impl(ordered_pm, "first", calls, tryfirst=True)
    register_impl(ordered_pm, "last", calls, trylast=True)

    class WrapperPlugin:
        @hookimpl(wrapper=True)
        def myhook(self) -> Generator[None, None, None]:
            calls.append("wrap-before")
            yield
            calls.append("wrap-after")

    ordered_pm.register(WrapperPlugin(), name="wrapper")

    ordered_pm.set_hookimpl_order("myhook", before={"c": "a", "a": "b"})
    ordered_pm.hook.myhook()
    # tryfirst/trylast keep their fixed tiers, the wrapper surrounds the
    # whole call, and only the regular implementations are reordered.
    assert calls == [
        "wrap-before",
        "first",
        "c",
        "a",
        "b",
        "last",
        "wrap-after",
    ]


def test_firstresult_stops_early(pm: PluginManager) -> None:
    pm.add_hookspecs(FirstResultSpec)
    calls: list[str] = []

    def make_plugin(name: str, result: object) -> None:
        class Plugin:
            @hookimpl
            def myhook(self) -> object:
                calls.append(name)
                return result

        pm.register(Plugin(), name=name)

    make_plugin("a", "result-a")
    make_plugin("b", "result-b")
    make_plugin("c", None)

    pm.set_hookimpl_order("myhook", before={"c": "a", "a": "b"})
    # c returns None, so a provides the first result and b is never called.
    assert pm.hook.myhook() == "result-a"
    assert calls == ["c", "a"]


def test_config_change_during_call_takes_effect_next_call(
    ordered_pm: PluginManager,
) -> None:
    calls: list[str] = []

    class PluginC:
        @hookimpl
        def myhook(self) -> None:
            calls.append("c")
            # Change the ordering in the middle of an ongoing call.
            ordered_pm.set_hookimpl_order("myhook", before={"a": "b"})

    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    ordered_pm.register(PluginC(), name="c")

    # First call: the order is fixed once the call starts, so the change
    # made by c does not affect the ongoing call (default LIFO order).
    ordered_pm.hook.myhook()
    assert calls == ["c", "b", "a"]
    calls.clear()

    # Second call: the new configuration is in effect.
    ordered_pm.hook.myhook()
    assert calls == ["c", "a", "b"]


def test_config_set_between_calls_takes_effect_next_call(
    ordered_pm: PluginManager,
) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)

    ordered_pm.hook.myhook()
    assert calls == ["b", "a"]
    calls.clear()

    ordered_pm.set_hookimpl_order("myhook", before={"a": "b"})
    ordered_pm.hook.myhook()
    assert calls == ["a", "b"]


def test_unregister_invalidates_ordering(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)
    ordered_pm.set_hookimpl_order("myhook", before={"c": "a", "a": "b"})
    ordered_pm.hook.myhook()
    assert calls == ["c", "a", "b"]
    calls.clear()

    ordered_pm.unregister(name="c")
    # The constraint referencing c must not be silently ignored.
    with pytest.raises(HookOrderingError) as excinfo:
        ordered_pm.hook.myhook()
    assert "'c'" in str(excinfo.value)
    assert "myhook" in str(excinfo.value)

    # Setting a new valid configuration makes the hook callable again.
    ordered_pm.set_hookimpl_order("myhook", before={"a": "b"})
    ordered_pm.hook.myhook()
    assert calls == ["a", "b"]


def test_clear_ordering(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    ordered_pm.set_hookimpl_order("myhook", before={"a": "b"})
    ordered_pm.hook.myhook()
    assert calls == ["a", "b"]
    calls.clear()

    # Clearing the configuration also lifts an invalidation.
    ordered_pm.unregister(name="a")
    ordered_pm.set_hookimpl_order("myhook")
    ordered_pm.hook.myhook()
    assert calls == ["b"]


def test_unrelated_unregister_keeps_ordering(ordered_pm: PluginManager) -> None:
    calls: list[str] = []
    register_impl(ordered_pm, "a", calls)
    register_impl(ordered_pm, "b", calls)
    register_impl(ordered_pm, "c", calls)
    ordered_pm.set_hookimpl_order("myhook", before={"a": "b"})

    # c is not referenced by the configuration; removing it is harmless.
    ordered_pm.unregister(name="c")
    ordered_pm.hook.myhook()
    assert calls == ["a", "b"]
