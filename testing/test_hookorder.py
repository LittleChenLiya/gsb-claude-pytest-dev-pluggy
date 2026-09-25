"""Tests for explicit hook implementation ordering
(``PluginManager.set_hook_impl_order``)."""

from __future__ import annotations

import pytest

from pluggy import HookimplMarker
from pluggy import HookOrderError
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Hooks:
    @hookspec
    def myhook(self, arg: int) -> int:
        pass


    @hookspec(firstresult=True)
    def firsthook(self, arg: int) -> int:
        pass


@pytest.fixture
def ordered_pm(pm: PluginManager) -> PluginManager:
    pm.add_hookspecs(Hooks)
    return pm


def make_plugin(name: str, log: list[str], **opts) -> object:
    """Create a plugin recording calls of ``myhook`` into ``log``."""

    class Plugin:
        @hookimpl(**opts)
        def myhook(self, arg: int) -> None:
            log.append(name)

    Plugin.__name__ = name
    return Plugin()


def register_abc(pm: PluginManager, log: list[str]) -> None:
    pm.register(make_plugin("a", log), name="a")
    pm.register(make_plugin("b", log), name="b")
    pm.register(make_plugin("c", log), name="c")


def test_order_takes_effect_between_calls(ordered_pm: PluginManager) -> None:
    """Two calls demonstrate when a new configuration takes effect:
    the first call uses the default order, the second the configured one."""
    log: list[str] = []
    register_abc(ordered_pm, log)

    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "b", "a"]  # default LIFO registration order

    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    log.clear()
    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "a", "b"]


def test_order_with_after(ordered_pm: PluginManager) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)

    ordered_pm.set_hook_impl_order("myhook", after={"a": "c", "b": "a"})

    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "a", "b"]


def test_order_accepts_single_string_values(ordered_pm: PluginManager) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)

    ordered_pm.set_hook_impl_order("myhook", before={"c": ["a", "b"]})
    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "b", "a"]


def test_partial_constraints_keep_original_order(
    ordered_pm: PluginManager,
) -> None:
    """Implementations without constraints keep their relative order and
    are picked in the original call order when nothing blocks them."""
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.register(make_plugin("d", log), name="d")

    ordered_pm.set_hook_impl_order("myhook", before={"c": "a"})

    ordered_pm.hook.myhook(arg=1)
    # Original call order is d, c, b, a; only c must come before a.
    assert log == ["d", "c", "b", "a"]


def test_no_configuration_keeps_default_order(ordered_pm: PluginManager) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "b", "a"]


def test_unknown_hook_name(ordered_pm: PluginManager) -> None:
    with pytest.raises(HookOrderError, match="'otherhook'"):
        ordered_pm.set_hook_impl_order("otherhook", before={"a": "b"})
    with pytest.raises(HookOrderError, match="'otherhook'"):
        ordered_pm.clear_hook_impl_order("otherhook")
    with pytest.raises(HookOrderError, match="'otherhook'"):
        ordered_pm.get_hook_impl_order("otherhook")


def test_unregistered_target_keeps_previous_config(
    ordered_pm: PluginManager,
) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    with pytest.raises(HookOrderError, match="ghost"):
        ordered_pm.set_hook_impl_order("myhook", before={"a": "ghost"})

    log.clear()
    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "a", "b"]


def test_target_not_implementing_hook_keeps_previous_config(
    ordered_pm: PluginManager,
) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.register(object(), name="empty")
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    with pytest.raises(HookOrderError, match="empty"):
        ordered_pm.set_hook_impl_order("myhook", before={"empty": "a"})

    log.clear()
    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "a", "b"]


def test_cycle_is_rejected_and_previous_config_kept(
    ordered_pm: PluginManager,
) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    with pytest.raises(HookOrderError) as excinfo:
        ordered_pm.set_hook_impl_order(
            "myhook", before={"a": "b", "b": "c", "c": "a"}
        )
    message = str(excinfo.value)
    assert "a" in message and "b" in message and "c" in message

    # A self-cycle is a cycle too.
    with pytest.raises(HookOrderError, match="a"):
        ordered_pm.set_hook_impl_order("myhook", before={"a": "a"})

    log.clear()
    ordered_pm.hook.myhook(arg=1)
    assert log == ["c", "a", "b"]


@pytest.mark.parametrize("opts", [{"tryfirst": True}, {"trylast": True}])
def test_cross_tier_conflict_is_rejected(
    ordered_pm: PluginManager, opts: dict[str, bool]
) -> None:
    """tryfirst/trylast implementations live in fixed tiers and cannot be
    referenced by ordering constraints."""
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.register(make_plugin("tiered", log, **opts), name="tiered")

    with pytest.raises(HookOrderError, match="tiered"):
        ordered_pm.set_hook_impl_order("myhook", before={"tiered": "a"})
    with pytest.raises(HookOrderError, match="tiered"):
        ordered_pm.set_hook_impl_order("myhook", before={"a": "tiered"})

    # No configuration was applied; the tiers still rule.
    ordered_pm.hook.myhook(arg=1)
    if opts.get("tryfirst"):
        assert log == ["tiered", "c", "b", "a"]
    else:
        assert log == ["c", "b", "a", "tiered"]


def test_wrapper_cannot_be_ordered(ordered_pm: PluginManager) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)

    class Wrapper:
        @hookimpl(wrapper=True)
        def myhook(self, arg: int):
            yield

    ordered_pm.register(Wrapper(), name="w")

    with pytest.raises(HookOrderError, match="w"):
        ordered_pm.set_hook_impl_order("myhook", before={"w": "a"})


def test_fixed_tiers_with_ordering(ordered_pm: PluginManager) -> None:
    """Ordering only shuffles the normal tier; tryfirst/trylast stay in
    their own tiers."""
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.register(make_plugin("first", log, tryfirst=True), name="first")
    ordered_pm.register(make_plugin("last", log, trylast=True), name="last")

    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    ordered_pm.hook.myhook(arg=1)
    assert log == ["first", "c", "a", "b", "last"]


def test_wrappers_wrap_the_ordered_call(ordered_pm: PluginManager) -> None:
    """Wrappers keep wrapping the (reordered) normal implementations."""
    log: list[str] = []
    register_abc(ordered_pm, log)

    class Wrapper:
        @hookimpl(wrapper=True)
        def myhook(self, arg: int):
            log.append("w-before")
            results = yield
            log.append("w-after")
            return results

    ordered_pm.register(Wrapper(), name="w")
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    ordered_pm.hook.myhook(arg=1)
    assert log == ["w-before", "c", "a", "b", "w-after"]


def test_firstresult_stops_early_in_configured_order(
    ordered_pm: PluginManager,
) -> None:
    """firstresult stops at the first non-None result following the
    configured order."""
    log: list[str] = []

    def make(name: str, result: int | None) -> object:
        class Plugin:
            @hookimpl
            def firsthook(self, arg: int) -> int | None:
                log.append(name)
                return result

        return Plugin()

    ordered_pm.register(make("a", None), name="a")
    ordered_pm.register(make("b", 10), name="b")
    ordered_pm.register(make("c", None), name="c")

    ordered_pm.set_hook_impl_order("firsthook", before={"b": "c"})

    assert ordered_pm.hook.firsthook(arg=1) == 10
    # b runs first and returns a result; c is never called.
    assert log == ["b"]


def test_configuration_change_during_call_affects_next_call(
    ordered_pm: PluginManager,
) -> None:
    """The order of a running call is fixed; changing the configuration
    from within a hook implementation only affects the next call."""
    log: list[str] = []

    class Reconfiguring:
        @hookimpl
        def myhook(self, arg: int) -> None:
            log.append("a")
            ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    class B:
        @hookimpl
        def myhook(self, arg: int) -> None:
            log.append("b")

    class C:
        @hookimpl
        def myhook(self, arg: int) -> None:
            log.append("c")

    ordered_pm.register(Reconfiguring(), name="a")
    ordered_pm.register(B(), name="b")
    ordered_pm.register(C(), name="c")

    ordered_pm.hook.myhook(arg=1)
    # The in-flight call keeps the order fixed when it started.
    assert log == ["c", "b", "a"]

    log.clear()
    ordered_pm.hook.myhook(arg=1)
    # The new configuration applies from the next call on.
    assert log == ["c", "a", "b"]


def test_unregister_invalidates_constraints(ordered_pm: PluginManager) -> None:
    """Unregistering a referenced plugin invalidates the configuration;
    the constraint is not silently ignored."""
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})

    ordered_pm.unregister(name="c")

    with pytest.raises(HookOrderError, match="c"):
        ordered_pm.hook.myhook(arg=1)

    # Clearing the invalid configuration makes the hook callable again.
    ordered_pm.clear_hook_impl_order("myhook")
    ordered_pm.hook.myhook(arg=1)
    assert log == ["a", "b"]


def test_unregister_then_reconfigure(ordered_pm: PluginManager) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})
    ordered_pm.unregister(name="c")

    # A new valid configuration replaces the invalidated one.
    ordered_pm.set_hook_impl_order("myhook", before={"b": "a"})
    ordered_pm.hook.myhook(arg=1)
    assert log == ["b", "a"]


def test_new_registration_respects_existing_constraints(
    ordered_pm: PluginManager,
) -> None:
    log: list[str] = []
    ordered_pm.register(make_plugin("a", log), name="a")
    ordered_pm.register(make_plugin("b", log), name="b")
    ordered_pm.set_hook_impl_order("myhook", before={"a": "b"})

    ordered_pm.register(make_plugin("c", log), name="c")
    ordered_pm.hook.myhook(arg=1)
    # c is unconstrained and comes first in call order; a stays before b.
    assert log == ["c", "a", "b"]


def test_get_and_clear_hook_impl_order(ordered_pm: PluginManager) -> None:
    log: list[str] = []
    register_abc(ordered_pm, log)

    assert ordered_pm.get_hook_impl_order("myhook") == {}
    ordered_pm.set_hook_impl_order("myhook", before={"c": "a", "a": "b"})
    assert ordered_pm.get_hook_impl_order("myhook") == {
        "c": frozenset({"a"}),
        "a": frozenset({"b"}),
    }

    ordered_pm.clear_hook_impl_order("myhook")
    assert ordered_pm.get_hook_impl_order("myhook") == {}
    ordered_pm.hook.myhook(arg=1)
    # Clearing keeps the current relative order.
    assert log == ["c", "a", "b"]
