"""Tests for ``PluginManager.register_many`` atomic batch registration."""

import pytest

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy import PluginValidationError


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Hooks:
    @hookspec(historic=True)
    def he_method1(self, arg):
        pass

    @hookspec
    def he_method2(self, arg):
        pass


@pytest.fixture
def he_pm(pm: PluginManager) -> PluginManager:
    pm.add_hookspecs(Hooks)
    return pm


def test_register_many_success_order_and_single_historic_replay(
    he_pm: PluginManager,
) -> None:
    calls = []

    class OldPlugin:
        @hookimpl
        def he_method1(self, arg):
            calls.append(("old", arg))

    he_pm.register(OldPlugin(), "old")
    he_pm.hook.he_method1.call_historic(kwargs=dict(arg=1))
    assert calls == [("old", 1)]

    class PluginA:
        @hookimpl(tryfirst=True)
        def he_method1(self, arg):
            calls.append(("a", arg))

    class PluginB:
        @hookimpl(trylast=True)
        def he_method1(self, arg):
            calls.append(("b", arg))

    pa, pb = PluginA(), PluginB()
    names = he_pm.register_many([(pa, "a"), (pb, "b")])
    assert names == ["a", "b"]
    assert he_pm.get_plugins() >= {pa, pb}
    # Each new impl saw the historic call exactly once, in input order.
    assert calls == [("old", 1), ("a", 1), ("b", 1)]

    # tryfirst/trylast ordering matches sequential register().
    hookimpls = he_pm.hook.he_method1.get_hookimpls()
    assert [impl.plugin_name for impl in hookimpls] == ["b", "old", "a"]

    calls.clear()
    he_pm.hook.he_method1.call_historic(kwargs=dict(arg=2))
    assert calls == [("a", 2), ("old", 2), ("b", 2)]


def test_register_many_matches_sequential_register_order(
    he_pm: PluginManager,
) -> None:
    def make_plugin(wrapper=False, **opts):
        class P:
            pass

        # he_method1 is historic, which is incompatible with wrappers.
        P.he_method1 = hookimpl(**opts)(lambda self, arg: None)
        if wrapper:
            opts = {**opts, "wrapper": True}

            def gen_impl(self, arg):
                yield

            P.he_method2 = hookimpl(**opts)(gen_impl)
        else:
            P.he_method2 = hookimpl(**opts)(lambda self, arg: None)
        return P()

    specs = [
        ("p1", dict(tryfirst=True)),
        ("p2", dict(wrapper=True, trylast=True)),
        ("p3", dict()),
        ("p4", dict(trylast=True)),
        ("p5", dict(wrapper=True)),
    ]

    seq_pm = PluginManager("example")
    seq_pm.add_hookspecs(Hooks)
    batch_pm = PluginManager("example")
    batch_pm.add_hookspecs(Hooks)
    for name, opts in specs:
        seq_pm.register(make_plugin(**opts), name)
    batch_pm.register_many([(make_plugin(**opts), name) for name, opts in specs])

    for hook_name in ("he_method1", "he_method2"):
        seq_hook = getattr(seq_pm.hook, hook_name)
        batch_hook = getattr(batch_pm.hook, hook_name)
        assert [i.plugin_name for i in batch_hook.get_hookimpls()] == [
            i.plugin_name for i in seq_hook.get_hookimpls()
        ]


def test_register_many_signature_error_rejects_whole_batch(
    he_pm: PluginManager,
) -> None:
    class Good:
        @hookimpl
        def he_method1(self, arg):
            pass

    class Bad:
        @hookimpl
        def he_method1(self, arg, unknownarg):
            pass

    good, bad = Good(), Bad()
    before_names = dict(he_pm._name2plugin)
    before_impls = he_pm.hook.he_method1.get_hookimpls()
    with pytest.raises(PluginValidationError) as excinfo:
        he_pm.register_many([(good, "good"), (bad, "bad")])
    assert excinfo.value.plugin is bad
    assert "bad" in str(excinfo.value)
    # Neither new plugin leaked into the manager.
    assert not he_pm.is_registered(good)
    assert not he_pm.is_registered(bad)
    assert he_pm.get_plugin("good") is None
    assert he_pm.get_plugin("bad") is None
    assert dict(he_pm._name2plugin) == before_names
    assert he_pm.hook.he_method1.get_hookimpls() == before_impls


def test_register_many_multi_hook_plugin_invalid_later_hook(
    he_pm: PluginManager,
) -> None:
    """A plugin with several hooks is rejected if a later one is invalid."""

    class Multi:
        @hookimpl
        def he_method1(self, arg):
            pass

        @hookimpl
        def he_method2(self, arg, unknownarg):
            pass

    multi = Multi()
    with pytest.raises(PluginValidationError) as excinfo:
        he_pm.register_many([(multi, "multi")])
    assert excinfo.value.plugin is multi
    assert not he_pm.is_registered(multi)
    assert he_pm.get_plugin("multi") is None
    assert he_pm.hook.he_method1.get_hookimpls() == []
    assert he_pm.hook.he_method2.get_hookimpls() == []


def test_register_many_batch_conflicts(he_pm: PluginManager) -> None:
    class P:
        pass

    p1, p2 = P(), P()
    he_pm.register(P(), "taken")
    he_pm.set_blocked("blocked")

    # Duplicate name within the batch.
    with pytest.raises(ValueError, match="twice in the same batch"):
        he_pm.register_many([(p1, "x"), (p2, "x")])
    # Duplicate object within the batch.
    with pytest.raises(ValueError, match="different name"):
        he_pm.register_many([(p1, "x"), (p1, "y")])
    # Name already registered.
    with pytest.raises(ValueError, match="already registered"):
        he_pm.register_many([(p1, "taken")])
    # Blocked name.
    with pytest.raises(ValueError, match="blocked"):
        he_pm.register_many([(p1, "blocked")])

    assert not he_pm.is_registered(p1)
    assert not he_pm.is_registered(p2)
    assert he_pm.get_plugin("x") is None
    assert he_pm.get_plugin("y") is None


def test_register_many_historic_replay_exception_keeps_batch(
    he_pm: PluginManager,
) -> None:
    calls = []

    class Boom(Exception):
        pass

    class Raiser:
        @hookimpl
        def he_method1(self, arg):
            calls.append("raiser")
            raise Boom("historic replay failed")

    class Calm:
        @hookimpl
        def he_method1(self, arg):
            calls.append("calm")

    he_pm.hook.he_method1.call_historic(kwargs=dict(arg=1))
    raiser, calm = Raiser(), Calm()
    with pytest.raises(Boom, match="historic replay failed"):
        he_pm.register_many([(raiser, "raiser"), (calm, "calm")])

    # The exception propagated but the published batch stays registered.
    assert he_pm.is_registered(raiser)
    assert he_pm.is_registered(calm)
    assert he_pm.get_plugin("raiser") is raiser
    assert he_pm.get_plugin("calm") is calm
    assert calls == ["raiser"]

    # The batch can still be unregistered one by one.
    assert he_pm.unregister(raiser) is raiser
    assert he_pm.unregister(calm) is calm
    assert not he_pm.is_registered(raiser)
    assert not he_pm.is_registered(calm)


def test_register_many_canonical_names(pm: PluginManager) -> None:
    class P:
        pass

    p = P()
    p.__name__ = "canonical"  # type: ignore[attr-defined]
    assert pm.register_many([(p, None)]) == ["canonical"]
    assert pm.get_plugin("canonical") is p


def test_register_many_empty(pm: PluginManager) -> None:
    assert pm.register_many([]) == []
    assert pm.get_plugins() == set()
