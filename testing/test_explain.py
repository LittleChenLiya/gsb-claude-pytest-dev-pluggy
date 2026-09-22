from __future__ import annotations

from collections.abc import Iterator

import pytest

import pluggy
from pluggy import ExplainDiagnostic
from pluggy import HookCallError
from pluggy import HookCallPlan
from pluggy import HookExplainStep


hookspec = pluggy.HookspecMarker("example")
hookimpl = pluggy.HookimplMarker("example")


class HookSpec:
    @hookspec
    def hook(self, x: object, y: object) -> object:
        pass


def _step_ids(steps: tuple[HookExplainStep, ...]) -> list[tuple[str, str]]:
    return [(s.plugin_name, s.kind) for s in steps]


def _plan_trace(plan: HookCallPlan) -> list[tuple[str, str]]:
    return (
        [(n, "enter") for n, k in _step_ids(plan.enter_steps)]
        + [(n, "impl") for n, k in _step_ids(plan.impl_steps)]
        + [(n, "exit") for n, k in _step_ids(plan.exit_steps)]
    )


class RecordingPlugins:
    """Two normal impls, one new wrapper and one old-style hookwrapper."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def _record(self, name: str, phase: str) -> None:
        self.log.append(f"{name}:{phase}")

    @hookimpl
    def first_impl(self, x: object) -> object:
        self._record("first_impl", "impl")
        return "first"

    @hookimpl
    def second_impl(self, x: object, y: object) -> object:
        self._record("second_impl", "impl")
        return "second"

    @hookimpl(wrapper=True)
    def new_wrapper(self, x: object) -> Iterator[None]:
        self._record("new_wrapper", "enter")
        yield
        self._record("new_wrapper", "exit")

    @hookimpl(hookwrapper=True)
    def old_wrapper(self, x: object, y: object) -> Iterator[None]:
        self._record("old_wrapper", "enter")
        yield
        self._record("old_wrapper", "exit")


def _patch_trylast(plugin: object, method_names: set[str]) -> None:
    for method_name in method_names:
        getattr(plugin, method_name).example_impl["trylast"] = True


def _build_pm(
    register_order: list[str],
    trylast: set[str] | None = None,
) -> tuple[pluggy.PluginManager, RecordingPlugins, dict[str, object]]:
    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(HookSpec)
    log: list[str] = []
    plugins = RecordingPlugins(log)
    if trylast:
        _patch_trylast(plugins, trylast)
    named: dict[str, object] = {
        "first_impl": _BoundImpl(plugins, "first_impl"),
        "second_impl": _BoundImpl(plugins, "second_impl"),
        "new_wrapper": _BoundImpl(plugins, "new_wrapper"),
        "old_wrapper": _BoundImpl(plugins, "old_wrapper"),
    }
    for name in register_order:
        pm.register(named[name], name=name)
    return pm, plugins, named


class _BoundImpl:
    """A single-method plugin delegating to a RecordingPlugins method."""

    def __init__(self, plugins: RecordingPlugins, method_name: str) -> None:
        self.hook = getattr(plugins, method_name)


@pytest.mark.parametrize(
    "register_order",
    [
        ["first_impl", "second_impl", "new_wrapper", "old_wrapper"],
        ["old_wrapper", "first_impl", "new_wrapper", "second_impl"],
        ["new_wrapper", "old_wrapper", "second_impl", "first_impl"],
    ],
)
@pytest.mark.parametrize("trylast", [None, {"second_impl"}, {"first_impl"}])
def test_plan_matches_real_call_order(register_order, trylast):
    pm, plugins, _named = _build_pm(register_order, trylast)
    plan = pm.explain_hook("hook", ["x", "y"])

    expected = _plan_trace(plan)
    pm.hook.hook(x=1, y=2)
    actual = [
        (event.rsplit(":", 1)[0], event.rsplit(":", 1)[1]) for event in plugins.log
    ]
    assert actual == expected

    # The plan reflects tryfirst/trylast ordering and carries the markers.
    by_name = {
        s.plugin_name: s for s in plan.enter_steps + plan.impl_steps + plan.exit_steps
    }
    if trylast:
        for name in trylast:
            assert by_name[name].trylast is True

    assert plan.diagnostics == ()


def test_subset_filter_uses_real_subset_rules():
    pm, plugins, named = _build_pm(
        ["first_impl", "second_impl", "new_wrapper", "old_wrapper"]
    )
    plan = pm.explain_hook(
        "hook", ["x", "y"], remove_plugins={"new_wrapper", named["second_impl"]}
    )
    names = {
        s.plugin_name for s in plan.enter_steps + plan.impl_steps + plan.exit_steps
    }
    assert names == {"first_impl", "old_wrapper"}

    pm.subset_hook_caller("hook", [named["new_wrapper"], named["second_impl"]])(
        x=1, y=2
    )
    assert plugins.log == [
        "old_wrapper:enter",
        "first_impl:impl",
        "old_wrapper:exit",
    ]


class RaisingPlugin:
    @hookimpl
    def hook(self, x: object) -> object:
        raise AssertionError("sentinel: implementation must not be called")


class RaisingWrapper:
    @hookimpl(wrapper=True)
    def hook(self, x: object) -> Iterator[None]:
        raise AssertionError("sentinel: wrapper must not be called")
        yield


class RaisingOldWrapper:
    @hookimpl(hookwrapper=True)
    def hook(self, x: object) -> Iterator[None]:
        raise AssertionError("sentinel: hookwrapper must not be called")
        yield


def test_explain_does_not_execute_anything():
    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(HookSpec)
    pm.register(RaisingPlugin(), name="raising_impl")
    pm.register(RaisingWrapper(), name="raising_wrapper")
    pm.register(RaisingOldWrapper(), name="raising_old_wrapper")

    plan = pm.explain_hook("hook", ["x", "y"])
    assert [s.plugin_name for s in plan.enter_steps] == [
        "raising_old_wrapper",
        "raising_wrapper",
    ]
    assert [s.plugin_name for s in plan.impl_steps] == ["raising_impl"]
    assert [s.plugin_name for s in plan.exit_steps] == [
        "raising_wrapper",
        "raising_old_wrapper",
    ]
    for step in plan.enter_steps + plan.impl_steps + plan.exit_steps:
        assert step.location is not None
        assert step.location[0].isalpha()
        assert ":1" in step.location or ":" in step.location
        assert step.argnames == ("x",)


class DescriptorPlugin:
    @property
    def hook(self) -> object:
        raise AssertionError("descriptor must not be executed")


def test_explain_does_not_execute_descriptors():
    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(HookSpec)
    pm.register(DescriptorPlugin(), name="descriptor_plugin")
    plan = pm.explain_hook("hook", ["x", "y"])
    assert plan.impl_steps == ()
    assert plan.enter_steps == ()


class FirstResultSpec:
    @hookspec(firstresult=True)
    def fr(self, x: object) -> object:
        pass


class FRPlugin:
    def __init__(self, value: object) -> None:
        self.value = value

    @hookimpl
    def fr(self, x: object) -> object:
        return self.value


def test_firstresult_steps_are_conditional_but_no_winner_predicted():
    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(FirstResultSpec)
    pm.register(FRPlugin(None), name="a")
    pm.register(FRPlugin("win"), name="b")
    pm.register(FRPlugin("later"), name="c")

    plan = pm.explain_hook("fr", ["x"])
    assert plan.firstresult is True
    names = [s.plugin_name for s in plan.impl_steps]
    assert names == ["c", "b", "a"]
    assert plan.impl_steps[0].conditional is False
    assert plan.impl_steps[1].conditional is True
    assert plan.impl_steps[2].conditional is True
    assert all(s.may_short_circuit for s in plan.impl_steps)

    # When the first possible short-circuit does return, c is skipped.
    assert pm.hook.fr(x=1) == "later"

    pm2 = pluggy.PluginManager("example")
    pm2.add_hookspecs(FirstResultSpec)
    pm2.register(FRPlugin(None), name="a")
    pm2.register(FRPlugin(None), name="b")
    pm2.register(FRPlugin("later"), name="c")
    assert pm2.hook.fr(x=1) == "later"


def test_plan_is_an_independent_snapshot():
    pm, _, named = _build_pm(
        ["first_impl", "second_impl", "new_wrapper", "old_wrapper"]
    )
    plan = pm.explain_hook("hook", ["x", "y"])
    before = (
        plan.enter_steps,
        plan.impl_steps,
        plan.exit_steps,
    )
    pm.unregister(named["first_impl"])
    pm.unregister(named["new_wrapper"])
    assert (
        plan.enter_steps,
        plan.impl_steps,
        plan.exit_steps,
    ) == before
    assert len(plan.impl_steps) == 2
    assert len(plan.enter_steps) == 2


def test_firstresult_conditional_nodes_match_real_skips():
    events: list[str] = []

    class FRImpl:
        def __init__(self, name: str, value: object) -> None:
            self.name = name
            self.value = value

        @hookimpl
        def fr(self, x: object) -> object:
            events.append(self.name)
            return self.value

    # Execution order for registered a,b,c is reversed: c, b, a.
    scenarios = [
        # values in registration order a, b, c; (expected ran set, winner)
        ((None, None, None), {"c", "b", "a"}, None),
        ((None, None, "win"), {"c"}, "win"),
        ((None, "win", None), {"c", "b"}, "win"),
        (("win", None, None), {"c", "b", "a"}, "win"),
        ((None, "later", "later"), {"c"}, "later"),
    ]
    for values, expected_ran, expected_winner in scenarios:
        events.clear()
        pm = pluggy.PluginManager("example")
        pm.add_hookspecs(FirstResultSpec)
        for name, value in zip(("a", "b", "c"), values, strict=True):
            pm.register(FRImpl(name, value), name=name)
        plan = pm.explain_hook("fr", ["x"])
        assert [s.plugin_name for s in plan.impl_steps] == ["c", "b", "a"]
        assert pm.hook.fr(x=1) == expected_winner
        assert set(events) == expected_ran


def test_firstresult_wrappers_always_enter_and_exit():
    events: list[str] = []

    class Win:
        @hookimpl
        def fr(self, x: object) -> object:
            return "won"

    class Wrap:
        @hookimpl(wrapper=True)
        def fr(self) -> Iterator[object | None]:
            events.append("wrap-enter")
            outcome = yield
            events.append("wrap-exit")
            return outcome

    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(FirstResultSpec)
    pm.register(Win(), name="win")
    pm.register(Wrap(), name="wrap")
    plan = pm.explain_hook("fr", ["x"])
    assert plan.enter_steps[0].conditional is False
    assert pm.hook.fr(x=1) == "won"
    assert events == ["wrap-enter", "wrap-exit"]


def test_unknown_hook_diagnostic():
    pm = pluggy.PluginManager("example")
    plan = pm.explain_hook("does_not_exist", ["x"])
    assert plan.enter_steps == plan.impl_steps == plan.exit_steps == ()
    assert plan.diagnostics == (
        ExplainDiagnostic("unknown_hook", "unknown hook 'does_not_exist'"),
    )


def test_missing_arguments_diagnostic():
    pm, _, _ = _build_pm(["first_impl", "second_impl"])
    plan = pm.explain_hook("hook", ["x"])
    codes = [d.code for d in plan.diagnostics]
    assert codes == ["missing_arguments", "missing_arguments"]
    assert "y" in plan.diagnostics[0].message
    by_name = {s.plugin_name: s for s in plan.impl_steps}
    assert by_name["first_impl"].missing_args == ()
    assert by_name["second_impl"].missing_args == ("y",)


def test_invalid_remove_plugin_diagnostic_distinguishes_name_and_object():
    pm, _, _ = _build_pm(["first_impl"])
    plan = pm.explain_hook(
        "hook", ["x", "y"], remove_plugins=["missing_name", object()]
    )
    assert [d.code for d in plan.diagnostics] == [
        "invalid_remove_plugin",
        "invalid_remove_plugin",
    ]


def test_invalid_filter_unknown_hook_and_missing_args_are_distinct():
    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(HookSpec)
    plan = pm.explain_hook("nope", remove_plugins=["nope_plugin"])
    codes = {d.code for d in plan.diagnostics}
    assert codes == {"unknown_hook", "invalid_remove_plugin"}


def test_missing_argument_real_call_raises_hookcallerror():
    pm, _, _ = _build_pm(["second_impl"])
    with (
        pytest.warns(UserWarning, match="cannot be found"),
        pytest.raises(HookCallError),
    ):
        pm.hook.hook(x=1)


def test_kwargs_accepts_mapping_and_iterable():
    pm, _, _ = _build_pm(["first_impl"])
    from_mapping = pm.explain_hook("hook", {"x": 1, "y": 2})
    from_iterable = pm.explain_hook("hook", ("x", "y"))
    assert from_mapping.kwargs == from_iterable.kwargs == ("x", "y")
    assert from_mapping.diagnostics == from_iterable.diagnostics == ()


def test_step_carries_binding_information():
    pm, _, _ = _build_pm(["second_impl", "old_wrapper"])
    plan = pm.explain_hook("hook", ["x"])
    by_name = {s.plugin_name: s for s in plan.enter_steps + plan.impl_steps}
    assert by_name["second_impl"].argnames == ("x", "y")
    assert by_name["second_impl"].missing_args == ("y",)
    assert by_name["old_wrapper"].function_name == "old_wrapper"
    assert by_name["old_wrapper"].kind == "hookwrapper"
    assert by_name["old_wrapper"].optionalhook is False


def test_optionalhook_impl_is_recorded():
    class OptionalPlugin:
        @hookimpl(optionalhook=True)
        def unregistered_hook(self, x: object) -> None:
            pass

    pm = pluggy.PluginManager("example")
    pm.register(OptionalPlugin(), name="opt")
    pm.check_pending()
    plan = pm.explain_hook("unregistered_hook", ["x"])
    assert plan.has_spec is False
    assert plan.impl_steps[0].optionalhook is True


def test_blocked_plugin_name_is_an_invalid_filter():
    pm, _, _ = _build_pm(["first_impl"])
    pm.set_blocked("first_impl")
    # Blocking also unregistered it.
    assert pm.get_plugin("first_impl") is None
    plan = pm.explain_hook("hook", ["x", "y"], remove_plugins=["first_impl"])
    assert [d.code for d in plan.diagnostics] == ["invalid_remove_plugin"]


def test_kwargs_are_copied_into_the_snapshot():
    pm, _, _ = _build_pm(["first_impl"])
    provided = ["x", "y"]
    plan = pm.explain_hook("hook", provided)
    provided.append("z")
    assert plan.kwargs == ("x", "y")


def test_historic_plan_is_flagged_and_callbacks_are_not_executed():
    class HistoricSpec:
        @hookspec(historic=True)
        def hist(self, x: object) -> None:
            pass

    class HistoricPlugin:
        @hookimpl
        def hist(self, x: object) -> None:
            pass

    pm = pluggy.PluginManager("example")
    pm.add_hookspecs(HistoricSpec)
    pm.register(HistoricPlugin(), name="h")
    plan = pm.explain_hook("hist", ["x"])
    assert plan.historic is True
    assert plan.impl_steps[0].plugin_name == "h"

    def callback(result: object) -> None:
        raise AssertionError("result callback must not run during explain")

    # A real historic call stores the callback for late registrations.
    pm.hook.hist.call_historic(callback, {"x": 1})

    # Explaining must not replay history or invoke the stored callback.
    plan = pm.explain_hook("hist", ["x"])
    assert plan.impl_steps[0].plugin_name == "h"


def test_exit_steps_are_reverse_of_enter_steps():
    pm, _, _ = _build_pm(["new_wrapper", "old_wrapper"])
    plan = pm.explain_hook("hook", ["x", "y"])
    assert [s.plugin_name for s in plan.exit_steps] == list(
        reversed([s.plugin_name for s in plan.enter_steps])
    )


def test_no_spec_hook_without_optionalhook_still_explains():
    class PlainPlugin:
        @hookimpl
        def unknown_spec_hook(self, x: object) -> None:
            pass

    pm = pluggy.PluginManager("example")
    pm.register(PlainPlugin(), name="plain")
    plan = pm.explain_hook("unknown_spec_hook", ["x"])
    assert plan.has_spec is False
    assert plan.diagnostics == ()
    assert plan.impl_steps[0].plugin_name == "plain"
