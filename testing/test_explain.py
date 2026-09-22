from __future__ import annotations

import pytest

from pluggy import HookimplMarker
from pluggy import HookPlan
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Spec:
    @hookspec
    def myhook(self, arg1, arg2):
        pass

    @hookspec(firstresult=True)
    def firsthook(self, arg1):
        pass


def make_pm() -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Spec)
    return pm


def make_logging_plugins(log: list[tuple[str, str]]):
    """Two plain impls, one new-style wrapper, one old-style hookwrapper."""

    class Plain1:
        @hookimpl
        def myhook(self, arg1, arg2):
            log.append(("impl", "Plain1"))
            return 1

    class Plain2:
        @hookimpl
        def myhook(self, arg1):
            log.append(("impl", "Plain2"))

    class NewWrapper:
        @hookimpl(wrapper=True)
        def myhook(self, arg1, arg2):
            log.append(("enter", "NewWrapper"))
            result = yield
            log.append(("exit", "NewWrapper"))
            return result

    class OldWrapper:
        @hookimpl(hookwrapper=True)
        def myhook(self, arg1):
            log.append(("enter", "OldWrapper"))
            result = yield
            log.append(("exit", "OldWrapper"))
            return result.get_result()

    return [Plain1(), Plain2(), NewWrapper(), OldWrapper()]


def plan_events(plan: HookPlan) -> list[tuple[str, str]]:
    return [(step.phase, step.plugin_name) for step in plan.steps]


def register_all(pm: PluginManager, plugins: list[object]) -> None:
    for plugin in plugins:
        pm.register(plugin, name=type(plugin).__name__)


def test_plan_matches_real_call() -> None:
    pm = make_pm()
    log: list[tuple[str, str]] = []
    plugins = make_logging_plugins(log)
    register_all(pm, plugins)

    plan = pm.explain_hook("myhook", {"arg1", "arg2"})
    assert plan.ok
    assert [step.kind for step in plan.enter] == ["hookwrapper", "wrapper"]
    assert [step.kind for step in plan.impls] == ["impl", "impl"]
    assert [step.kind for step in plan.exit] == ["wrapper", "hookwrapper"]

    pm.hook.myhook(arg1=1, arg2=2)
    assert log == plan_events(plan)


def test_plan_matches_real_call_reordered_trylast_subset() -> None:
    pm = make_pm()
    log: list[tuple[str, str]] = []
    plain1, plain2, newwrap, oldwrap = make_logging_plugins(log)

    class TryLast:
        @hookimpl(trylast=True)
        def myhook(self, arg1):
            log.append(("impl", "TryLast"))

    trylast = TryLast()
    # Different registration order than test_plan_matches_real_call.
    for plugin in [oldwrap, trylast, plain2, newwrap, plain1]:
        pm.register(plugin, name=type(plugin).__name__)

    plan = pm.explain_hook("myhook", {"arg1", "arg2"}, remove_plugins={plain1})
    assert plan.ok
    assert "Plain1" not in {step.plugin_name for step in plan.steps}

    subset = pm.subset_hook_caller("myhook", {plain1})
    subset(arg1=1, arg2=2)
    assert log == plan_events(plan)
    # trylast runs last among the regular implementations.
    # trylast runs last among the regular implementations.
    assert plan.impls[-1].plugin_name == "TryLast"
    assert plan.impls[-1].trylast


def test_sentinel_impls_are_not_executed() -> None:
    pm = make_pm()

    class Sentinel:
        @hookimpl
        def myhook(self, arg1):
            raise AssertionError("sentinel impl executed")

        @hookimpl(wrapper=True)
        def firsthook(self, arg1):
            raise AssertionError("sentinel wrapper executed")
            yield

        @hookimpl(hookwrapper=True)
        def firsthook_legacy(self, arg1):
            raise AssertionError("sentinel old hookwrapper executed")
            yield

    pm.register(Sentinel(), name="Sentinel")

    plan = pm.explain_hook("myhook", {"arg1", "arg2"})
    assert plan.ok
    assert [step.plugin_name for step in plan.impls] == ["Sentinel"]

    wrapper_plan = pm.explain_hook("firsthook", {"arg1"})
    assert [step.kind for step in wrapper_plan.enter] == ["wrapper"]

    legacy_plan = pm.explain_hook(
        "firsthook_legacy", {"arg1"}, remove_plugins=["Sentinel"]
    )
    assert legacy_plan.ok
    assert legacy_plan.enter == ()

    legacy_plan_with_sentinel = pm.explain_hook("firsthook_legacy", {"arg1"})
    assert [step.kind for step in legacy_plan_with_sentinel.enter] == ["hookwrapper"]

    with pytest.raises(AssertionError, match="sentinel impl executed"):
        pm.hook.myhook(arg1=1, arg2=2)


def test_firstresult_conditional_steps() -> None:
    pm = make_pm()
    calls: list[str] = []

    class First:
        @hookimpl
        def firsthook(self, arg1):
            calls.append("first")
            return "fallback"

    class Second:
        @hookimpl
        def firsthook(self, arg1):
            calls.append("second")
            return "winner"

    pm.register(First(), name="First")
    pm.register(Second(), name="Second")

    plan = pm.explain_hook("firsthook", {"arg1"})
    assert plan.firstresult
    assert len(plan.impls) == 2
    assert not plan.impls[0].conditional
    assert plan.impls[1].conditional
    assert all(step.may_shortcut for step in plan.impls)

    result = pm.hook.firsthook(arg1=1)
    assert result == "winner"
    # "Second" is registered last, so it runs first and wins; the plan must
    # not have pre-decided this, only marked "first" as conditional.
    assert calls == ["second"]
    assert plan.impls[0].plugin_name == "Second"
    assert plan.impls[1].plugin_name == "First"


def test_plan_is_a_snapshot() -> None:
    pm = make_pm()
    log: list[tuple[str, str]] = []
    plugins = make_logging_plugins(log)
    register_all(pm, plugins)

    plan = pm.explain_hook("myhook", {"arg1", "arg2"})
    names_before = {step.plugin_name for step in plan.steps}

    pm.unregister(plugins[0])
    pm.unregister(plugins[2])

    # The old plan is unchanged.
    assert {step.plugin_name for step in plan.steps} == names_before
    assert "Plain1" in names_before

    new_plan = pm.explain_hook("myhook", {"arg1", "arg2"})
    assert {step.plugin_name for step in new_plan.steps} == names_before - {
        "Plain1",
        "NewWrapper",
    }


def test_step_details() -> None:
    pm = make_pm()
    log: list[tuple[str, str]] = []
    plugins = make_logging_plugins(log)
    register_all(pm, plugins)

    plan = pm.explain_hook("myhook", {"arg1", "arg2"})
    by_name = {step.plugin_name: step for step in plan.impls}
    plain1 = by_name["Plain1"]
    assert plain1.function_name == "myhook"
    assert plain1.argnames == ("arg1", "arg2")
    first_lineno = plugins[0].myhook.__code__.co_firstlineno  # type: ignore[attr-defined]
    assert plain1.location.endswith(".py:" + str(first_lineno))
    assert by_name["Plain2"].argnames == ("arg1",)
    for step in plan.steps:
        assert not step.conditional
        assert not step.may_shortcut


def test_diagnostics_unknown_hook() -> None:
    pm = make_pm()
    plan = pm.explain_hook("nosuchhook", {"arg1"})
    assert not plan.ok
    assert [d.kind for d in plan.diagnostics] == ["unknown-hook"]
    assert plan.steps == ()


def test_diagnostics_missing_arguments() -> None:
    pm = make_pm()
    log: list[tuple[str, str]] = []
    plugins = make_logging_plugins(log)
    register_all(pm, plugins)

    # arg2 is declared by the hookspec and required by Plain1.
    plan = pm.explain_hook("myhook", {"arg1"})
    kinds = [d.kind for d in plan.diagnostics]
    assert "missing-arguments" in kinds
    assert "missing-impl-arguments" in kinds
    missing_impl_messages = [
        d.message for d in plan.diagnostics if d.kind == "missing-impl-arguments"
    ]
    assert any("Plain1" in msg and "arg2" in msg for msg in missing_impl_messages)


def test_diagnostics_invalid_plugin_filter() -> None:
    pm = make_pm()
    log: list[tuple[str, str]] = []
    plugins = make_logging_plugins(log)
    register_all(pm, plugins)

    plan = pm.explain_hook(
        "myhook",
        {"arg1", "arg2"},
        remove_plugins={object(), "not-a-plugin", plugins[0]},
    )
    kinds = [d.kind for d in plan.diagnostics]
    assert kinds.count("invalid-plugin-filter") == 2
    # The valid filter entry is still applied.
    assert "Plain1" not in {step.plugin_name for step in plan.steps}


def test_diagnostics_no_spec_non_optional() -> None:
    pm = make_pm()

    class Stray:
        @hookimpl
        def unknownhook(self, arg1):
            pass  # pragma: no cover

    pm.register(Stray())
    plan = pm.explain_hook("unknownhook", {"arg1"})
    assert not plan.has_spec
    assert [d.kind for d in plan.diagnostics] == ["no-spec"]


def test_optionalhook_no_spec_is_ok() -> None:
    pm = make_pm()

    class Stray:
        @hookimpl(optionalhook=True)
        def unknownhook(self, arg1):
            pass  # pragma: no cover

    pm.register(Stray())
    plan = pm.explain_hook("unknownhook", {"arg1"})
    assert plan.ok
    assert [step.optionalhook for step in plan.impls] == [True]
