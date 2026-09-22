"""Demonstrate PluginManager.explain_hook against real hook execution.

Run with:

    uv run python docs/examples/explain_hook_demo.py

The script prints, for each scenario, the static explanation plan followed by
the phase log of a real (non-sentinel) hook call so the correspondence is
visible. It also shows the sentinel, unknown-hook and missing-argument cases.
Hook implementations used for explaining are never called.
"""

from __future__ import annotations

from collections.abc import Iterator

import pluggy
from pluggy import HookCallPlan
from pluggy import HookExplainStep


hookspec = pluggy.HookspecMarker("demo")
hookimpl = pluggy.HookimplMarker("demo")


class DemoSpec:
    @hookspec
    def demo_hook(self, x: object, y: object) -> object:
        pass


class FirstImpl:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    @hookimpl
    def demo_hook(self, x: object) -> object:
        self.log.append("first_impl:impl")
        return "first-result"


class LastImpl:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    @hookimpl(trylast=True)
    def demo_hook(self, x: object, y: object) -> object:
        self.log.append("last_impl:impl (trylast)")
        return "last-result"


class NewWrapper:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    @hookimpl(wrapper=True)
    def demo_hook(self, x: object) -> Iterator[None]:
        self.log.append("new_wrapper:enter")
        yield
        self.log.append("new_wrapper:exit")


class OldWrapper:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    @hookimpl(hookwrapper=True)
    def demo_hook(self, x: object, y: object) -> Iterator[None]:
        self.log.append("old_wrapper:enter")
        yield
        self.log.append("old_wrapper:exit")


def _print_step(step: HookExplainStep) -> None:
    flags: list[str] = []
    if step.tryfirst:
        flags.append("tryfirst")
    if step.trylast:
        flags.append("trylast")
    if step.optionalhook:
        flags.append("optionalhook")
    if step.conditional:
        flags.append("CONDITIONAL (firstresult may skip)")
    if step.may_short_circuit:
        flags.append("may short-circuit on non-None")
    if step.missing_args:
        flags.append("MISSING ARGS " + repr(step.missing_args))
    suffix = "  [" + ", ".join(flags) + "]" if flags else ""
    line = (
        f"    {step.plugin_name:12} {step.kind:11} "
        f"args={step.argnames} at {step.location}{suffix}"
    )
    print(line)


def print_plan(plan: HookCallPlan) -> None:
    print(
        "  explain_hook("
        + repr(plan.hook_name)
        + ", kwargs="
        + repr(list(plan.kwargs))
        + ")"
    )
    print("  firstresult=" + str(plan.firstresult) + " has_spec=" + str(plan.has_spec))
    print("  ENTER (wrapper setup, execution order):")
    for step in plan.enter_steps:
        _print_step(step)
    print("  IMPL (normal implementations, execution order):")
    for step in plan.impl_steps:
        _print_step(step)
    print("  EXIT (wrapper teardown, execution order):")
    for step in plan.exit_steps:
        _print_step(step)
    if plan.diagnostics:
        print("  DIAGNOSTICS:")
        for diagnostic in plan.diagnostics:
            print("    [" + diagnostic.code + "] " + diagnostic.message)


def build_manager(
    register_order: list[str],
) -> tuple[pluggy.PluginManager, list[str], dict[str, object]]:
    pm = pluggy.PluginManager("demo")
    pm.add_hookspecs(DemoSpec)
    log: list[str] = []
    plugins: dict[str, object] = {
        "first_impl": FirstImpl(log),
        "last_impl": LastImpl(log),
        "new_wrapper": NewWrapper(log),
        "old_wrapper": OldWrapper(log),
    }
    for name in register_order:
        pm.register(plugins[name], name=name)
    return pm, log, plugins


def scenario(
    title: str,
    register_order: list[str],
    *,
    remove_plugins: set[str] | None = None,
) -> None:
    print("=" * 72)
    print(title)
    pm, log, plugins = build_manager(register_order)
    remove = remove_plugins or set()
    plan = pm.explain_hook("demo_hook", ["x", "y"], remove_plugins=remove)
    print("-- PLAN (nothing is executed) --")
    print_plan(plan)

    print("-- REAL CALL LOG --")
    if remove:
        caller = pm.subset_hook_caller("demo_hook", [plugins[name] for name in remove])
    else:
        caller = pm.hook.demo_hook
    caller(x=1, y=2)
    for entry in log:
        print("    " + entry)
    print()


def sentinel_scenario() -> None:
    print("=" * 72)
    print("SENTINEL SCENARIO: every implementation raises on call")

    class RaisingImpl:
        @hookimpl
        def demo_hook(self, x: object) -> object:
            raise RuntimeError("sentinel called")

    class RaisingWrapper:
        @hookimpl(wrapper=True)
        def demo_hook(self, x: object) -> Iterator[None]:
            raise RuntimeError("sentinel called")
            yield

    pm = pluggy.PluginManager("demo")
    pm.add_hookspecs(DemoSpec)
    pm.register(RaisingImpl(), name="raising_impl")
    pm.register(RaisingWrapper(), name="raising_wrapper")
    plan = pm.explain_hook("demo_hook", ["x", "y"])
    print_plan(plan)
    print("  -> explain_hook succeeded; a real call would raise instead")
    print()


def missing_arguments_scenario() -> None:
    print("=" * 72)
    print("MISSING ARGUMENTS DIAGNOSTIC")
    pm, _, _ = build_manager(["first_impl", "last_impl"])
    plan = pm.explain_hook("demo_hook", ["x"])  # y is missing
    print_plan(plan)


def unknown_hook_scenario() -> None:
    print("=" * 72)
    print("UNKNOWN HOOK + INVALID PLUGIN FILTER DIAGNOSTICS")
    pm, _, _ = build_manager(["first_impl"])
    plan = pm.explain_hook("does_not_exist", ["x"], remove_plugins={"not_registered"})
    print_plan(plan)


if __name__ == "__main__":
    scenario(
        "BASELINE: 2 impls + new wrapper + old hookwrapper",
        ["first_impl", "last_impl", "new_wrapper", "old_wrapper"],
    )
    scenario(
        "REORDERED + SUBSET FILTER (remove new_wrapper and first_impl)",
        ["old_wrapper", "first_impl", "new_wrapper", "last_impl"],
        remove_plugins={"new_wrapper", "first_impl"},
    )
    sentinel_scenario()
    unknown_hook_scenario()
    missing_arguments_scenario()
