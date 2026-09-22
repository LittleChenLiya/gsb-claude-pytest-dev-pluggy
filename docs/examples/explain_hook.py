"""
Demonstration for PluginManager.explain_hook.

Run with:  python docs/examples/explain_hook.py

It first prints a static
plan of a hook call (without executing any plugin), then performs the real
call so the enter/impl/exit phases can be compared side by side, and finally
shows the diagnostics produced when a required argument is missing.
"""

from __future__ import annotations

import pluggy


hookspec = pluggy.HookspecMarker("demo")
hookimpl = pluggy.HookimplMarker("demo")


class DemoSpec:
    @hookspec
    def process(self, value, config):
        """Process a value."""


class PluginA:
    @hookimpl
    def process(self, value, config):
        print(f"  impl    PluginA.process(value={value}, config=...) -> {value} + a")
        return value + "a"


class PluginB:
    @hookimpl
    def process(self, value):
        print(f"  impl    PluginB.process(value={value})")


class LogWrapper:
    """New-style wrapper."""

    @hookimpl(wrapper=True)
    def process(self, value, config):
        print(f"  enter   LogWrapper.process(value={value}, config=...)")
        result = yield
        print(f"  exit    LogWrapper.process -> {result}")
        return result


class LegacyWrapper:
    """Old-style hookwrapper."""

    @hookimpl(hookwrapper=True)
    def process(self, value):
        print(f"  enter   LegacyWrapper.process(value={value})")
        outcome = yield
        print(f"  exit    LegacyWrapper.process -> {outcome.get_result()}")


def print_plan(plan: pluggy.HookPlan) -> None:
    print(
        f"Plan for hook {plan.hook_name!r} "
        f"(firstresult={plan.firstresult}, has_spec={plan.has_spec})"
    )
    for phase_name in ("enter", "impls", "exit"):
        print(f"  [{phase_name}]")
        for step in getattr(plan, phase_name):
            flags = []
            if step.tryfirst:
                flags.append("tryfirst")
            if step.trylast:
                flags.append("trylast")
            if step.optionalhook:
                flags.append("optionalhook")
            if step.conditional:
                flags.append("conditional")
            if step.may_shortcut:
                flags.append("may-shortcut")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(
                f"    {step.kind:12} {step.plugin_name}.{step.function_name}"
                f"({', '.join(step.argnames)}) at {step.location}{suffix}"
            )
    if plan.diagnostics:
        print("  diagnostics:")
        for diag in plan.diagnostics:
            print(f"    - {diag.kind}: {diag.message}")


def main() -> None:
    pm = pluggy.PluginManager("demo")
    pm.add_hookspecs(DemoSpec)
    for plugin, name in [
        (PluginA(), "PluginA"),
        (PluginB(), "PluginB"),
        (LogWrapper(), "LogWrapper"),
        (LegacyWrapper(), "LegacyWrapper"),
    ]:
        pm.register(plugin, name=name)

    print("=== Static plan (nothing executed) ===")
    plan = pm.explain_hook("process", {"value", "config"})
    print_plan(plan)

    print()
    print("=== Real call log ===")
    result = pm.hook.process(value="x", config={})
    print(f"  result: {result}")

    print()
    print("=== Diagnostics for a call missing 'config' ===")
    bad_plan = pm.explain_hook("process", {"value"})
    for diag in bad_plan.diagnostics:
        print(f"  {diag.kind}: {diag.message}")


if __name__ == "__main__":
    main()
