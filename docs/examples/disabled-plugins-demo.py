"""Demonstrate temporarily disabling plugins with PluginManager.disabled_plugins().

Run with:

    uv run python docs/examples/disabled-plugins-demo.py
"""

import pluggy


hookspec = pluggy.HookspecMarker("demo")
hookimpl = pluggy.HookimplMarker("demo")


class Spec:
    @hookspec
    def render(self, text):
        """Render a piece of text."""


class PluginA:
    @hookimpl
    def render(self, text):
        print("    plugin A runs")
        return "A"


class PluginB:
    @hookimpl
    def render(self, text):
        print("    plugin B runs")
        return "B"


class TracingWrapper:
    @hookimpl(wrapper=True)
    def render(self, text):
        print("    wrapper: before impls")
        result = yield
        print("    wrapper: after impls")
        return result


def main():
    pm = pluggy.PluginManager("demo")
    pm.add_hookspecs(Spec)
    pm.register(PluginA(), name="A")
    pm.register(PluginB(), name="B")
    pm.register(TracingWrapper(), name="tracing")

    def call(label):
        print(f"[{label}]")
        result = pm.hook.render(text="hello")
        print(f"    -> results: {result}")

    call("no plugins disabled")
    with pm.disabled_plugins({"B"}):
        call("B disabled")
        with pm.disabled_plugins({"A"}):
            call("A and B disabled (nested scopes form a union)")
        call("B disabled again (inner scope exited)")
    call("restored after exiting all scopes")


if __name__ == "__main__":
    main()
