"""Demo of locally blocking plugins with PluginManager.without_plugins().

Run it with:

    uv run python docs/examples/without_plugins.py

It prints the hookimpl call order observed in each blocking scope.
"""

import pluggy


hookspec = pluggy.HookspecMarker("demo")
hookimpl = pluggy.HookimplMarker("demo")

calls = []


class Spec:
    @hookspec
    def render(self, arg):
        """A demo hook."""


class PluginA:
    @hookimpl
    def render(self, arg):
        calls.append("A")
        return "a"


class PluginB:
    @hookimpl
    def render(self, arg):
        calls.append("B")
        return "b"


class PluginW:
    @hookimpl(wrapper=True)
    def render(self, arg):
        calls.append("W-setup")
        try:
            result = yield
        finally:
            calls.append("W-teardown")
        return result


pm = pluggy.PluginManager("demo")
pm.add_hookspecs(Spec)
pm.register(PluginA(), "A")
pm.register(PluginB(), "B")
pm.register(PluginW(), "W")


def show(label):
    calls.clear()
    pm.hook.render(arg=0)
    print(f"{label:<26} -> {' -> '.join(calls)}")


show("no blocking")
with pm.without_plugins({"B"}):
    show("block B")
    with pm.without_plugins({"A"}):
        show("nested: block A and B")
    show("back to block B")
show("restored after exit")
