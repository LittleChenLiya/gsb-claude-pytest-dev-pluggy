"""Demo: atomic batch registration with ``PluginManager.register_many``.

Run with: python scripts/demo_register_many.py

Shows the plugin list and hook order before/after publishing a batch, how a
validation failure rejects the whole batch (naming the offending plugin),
and how a historic-replay exception propagates while the batch stays
registered.
"""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pluggy import HookimplMarker  # noqa: E402
from pluggy import HookspecMarker  # noqa: E402
from pluggy import PluginManager  # noqa: E402
from pluggy import PluginValidationError  # noqa: E402


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Hooks:
    @hookspec(historic=True)
    def myhook(self, arg):
        pass


def show_state(pm, label):
    names = [name for name, plugin in pm.list_name_plugin() if plugin is not None]
    order = [impl.plugin_name for impl in pm.hook.myhook.get_hookimpls()]
    print(f"{label}: plugins={names} hook-order={order}")


def main():
    calls = []

    class OldPlugin:
        @hookimpl
        def myhook(self, arg):
            calls.append(("old", arg))

    class GoodA:
        @hookimpl(tryfirst=True)
        def myhook(self, arg):
            calls.append(("good-a", arg))

    class GoodB:
        @hookimpl(trylast=True)
        def myhook(self, arg):
            calls.append(("good-b", arg))

    class BadSig:
        @hookimpl
        def myhook(self, arg, unknownarg):
            pass

    class Raiser:
        @hookimpl
        def myhook(self, arg):
            raise RuntimeError("boom from historic replay")

    pm = PluginManager("example")
    pm.add_hookspecs(Hooks)
    pm.register(OldPlugin(), "old")
    pm.hook.myhook.call_historic(kwargs=dict(arg=1))

    print("== 1. happy path: batch of two valid plugins ==")
    show_state(pm, "before")
    names = pm.register_many([(GoodA(), "good-a"), (GoodB(), "good-b")])
    print(f"registered: {names}")
    show_state(pm, "after ")
    print(f"historic replays so far: {calls}")

    print()
    print("== 2. validation failure: second plugin has a bad signature ==")
    try:
        pm.register_many([(GoodA(), "another-a"), (BadSig(), "bad-sig")])
    except PluginValidationError as exc:
        print(f"rejected, offending plugin: {exc.plugin!r}")
        print(f"error: {exc}")
    show_state(pm, "after ")

    print()
    print("== 3. historic replay raises: batch stays registered ==")
    try:
        pm.register_many([(Raiser(), "raiser"), (GoodB(), "late-b")])
    except RuntimeError as exc:
        print(f"propagated: {exc!r}")
    show_state(pm, "after ")
    for name in ("raiser", "late-b"):
        pm.unregister(name=name)
        print(f"unregistered {name}")
    show_state(pm, "final ")


if __name__ == "__main__":
    main()
