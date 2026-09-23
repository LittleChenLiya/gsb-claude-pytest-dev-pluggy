"""Demo: atomic batch registration with PluginManager.register_many.

Shows the plugin list and hook order before/after publishing a batch,
how a validation failure rejects the whole batch, and how a historic
replay exception propagates while the published batch stays registered.
"""

from __future__ import annotations

from typing import Any

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy import PluginValidationError


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Hooks:
    @hookspec(historic=True)
    def he_method1(self, arg: int) -> None: ...


def show_state(pm: PluginManager, label: str) -> None:
    names = [name for name, _ in pm.list_name_plugin()]
    order = [impl.plugin_name for impl in pm.hook.he_method1.get_hookimpls()]
    print(f"{label}: plugins={names} hookimpl-order={order}")


def main() -> None:
    pm = PluginManager("example")
    pm.add_hookspecs(Hooks)

    out: list[tuple[str, Any]] = []

    class Old:
        @hookimpl
        def he_method1(self, arg: int) -> None:
            out.append(("old", arg))

    pm.register(Old(), "old")
    pm.hook.he_method1.call_historic(kwargs={"arg": 1})
    show_state(pm, "before batch")

    class P1:
        @hookimpl
        def he_method1(self, arg: int) -> None:
            out.append(("p1", arg))

    class P2:
        @hookimpl(tryfirst=True)
        def he_method1(self, arg: int) -> None:
            out.append(("p2", arg))

    names = pm.register_many([(P1(), "p1"), (P2(), "p2")])
    show_state(pm, f"after batch {names}")
    print("historic replays:", out)

    # A bad signature anywhere rejects the whole batch.
    class Bad:
        @hookimpl
        def he_method1(self, wrongarg: int) -> None: ...

    try:
        pm.register_many([(P1(), "p3"), (Bad(), "bad")])
    except PluginValidationError as e:
        print(f"batch rejected, offending plugin: {e.plugin!r}")
        print(f"  reason: {e}")
    show_state(pm, "after failed batch")

    # A historic replay that raises propagates; the batch stays registered.
    class Boom(Exception):
        pass

    class Exploding:
        @hookimpl
        def he_method1(self, arg: int) -> None:
            raise Boom("replay exploded")

    try:
        pm.register_many([(Exploding(), "exploding")])
    except Boom as e:
        print(f"replay raised: {e!r} (batch kept)")
    show_state(pm, "after replay failure")
    pm.unregister(name="exploding")
    show_state(pm, "after unregister")


if __name__ == "__main__":
    main()
