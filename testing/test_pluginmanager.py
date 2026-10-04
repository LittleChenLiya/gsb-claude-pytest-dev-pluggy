"""
``PluginManager`` unit and public API testing.
"""

import importlib.metadata
import types
from typing import Any
from typing import cast

import pytest

from pluggy import HookCallError
from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy import PluginValidationError


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


def test_plugin_double_register(pm: PluginManager) -> None:
    """Registering the same plugin more then once isn't allowed"""
    pm.register(42, name="abc")
    with pytest.raises(ValueError):
        pm.register(42, name="abc")
    with pytest.raises(ValueError):
        pm.register(42, name="def")


def test_pm(pm: PluginManager) -> None:
    """Basic registration with objects"""

    class A:
        pass

    a1, a2 = A(), A()
    pm.register(a1)
    assert pm.is_registered(a1)
    pm.register(a2, "hello")
    assert pm.is_registered(a2)
    out = pm.get_plugins()
    assert a1 in out
    assert a2 in out
    assert pm.get_plugin("hello") == a2
    assert pm.unregister(a1) == a1
    assert not pm.is_registered(a1)

    out2 = pm.list_name_plugin()
    assert len(out2) == 1
    assert out2 == [("hello", a2)]


def test_has_plugin(pm: PluginManager) -> None:
    class A:
        pass

    a1 = A()
    pm.register(a1, "hello")
    assert pm.is_registered(a1)
    assert pm.has_plugin("hello")


def test_register_dynamic_attr(he_pm: PluginManager) -> None:
    class A:
        def __getattr__(self, name):
            if name[0] != "_":
                return 42
            raise AttributeError(name)

    a = A()
    _ = a.test

    he_pm.register(a)
    assert not he_pm.get_hookcallers(a)


def test_pm_name(pm: PluginManager) -> None:
    class A:
        pass

    a1 = A()
    name = pm.register(a1, name="hello")
    assert name == "hello"
    pm.unregister(a1)
    assert pm.get_plugin("hello") is None
    assert not pm.is_registered(a1)
    assert not pm.get_plugins()
    name2 = pm.register(a1, name="hello")
    assert name2 == name
    pm.unregister(name="hello")
    assert pm.get_plugin("hello") is None
    assert not pm.is_registered(a1)
    assert not pm.get_plugins()


def test_set_blocked(pm: PluginManager) -> None:
    class A:
        pass

    a1 = A()
    name = pm.register(a1)
    assert name is not None
    assert pm.is_registered(a1)
    assert not pm.is_blocked(name)
    assert pm.get_plugins() == {a1}

    pm.set_blocked(name)
    assert pm.is_blocked(name)
    assert not pm.is_registered(a1)
    assert pm.get_plugins() == set()

    pm.set_blocked("somename")
    assert pm.is_blocked("somename")
    assert not pm.register(A(), "somename")
    pm.unregister(name="somename")
    assert pm.is_blocked("somename")
    assert pm.get_plugins() == set()

    # Unblock.
    assert not pm.unblock("someothername")
    assert pm.unblock("somename")
    assert not pm.is_blocked("somename")
    assert not pm.unblock("somename")
    assert pm.register(A(), "somename")


def test_register_ignores_properties(he_pm: PluginManager) -> None:
    class ClassWithProperties:
        property_was_executed: bool = False

        @property
        def some_func(self):
            self.property_was_executed = True  # pragma: no cover

    # Registering the class is harmless (getattr returns the property object).
    he_pm.register(ClassWithProperties)
    # Registering an instance must not evaluate the property getter.
    test_plugin = ClassWithProperties()
    he_pm.register(test_plugin)
    assert not test_plugin.property_was_executed


def test_register_ignores_cached_property(he_pm: PluginManager) -> None:
    from functools import cached_property

    class ClassWithCachedProperty:
        cached_was_executed: bool = False

        @cached_property
        def some_func(self) -> None:
            self.cached_was_executed = True  # pragma: no cover

    test_plugin = ClassWithCachedProperty()
    he_pm.register(test_plugin)
    assert not test_plugin.cached_was_executed
    assert "some_func" not in test_plugin.__dict__


def test_register_ignores_raising_descriptors(he_pm: PluginManager) -> None:
    """Descriptor attrs are skipped without accessing them via getattr."""

    class RaisingDescriptor:
        def __get__(self, obj: object, owner: type | None = None) -> object:
            raise AttributeError("descriptor access failed")

    class PluginWithRaisingDescriptor:
        weird_attr = RaisingDescriptor()

        @hookimpl
        def he_method1(self, arg: object) -> list[object]:
            return [arg]

    plugin = PluginWithRaisingDescriptor()
    assert "weird_attr" in dir(plugin)
    with pytest.raises(AttributeError):
        _ = plugin.weird_attr

    he_pm.register(plugin)
    assert he_pm.hook.he_method1(arg=1) == [[1]]


def test_register_hookimpl_above_classmethod(he_pm: PluginManager) -> None:
    """@hookimpl applied above @classmethod is discoverable via static lookup."""

    class Plugin:
        @hookimpl
        @classmethod
        def he_method1(cls, arg: object) -> list[object]:
            return [arg]

    he_pm.register(Plugin())
    assert he_pm.hook.he_method1(arg=1) == [[1]]


def test_register_hookimpl_above_staticmethod(he_pm: PluginManager) -> None:
    """@hookimpl applied above @staticmethod is discoverable via static lookup."""

    class Plugin:
        @hookimpl
        @staticmethod
        def he_method1(arg: object) -> list[object]:
            return [arg]

    he_pm.register(Plugin())
    assert he_pm.hook.he_method1(arg=1) == [[1]]


def test_register_instance_attribute_function_is_not_bound(
    he_pm: PluginManager,
) -> None:
    """A function stored on the instance is called unbound, as ``getattr`` does."""

    class Plugin:
        def __init__(self) -> None:
            @hookimpl
            def he_method1(arg: object) -> list[object]:
                return [arg]

            # Instance attributes bypass the descriptor protocol, so no ``self``
            # is injected and the hook must keep its declared signature.
            self.he_method1 = he_method1

    he_pm.register(Plugin())
    assert he_pm.hook.he_method1(arg=1) == [[1]]


def test_register_instance_attribute_bound_method(he_pm: PluginManager) -> None:
    """A bound method stored on the instance stays bound to its own object."""

    class Impl:
        @hookimpl
        def he_method1(self, arg: object) -> list[object]:
            return [self, arg]

    impl = Impl()

    class Plugin:
        pass

    plugin = Plugin()
    plugin.he_method1 = impl.he_method1  # type: ignore[attr-defined]

    he_pm.register(plugin)
    assert he_pm.hook.he_method1(arg=1) == [[impl, 1]]


def test_register_slot_attribute_function_is_not_bound(he_pm: PluginManager) -> None:
    """A function stored in a ``__slots__`` slot is found and called unbound."""

    class Plugin:
        __slots__ = ("he_method1", "unset")

        def __init__(self) -> None:
            @hookimpl
            def he_method1(arg: object) -> list[object]:
                return [arg]

            self.he_method1 = he_method1

    plugin = Plugin()
    # ``dir()`` lists ``unset`` too; an unassigned slot must be skipped quietly.
    assert "unset" in dir(plugin)
    he_pm.register(plugin)
    assert he_pm.hook.he_method1(arg=1) == [[1]]


def test_register_module_level_bound_method(he_pm: PluginManager) -> None:
    """A bound method assigned onto a module namespace is hookable."""

    class Impl:
        @hookimpl
        def he_method1(self, arg: object) -> list[object]:
            return [self, arg]

    impl = Impl()
    module = types.ModuleType("module_plugin")
    module.he_method1 = impl.he_method1  # type: ignore[attr-defined]

    he_pm.register(module)
    assert he_pm.hook.he_method1(arg=1) == [[impl, 1]]


def test_register_ignores_unmarked_class_and_static_methods(
    he_pm: PluginManager,
) -> None:
    """Undecorated classmethods/staticmethods carry no marker on either side."""

    class Plugin:
        @classmethod
        def not_a_hook_cm(cls) -> None: ...

        @staticmethod
        def not_a_hook_sm() -> None: ...

        @hookimpl
        def he_method1(self, arg: object) -> list[object]:
            return [arg]

    plugin = Plugin()
    assert he_pm.parse_hookimpl_opts(plugin, "not_a_hook_cm") is None
    assert he_pm.parse_hookimpl_opts(plugin, "not_a_hook_sm") is None

    he_pm.register(plugin)
    assert he_pm.hook.he_method1(arg=1) == [[1]]


def test_register_ignores_non_callable_instance_attributes(pm: PluginManager) -> None:
    """Instance attributes that are not routines are never hook candidates."""

    class Plugin:
        def __init__(self) -> None:
            self.some_value = 42

    assert pm.parse_hookimpl_opts(Plugin(), "some_value") is None


def test_hookspec_lookup_ignores_properties(pm: PluginManager) -> None:
    """Hookspec discovery must not evaluate descriptors either."""

    class Spec:
        was_executed = False

        @property
        def not_a_spec(self) -> None:
            type(self).was_executed = True  # pragma: no cover

        @hookspec
        def he_method1(self, arg: object) -> object: ...

    # Spec instances are supported at runtime, see the ``he_pm`` fixture.
    spec = cast(Any, Spec())
    pm.add_hookspecs(spec)
    assert not Spec.was_executed
    assert pm.parse_hookspec_opts(spec, "not_a_spec") is None


def test_register_mismatch_method(he_pm: PluginManager) -> None:
    class hello:
        @hookimpl
        def he_method_notexists(self):
            pass  # pragma: no cover

    plugin = hello()

    he_pm.register(plugin)
    with pytest.raises(PluginValidationError) as excinfo:
        he_pm.check_pending()
    assert excinfo.value.plugin is plugin


def test_register_mismatch_arg(he_pm: PluginManager) -> None:
    class hello:
        @hookimpl
        def he_method1(self, qlwkje):
            pass  # pragma: no cover

    plugin = hello()

    with pytest.raises(PluginValidationError) as excinfo:
        he_pm.register(plugin)
    assert excinfo.value.plugin is plugin


def test_register_hookwrapper_not_a_generator_function(he_pm: PluginManager) -> None:
    class hello:
        @hookimpl(hookwrapper=True)
        def he_method1(self):
            pass  # pragma: no cover

    plugin = hello()

    with pytest.raises(PluginValidationError, match="generator function") as excinfo:
        he_pm.register(plugin)
    assert excinfo.value.plugin is plugin


def test_register_both_wrapper_and_hookwrapper(he_pm: PluginManager) -> None:
    class hello:
        @hookimpl(wrapper=True, hookwrapper=True)
        def he_method1(self):
            yield  # pragma: no cover

    plugin = hello()

    with pytest.raises(
        PluginValidationError, match="wrapper.*hookwrapper.*mutually exclusive"
    ) as excinfo:
        he_pm.register(plugin)
    assert excinfo.value.plugin is plugin


def test_register(pm: PluginManager) -> None:
    class MyPlugin:
        @hookimpl
        def he_method1(self): ...  # pragma: no cover

    my = MyPlugin()
    pm.register(my)
    assert pm.get_plugins() == {my}
    my2 = MyPlugin()
    pm.register(my2)
    assert pm.get_plugins() == {my, my2}

    assert pm.is_registered(my)
    assert pm.is_registered(my2)
    pm.unregister(my)
    assert not pm.is_registered(my)
    assert pm.get_plugins() == {my2}

    with pytest.raises(AssertionError, match=r"not registered"):
        pm.unregister(my)


def test_unregister_blocked(pm: PluginManager) -> None:
    class Plugin:
        pass

    p = Plugin()
    pm.set_blocked("error")
    pm.register(p, "error")
    # bloked plugins can be unregistred many times atm
    pm.unregister(p, "error")
    pm.unregister(p, "error")


def test_register_unknown_hooks(pm: PluginManager) -> None:
    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            return arg + 1

    pname = pm.register(Plugin1())
    assert pname is not None

    class Hooks:
        @hookspec
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    # assert not pm._unverified_hooks
    assert pm.hook.he_method1(arg=1) == [2]
    hookcallers = pm.get_hookcallers(pm.get_plugin(pname))
    assert hookcallers is not None
    assert len(hookcallers) == 1


def test_register_historic(pm: PluginManager) -> None:
    class Hooks:
        @hookspec(historic=True)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)

    pm.hook.he_method1.call_historic(kwargs={"arg": 1})
    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    assert out == [1]

    class Plugin2:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg * 10)

    pm.register(Plugin2())
    assert out == [1, 10]
    pm.hook.he_method1.call_historic(kwargs={"arg": 12})
    assert out == [1, 10, 120, 12]


# --- historic history window tests -----------------------------------------


def test_historic_history_window_replays_retained(pm: PluginManager) -> None:
    """With a history window, a late plugin only replays retained records."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    for i in (1, 2, 3):
        pm.hook.he_method1.call_historic(kwargs={"arg": i})

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    # Capacity 2: only records 2 and 3 are replayed, in sequence.
    assert out == [2, 3]


def test_historic_history_window_strict_registration_rejects_gap(
    pm: PluginManager,
) -> None:
    """Strict registration rejects before calling the plugin when records were
    trimmed."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    for i in (1, 2, 3):
        pm.hook.he_method1.call_historic(kwargs={"arg": i})

    calls = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            calls.append(arg)

    with pytest.raises(PluginValidationError, match=r"record\(s\) 1-1"):
        pm.register(Plugin(), strict_history=True)
    # The plugin must not have been called at all.
    assert calls == []


def test_historic_history_window_strict_registration_allows_no_gap(
    pm: PluginManager,
) -> None:
    """Strict registration succeeds when nothing was trimmed."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    pm.hook.he_method1.call_historic(kwargs={"arg": 1})
    pm.hook.he_method1.call_historic(kwargs={"arg": 2})

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin(), strict_history=True)
    assert out == [1, 2]


def test_historic_history_window_snapshot_kwargs(pm: PluginManager) -> None:
    """A caller-provided snapshot freezes the recorded kwargs."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    kwargs = {"arg": 1}
    pm.hook.he_method1.call_historic(kwargs=kwargs, snapshot=dict)
    kwargs["arg"] = 999

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    # The replay uses the snapshot, not the later-mutated dict.
    assert out == [1]


def test_historic_history_window_without_snapshot_keeps_reference(
    pm: PluginManager,
) -> None:
    """Without a snapshot the recorded kwargs dict is referenced as before."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    kwargs = {"arg": 1}
    pm.hook.he_method1.call_historic(kwargs=kwargs)
    kwargs["arg"] = 999

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    assert out == [999]


def test_historic_history_window_trim_does_not_reinvoke_callback(
    pm: PluginManager,
) -> None:
    """Trimming records does not re-invoke their result_callback."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    results = []

    def callback(res) -> None:
        results.append(res)

    for i in (1, 2, 3):
        pm.hook.he_method1.call_historic(result_callback=callback, kwargs={"arg": i})

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            return arg * 10

    pm.register(Plugin())
    # Only retained records (2, 3) are replayed; record 1's callback is not
    # invoked by the trim.
    assert results == [20, 30]


def test_historic_history_window_nested_registration(pm: PluginManager) -> None:
    """A plugin registered during history replay still gets the retained window."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    for i in (1, 2, 3):
        pm.hook.he_method1.call_historic(kwargs={"arg": i})

    out = []

    class Nested:
        @hookimpl
        def he_method1(self, arg):
            out.append(("nested", arg))

    class Plugin:
        def __init__(self) -> None:
            self.registered = False

        @hookimpl
        def he_method1(self, arg):
            out.append(("plugin", arg))
            if not self.registered:
                pm.register(Nested())
                self.registered = True

    pm.register(Plugin())
    # Both the plugin and the nested plugin receive only the retained window.
    assert out == [
        ("plugin", 2),
        ("nested", 2),
        ("nested", 3),
        ("plugin", 3),
    ]


@pytest.mark.parametrize(
    ("size", "calls", "expected"),
    [
        (1, 3, [3]),  # only the most recent record
        (2, 3, [2, 3]),  # window of 2
        (3, 3, [1, 2, 3]),  # exact window, no gap
        (3, 4, [2, 3, 4]),  # one record trimmed
        (2, 2, [1, 2]),  # boundary: exactly capacity
    ],
)
def test_historic_history_window_boundary(
    pm: PluginManager, size: int, calls: int, expected: list[int]
) -> None:
    """Boundary conditions for the history window size."""

    class Hooks:
        @hookspec(historic=True, history_size=size)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    for i in range(1, calls + 1):
        pm.hook.he_method1.call_historic(kwargs={"arg": i})

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    assert out == expected


def test_historic_history_window_reports_trimmed_range(
    pm: PluginManager,
) -> None:
    """Trimming reports the trimmed sequence range via ``history_trimmed``."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    hook = pm.hook.he_method1

    def get_trimmed() -> tuple[int, int] | None:
        return hook.history_trimmed

    assert hook.history_size == 2
    assert get_trimmed() is None
    hook.call_historic(kwargs={"arg": 1})
    hook.call_historic(kwargs={"arg": 2})
    hook.call_historic(kwargs={"arg": 3})
    assert get_trimmed() == (1, 1)
    hook.call_historic(kwargs={"arg": 4})
    assert get_trimmed() == (1, 2)


def test_historic_history_window_with_subset_caller(pm: PluginManager) -> None:
    """The history window works through a subset hook caller."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)

    class Blocker:
        @hookimpl
        def he_method1(self, arg):
            pass

    blocker = Blocker()
    pm.register(blocker)

    shc = pm.subset_hook_caller("he_method1", remove_plugins=[blocker])
    for i in (1, 2, 3):
        shc.call_historic(kwargs={"arg": i})

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    assert out == [2, 3]


def test_historic_history_backward_compat_reset(pm: PluginManager) -> None:
    """Setting ``hook._call_history = []`` (as pytest does to reset a historic
    hook) clears the retained records and restarts sequence/gap tracking."""

    class Hooks:
        @hookspec(historic=True, history_size=2)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)
    hook = pm.hook.he_method1

    def get_trimmed() -> tuple[int, int] | None:
        return hook.history_trimmed

    for i in (1, 2, 3):
        hook.call_historic(kwargs={"arg": i})
    assert get_trimmed() == (1, 1)

    # Legacy reset, as performed by pytest's own unconfigure.
    hook._call_history = []
    assert get_trimmed() is None
    assert hook.history_size == 2

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    pm.register(Plugin())
    # After the reset, no records are replayed.
    assert out == []
    # ...and a fresh call starts a new sequence.
    hook.call_historic(kwargs={"arg": 9})
    assert hook.history_trimmed is None


def test_historic_history_size_validation() -> None:
    """``history_size`` must be a positive integer or ``None``."""

    def dummy() -> None: ...

    bad_values: tuple[Any, ...] = (0, -1, "2", 2.0, True)
    for bad in bad_values:
        decorator = hookspec(historic=True, history_size=bad)
        with pytest.raises(ValueError, match="history_size"):
            decorator(dummy)
    # Valid values are accepted.
    hookspec(historic=True, history_size=1)(dummy)
    hookspec(historic=True, history_size=None)(dummy)
    hookspec(historic=True)(dummy)


def test_historic_with_subset_hook_caller(pm: PluginManager) -> None:
    class Hooks:
        @hookspec(historic=True)
        def he_method1(self, arg): ...

    pm.add_hookspecs(Hooks)

    out = []

    class Plugin:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    plugin = Plugin()
    pm.register(plugin)

    class Plugin2:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg * 10)

    shc = pm.subset_hook_caller("he_method1", remove_plugins=[plugin])
    shc.call_historic(kwargs={"arg": 1})

    pm.register(Plugin2())
    assert out == [10]

    pm.register(Plugin())
    assert out == [10, 1]


@pytest.mark.parametrize("result_callback", [True, False])
def test_with_result_memorized(pm: PluginManager, result_callback: bool) -> None:
    """Verify that ``HookCaller._maybe_apply_history()`
    correctly applies the ``result_callback`` function, when provided,
    to the result from calling each newly registered hook.
    """
    out = []
    if not result_callback:
        callback = None
    else:

        def callback(res) -> None:
            out.append(res)

    class Hooks:
        @hookspec(historic=True)
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            return arg * 10

    pm.register(Plugin1())

    he_method1 = pm.hook.he_method1
    he_method1.call_historic(result_callback=callback, kwargs={"arg": 1})

    class Plugin2:
        @hookimpl
        def he_method1(self, arg):
            return arg * 10

    pm.register(Plugin2())
    if result_callback:
        assert out == [10, 10]
    else:
        assert out == []


def test_with_callbacks_immediately_executed(pm: PluginManager) -> None:
    class Hooks:
        @hookspec(historic=True)
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            return arg * 10

    class Plugin2:
        @hookimpl
        def he_method1(self, arg):
            return arg * 20

    class Plugin3:
        @hookimpl
        def he_method1(self, arg):
            return arg * 30

    out = []
    pm.register(Plugin1())
    pm.register(Plugin2())

    he_method1 = pm.hook.he_method1
    he_method1.call_historic(lambda res: out.append(res), {"arg": 1})
    assert out == [20, 10]
    pm.register(Plugin3())
    assert out == [20, 10, 30]


def test_register_historic_incompat_hookwrapper(pm: PluginManager) -> None:
    class Hooks:
        @hookspec(historic=True)
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    out = []

    class Plugin:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            out.append(arg)  # pragma: no cover

    with pytest.raises(PluginValidationError):
        pm.register(Plugin())


def test_register_historic_incompat_wrapper(pm: PluginManager) -> None:
    class Hooks:
        @hookspec(historic=True)
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    class Plugin:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield  # pragma: no cover

    with pytest.raises(PluginValidationError):
        pm.register(Plugin())


def test_call_extra(pm: PluginManager) -> None:
    class Hooks:
        @hookspec
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    def he_method1(arg):
        return arg * 10

    out = pm.hook.he_method1.call_extra([he_method1], {"arg": 1})
    assert out == [10]


def test_call_with_too_few_args(pm: PluginManager) -> None:
    class Hooks:
        @hookspec
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            raise ZeroDivisionError

    pm.register(Plugin1())
    with pytest.raises(ZeroDivisionError):
        pm.hook.he_method1(arg="works")

    with pytest.raises(HookCallError), pytest.warns(UserWarning):
        pm.hook.he_method1()


def test_subset_hook_caller(pm: PluginManager) -> None:
    class Hooks:
        @hookspec
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)

    out = []

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg)

    class Plugin2:
        @hookimpl
        def he_method1(self, arg):
            out.append(arg * 10)

    class PluginNo:
        pass

    plugin1, plugin2, plugin3 = Plugin1(), Plugin2(), PluginNo()
    pm.register(plugin1)
    pm.register(plugin2)
    pm.register(plugin3)
    pm.hook.he_method1(arg=1)
    assert out == [10, 1]
    out[:] = []

    hc = pm.subset_hook_caller("he_method1", [plugin1])
    hc(arg=2)
    assert out == [20]
    out[:] = []

    hc = pm.subset_hook_caller("he_method1", [plugin2])
    hc(arg=2)
    assert out == [2]
    out[:] = []

    pm.unregister(plugin1)
    hc(arg=2)
    assert out == []
    out[:] = []

    pm.hook.he_method1(arg=1)
    assert out == [10]

    assert repr(hc) == "<_SubsetHookCaller 'he_method1'>"


def test_subset_hook_caller_with_specname(pm: PluginManager) -> None:
    """Test that subset_hook_caller doesn't get confused by specname."""

    class Hooks:
        @hookspec
        def he_method1(self):
            pass

        @hookspec
        def he_method2(self):
            pass

    pm.add_hookspecs(Hooks)

    class Plugin1:
        @hookimpl(specname="he_method1")
        def alias(self):
            return 1  # pragma: no cover

    class Plugin2:
        @hookimpl
        def he_method1(self):
            return 2

    class Plugin3:
        @hookimpl(specname="he_method2")
        def he_method1(self):
            return 3  # pragma: no cover

    plugin1, plugin2, plugin3 = Plugin1(), Plugin2(), Plugin3()
    pm.register(plugin1)
    pm.register(plugin2)
    pm.register(plugin3)

    hc = pm.subset_hook_caller("he_method1", remove_plugins=[plugin1])
    assert hc() == [2]


def test_get_hookimpls(pm: PluginManager) -> None:
    class Hooks:
        @hookspec
        def he_method1(self, arg):
            pass

    pm.add_hookspecs(Hooks)
    assert pm.hook.he_method1.get_hookimpls() == []

    class Plugin1:
        @hookimpl
        def he_method1(self, arg):
            pass

    class Plugin2:
        @hookimpl
        def he_method1(self, arg):
            pass

    class PluginNo:
        pass

    plugin1, plugin2, plugin3 = Plugin1(), Plugin2(), PluginNo()
    pm.register(plugin1)
    pm.register(plugin2)
    pm.register(plugin3)

    hookimpls = pm.hook.he_method1.get_hookimpls()
    hook_plugins = [item.plugin for item in hookimpls]
    assert hook_plugins == [plugin1, plugin2]


def test_get_hookcallers(pm: PluginManager) -> None:
    class Hooks:
        @hookspec
        def he_method1(self): ...

        @hookspec
        def he_method2(self): ...

    pm.add_hookspecs(Hooks)

    class Plugin1:
        @hookimpl
        def he_method1(self): ...

        @hookimpl
        def he_method2(self): ...

    class Plugin2:
        @hookimpl
        def he_method1(self): ...

    class Plugin3:
        @hookimpl
        def he_method2(self): ...

    plugin1 = Plugin1()
    pm.register(plugin1)
    plugin2 = Plugin2()
    pm.register(plugin2)
    plugin3 = Plugin3()
    pm.register(plugin3)

    hookcallers1 = pm.get_hookcallers(plugin1)
    assert hookcallers1 is not None
    assert len(hookcallers1) == 2
    hookcallers2 = pm.get_hookcallers(plugin2)
    assert hookcallers2 is not None
    assert len(hookcallers2) == 1
    hookcallers3 = pm.get_hookcallers(plugin3)
    assert hookcallers3 is not None
    assert len(hookcallers3) == 1
    assert hookcallers1 == hookcallers2 + hookcallers3

    assert pm.get_hookcallers(object()) is None


def test_add_hookspecs_nohooks(pm: PluginManager) -> None:
    class NoHooks:
        pass

    with pytest.raises(ValueError):
        pm.add_hookspecs(NoHooks)


def test_load_setuptools_instantiation(
    monkeypatch: pytest.MonkeyPatch, pm: PluginManager
) -> None:
    class EntryPoint:
        name = "myname"
        group = "hello"
        value = "myname:foo"

        def load(self):
            class PseudoPlugin:
                x = 42

            return PseudoPlugin()

    class Distribution:
        entry_points = (EntryPoint(),)

    # Cast mock Distribution to satisfy mypy type checking
    dist = cast(importlib.metadata.Distribution, Distribution())

    def my_distributions():
        return (dist,)

    monkeypatch.setattr(importlib.metadata, "distributions", my_distributions)
    num = pm.load_setuptools_entrypoints("hello")
    assert num == 1
    plugin = pm.get_plugin("myname")
    assert plugin is not None
    assert plugin.x == 42
    ret = pm.list_plugin_distinfo()
    # poor man's `assert ret == [(plugin, mock.ANY)]`
    assert len(ret) == 1
    assert len(ret[0]) == 2
    assert ret[0][0] == plugin
    assert ret[0][1]._dist == dist
    num = pm.load_setuptools_entrypoints("hello")
    assert num == 0  # no plugin loaded by this call

    ret_distributions = pm.list_plugin_distributions()
    assert ret_distributions == [(plugin, dist)]


def test_add_tracefuncs(he_pm: PluginManager) -> None:
    out: list[Any] = []

    class api1:
        @hookimpl
        def he_method1(self):
            out.append("he_method1-api1")

    class api2:
        @hookimpl
        def he_method1(self):
            out.append("he_method1-api2")

    he_pm.register(api1())
    he_pm.register(api2())

    def before(hook_name, hook_impls, kwargs):
        out.append((hook_name, list(hook_impls), kwargs))

    def after(outcome, hook_name, hook_impls, kwargs):
        out.append((outcome, hook_name, list(hook_impls), kwargs))

    undo = he_pm.add_hookcall_monitoring(before, after)

    he_pm.hook.he_method1(arg=1)
    assert len(out) == 4
    assert out[0][0] == "he_method1"
    assert len(out[0][1]) == 2
    assert isinstance(out[0][2], dict)
    assert out[1] == "he_method1-api2"
    assert out[2] == "he_method1-api1"
    assert len(out[3]) == 4
    assert out[3][1] == out[0][0]

    undo()
    he_pm.hook.he_method1(arg=1)
    assert len(out) == 4 + 2


def test_hook_tracing(he_pm: PluginManager) -> None:
    saveindent = []

    class api1:
        @hookimpl
        def he_method1(self):
            saveindent.append(he_pm.trace.root.indent)

    class api2:
        @hookimpl
        def he_method1(self):
            saveindent.append(he_pm.trace.root.indent)
            raise ValueError()

    he_pm.register(api1())
    out: list[Any] = []
    he_pm.trace.root.setwriter(out.append)
    undo = he_pm.enable_tracing()
    try:
        indent = he_pm.trace.root.indent
        he_pm.hook.he_method1(arg=1)
        assert indent == he_pm.trace.root.indent
        assert len(out) == 2
        assert "he_method1" in out[0]
        assert "finish" in out[1]

        out[:] = []
        he_pm.register(api2())

        with pytest.raises(ValueError):
            he_pm.hook.he_method1(arg=1)
        assert he_pm.trace.root.indent == indent
        assert saveindent[0] > indent
    finally:
        undo()


@pytest.mark.parametrize("historic", [False, True])
def test_register_while_calling(
    pm: PluginManager,
    historic: bool,
) -> None:
    """Test that registering an impl of a hook while it is being called does
    not affect the call itself, only later calls.

    For historic hooks however, the hook is called immediately on registration.

    Regression test for #438.
    """
    hookspec = HookspecMarker("example")

    class Hooks:
        @hookspec(historic=historic)
        def configure(self) -> int:
            raise NotImplementedError()

    class Plugin1:
        @hookimpl
        def configure(self) -> int:
            return 1

    class Plugin2:
        def __init__(self) -> None:
            self.already_registered = False

        @hookimpl
        def configure(self) -> int:
            if not self.already_registered:
                pm.register(Plugin4())
                pm.register(Plugin5())
                pm.register(Plugin6())
                self.already_registered = True
            return 2

    class Plugin3:
        @hookimpl
        def configure(self) -> int:
            return 3

    class Plugin4:
        @hookimpl(tryfirst=True)
        def configure(self) -> int:
            return 4

    class Plugin5:
        @hookimpl()
        def configure(self) -> int:
            return 5

    class Plugin6:
        @hookimpl(trylast=True)
        def configure(self) -> int:
            return 6

    pm.add_hookspecs(Hooks)
    pm.register(Plugin1())
    pm.register(Plugin2())
    pm.register(Plugin3())

    if not historic:
        result = pm.hook.configure()
        assert result == [3, 2, 1]
        result = pm.hook.configure()
        assert result == [4, 5, 3, 2, 1, 6]
    else:
        result = []
        pm.hook.configure.call_historic(result.append)
        assert result == [4, 5, 6, 3, 2, 1]
        result = []
        pm.hook.configure.call_historic(result.append)
        assert result == [4, 5, 3, 2, 1, 6]


def test_check_pending_skips_underscore(pm: PluginManager) -> None:
    # todo: determine what we want to do with the namespace
    class Plugin:
        @hookimpl
        def _problem(self):
            pass

    pm.register(Plugin())
    pm.hook._problem()
    pm.check_pending()


def test_check_pending_optionalhook(
    pm: PluginManager,
) -> None:
    class Plugin:
        @hookimpl(optionalhook=True)
        def a_hook(self, param):
            pass

    pm.register(Plugin())
    pm.hook.a_hook(param=1)
    pm.check_pending()


def test_check_pending_nonspec_hook(
    pm: PluginManager,
) -> None:
    hookimpl = HookimplMarker("example")

    class Plugin:
        @hookimpl
        def a_hook(self, param):
            pass

    pm.register(Plugin())
    with pytest.raises(HookCallError, match="hook call must provide argument 'param'"):
        pm.hook.a_hook()

    with pytest.raises(
        PluginValidationError, match="unknown hook 'a_hook' in plugin .*"
    ):
        pm.check_pending()


def test_unregister_plugin_with_multi_hookimpls(pm: PluginManager) -> None:
    """Verify that unregistering a plugin with multiple hookimpls on the
    same hook (via specname) removes all of them (#431)."""

    class Api:
        @hookspec
        def hello(self, arg: object) -> object: ...

    pm.add_hookspecs(Api)

    class Plugin:
        @hookimpl
        def hello(self, arg: object) -> int:
            return arg + 1  # type: ignore[operator]

        @hookimpl(specname="hello")
        def hello_again(self, arg: object) -> int:
            return arg + 100  # type: ignore[operator]

    plugin = Plugin()
    pm.register(plugin)

    # Both implementations should be registered.
    impls = pm.hook.hello.get_hookimpls()
    assert len(impls) == 2

    # Both implementations should run.
    out = pm.hook.hello(arg=3)
    assert sorted(out) == [4, 103]

    # After unregister, no implementations should remain.
    pm.unregister(plugin)
    assert pm.hook.hello(arg=3) == []
    assert pm.hook.hello.get_hookimpls() == []


def test_get_hookcallers_no_duplicates(pm: PluginManager) -> None:
    """Verify that get_hookcallers does not return duplicate HookCaller
    entries when a plugin has multiple hookimpls on the same hook (#431)."""

    class Api:
        @hookspec
        def hello(self, arg: object) -> object: ...

        @hookspec
        def goodbye(self, arg: object) -> object: ...

    pm.add_hookspecs(Api)

    class Plugin:
        @hookimpl
        def hello(self, arg: object) -> int:
            return arg + 1  # type: ignore[operator]

        @hookimpl(specname="hello")
        def hello_again(self, arg: object) -> int:
            return arg + 100  # type: ignore[operator]

        @hookimpl
        def goodbye(self, arg: object) -> int:
            return arg + 200  # type: ignore[operator]

    plugin = Plugin()
    pm.register(plugin)

    hookcallers = pm.get_hookcallers(plugin)
    assert hookcallers is not None
    # Should return 2 unique callers (hello + goodbye), not 3.
    assert len(hookcallers) == 2
    caller_names = {hc.name for hc in hookcallers}
    assert caller_names == {"hello", "goodbye"}
    # Both hello impls are still wired up, and each caller works.
    assert pm.hook.hello(arg=1) == [101, 2]
    assert pm.hook.goodbye(arg=1) == [201]
