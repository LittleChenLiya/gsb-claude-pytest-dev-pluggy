from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import inspect
import types
from typing import Any
from typing import cast
from typing import Final
from typing import TYPE_CHECKING
from typing import TypeAlias
import warnings

from . import _tracing
from ._callers import _multicall
from ._hooks import _HookImplFunction
from ._hooks import _is_orderable
from ._hooks import _Namespace
from ._hooks import _Plugin
from ._hooks import _SubsetHookCaller
from ._hooks import HookCaller
from ._hooks import HookImpl
from ._hooks import HookimplOpts
from ._hooks import HookOrderError
from ._hooks import HookRelay
from ._hooks import HookspecOpts
from ._hooks import normalize_hookimpl_opts
from ._result import Result


if TYPE_CHECKING:
    import importlib.metadata

    from ._compat import DistFacade


_BeforeTrace: TypeAlias = Callable[[str, Sequence[HookImpl], Mapping[str, Any]], None]
_AfterTrace: TypeAlias = Callable[
    [Result[Any], str, Sequence[HookImpl], Mapping[str, Any]], None
]


def _warn_for_function(warning: Warning, function: Callable[..., object]) -> None:
    func = cast(types.FunctionType, function)
    warnings.warn_explicit(
        warning,
        type(warning),
        lineno=func.__code__.co_firstlineno,
        filename=func.__code__.co_filename,
    )


_ABSENT: Final = object()


def _instance_dict(obj: object) -> dict[str, Any]:
    """Return ``obj``'s own ``__dict__``, without going through ``getattr``."""
    try:
        instance_dict = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return {}
    return instance_dict if isinstance(instance_dict, dict) else {}


def _static_hook_attr(
    obj: object, name: str
) -> tuple[object, Callable[..., object]] | None:
    """Return ``(marker_holder, function)`` for ``name`` on ``obj``, else None.

    Lookup avoids ``getattr()`` so that properties, ``cached_property`` and
    other descriptors are skipped instead of being evaluated -- hooks provided
    through such a descriptor (or only through ``__getattr__``) are not
    supported. Only plain functions, :class:`classmethod`, :class:`staticmethod`
    and already-bound :class:`~types.MethodType` objects are hookable.

    ``marker_holder`` is the object that may carry the hookimpl/hookspec
    options: the ``classmethod``/``staticmethod`` wrapper when the marker was
    applied above it, the underlying function otherwise. ``function`` is bound
    exactly the way ``getattr()`` would have bound it.
    """
    if not (inspect.isclass(obj) or inspect.ismodule(obj)):
        # Values stored on the instance itself -- in ``__dict__`` or in a
        # ``__slots__`` slot -- are handed out by ``getattr()`` as they are,
        # bypassing the descriptor protocol, so they must not be bound here.
        value = _instance_dict(obj).get(name, _ABSENT)
        if value is _ABSENT:
            slot = inspect.getattr_static(obj, name, None)
            if isinstance(slot, types.MemberDescriptorType):
                # Reading a slot runs no user code.
                try:
                    value = slot.__get__(obj, type(obj))
                except AttributeError:
                    value = _ABSENT
        if value is not _ABSENT:
            if isinstance(value, types.MethodType):
                return value.__func__, value
            if inspect.isfunction(value):
                return value, value
            return None

    static: object = inspect.getattr_static(obj, name, None)
    if isinstance(static, staticmethod):
        return static, static.__func__
    if isinstance(static, classmethod):
        owner = obj if inspect.isclass(obj) else type(obj)
        return static, cast(Callable[..., object], static.__get__(owner, owner))
    if isinstance(static, types.MethodType):
        return static.__func__, static
    if inspect.isfunction(static):
        if inspect.isclass(obj) or inspect.ismodule(obj):
            return static, static
        return static, static.__get__(obj, type(obj))
    return None


def _get_marker_opts(holder: object, attrname: str) -> dict[str, Any] | None:
    """Read marker options from ``holder``, falling back to ``__func__``."""
    opts = getattr(holder, attrname, None)
    if opts is not None:
        return opts if isinstance(opts, dict) else None
    func = getattr(holder, "__func__", None)
    if func is not None:
        opts = getattr(func, attrname, None)
        if opts is not None:
            return opts if isinstance(opts, dict) else None
    return None


class PluginValidationError(Exception):
    """Plugin failed validation.

    :param plugin: The plugin which failed validation.
    :param message: Error message.
    """

    def __init__(self, plugin: _Plugin, message: str) -> None:
        super().__init__(message)
        #: The plugin which failed validation.
        self.plugin = plugin


class PluginManager:
    """Core class which manages registration of plugin objects and 1:N hook
    calling.

    You can register new hooks by calling :meth:`add_hookspecs(module_or_class)
    <PluginManager.add_hookspecs>`.

    You can register plugin objects (which contain hook implementations) by
    calling :meth:`register(plugin) <PluginManager.register>`.

    For debugging purposes you can call :meth:`PluginManager.enable_tracing`
    which will subsequently send debug information to the trace helper.

    :param project_name:
        The short project name. Prefer snake case. Make sure it's unique!
    """

    def __init__(self, project_name: str) -> None:
        #: The project name.
        self.project_name: Final = project_name
        self._name2plugin: Final[dict[str, _Plugin]] = {}
        self._plugin_distinfo: Final[
            list[tuple[_Plugin, importlib.metadata.Distribution]]
        ] = []
        #: The "hook relay", used to call a hook on all registered plugins.
        #: See :ref:`calling`.
        self.hook: Final = HookRelay()
        #: The tracing entry point. See :ref:`tracing`.
        self.trace: Final[_tracing.TagTracerSub] = _tracing.TagTracer().get(
            "pluginmanage"
        )
        self._inner_hookexec = _multicall

    def _hookexec(
        self,
        hook_name: str,
        methods: Sequence[HookImpl],
        kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        # called from all hookcaller instances.
        # enable_tracing will set its own wrapping function at self._inner_hookexec
        return self._inner_hookexec(hook_name, methods, kwargs, firstresult)

    def register(self, plugin: _Plugin, name: str | None = None) -> str | None:
        """Register a plugin and return its name.

        :param name:
            The name under which to register the plugin. If not specified, a
            name is generated using :func:`get_canonical_name`.

        :returns:
            The plugin name. If the name is blocked from registering, returns
            ``None``.

        If the plugin is already registered, raises a :exc:`ValueError`.
        """
        plugin_name = name or self.get_canonical_name(plugin)

        if plugin_name in self._name2plugin:
            if self._name2plugin.get(plugin_name, -1) is None:
                return None  # blocked plugin, return None to indicate no registration
            raise ValueError(
                "Plugin name already registered: "
                f"{plugin_name}={plugin}\n{self._name2plugin}"
            )

        if plugin in self._name2plugin.values():
            raise ValueError(
                "Plugin already registered under a different name: "
                f"{plugin_name}={plugin}\n{self._name2plugin}"
            )

        # XXX if an error happens we should make sure no state has been
        # changed at point of return
        self._name2plugin[plugin_name] = plugin

        # register matching hook implementations of the plugin
        for attr_name in dir(plugin):
            hookimpl_opts = self.parse_hookimpl_opts(plugin, attr_name)
            if hookimpl_opts is not None:
                normalize_hookimpl_opts(hookimpl_opts)
                found = _static_hook_attr(plugin, attr_name)
                # Only reachable when a subclass overrode parse_hookimpl_opts
                # to claim an attribute pluggy cannot bind.
                assert found is not None, (
                    f"{plugin!r}.{attr_name} is not a hookable attribute"
                )
                method: _HookImplFunction[object] = found[1]
                hookimpl = HookImpl(plugin, plugin_name, method, hookimpl_opts)
                hook_name = hookimpl_opts.get("specname") or attr_name
                hook: HookCaller | None = getattr(self.hook, hook_name, None)
                if hook is None:
                    hook = HookCaller(hook_name, self._hookexec)
                    setattr(self.hook, hook_name, hook)
                elif hook.has_spec():
                    self._verify_hook(hook, hookimpl)
                    hook._maybe_apply_history(hookimpl)
                hook._add_hookimpl(hookimpl)
        return plugin_name

    def parse_hookimpl_opts(self, plugin: _Plugin, name: str) -> HookimplOpts | None:
        """Try to obtain a hook implementation from an item with the given name
        in the given plugin which is being searched for hook impls.

        :returns:
            The parsed hookimpl options, or None to skip the given item.

        This method can be overridden by ``PluginManager`` subclasses to
        customize how hook implementation are picked up. By default, returns the
        options for items decorated with :class:`HookimplMarker`.

        Discovery uses :func:`inspect.getattr_static` so properties and other
        descriptors are not executed. Only functions, classmethods,
        staticmethods, and bound methods are considered.
        """
        found = _static_hook_attr(plugin, name)
        if found is None:
            return None
        return cast(
            HookimplOpts | None,
            _get_marker_opts(found[0], self.project_name + "_impl"),
        )

    def unregister(
        self, plugin: _Plugin | None = None, name: str | None = None
    ) -> Any | None:
        """Unregister a plugin and all of its hook implementations.

        The plugin can be specified either by the plugin object or the plugin
        name. If both are specified, they must agree.

        Returns the unregistered plugin, or ``None`` if not found.
        """
        if name is None:
            assert plugin is not None, "one of name or plugin needs to be specified"
            name = self.get_name(plugin)
            assert name is not None, "plugin is not registered"

        if plugin is None:
            plugin = self.get_plugin(name)
            if plugin is None:
                return None

        hookcallers = self.get_hookcallers(plugin)
        if hookcallers:
            for hookcaller in hookcallers:
                hookcaller._remove_plugin(plugin)

        # if self._name2plugin[name] == None registration was blocked: ignore
        if self._name2plugin.get(name):
            assert name is not None
            del self._name2plugin[name]

        return plugin

    def set_blocked(self, name: str) -> None:
        """Block registrations of the given name, unregister if already registered."""
        self.unregister(name=name)
        self._name2plugin[name] = None

    def is_blocked(self, name: str) -> bool:
        """Return whether the given plugin name is blocked."""
        return name in self._name2plugin and self._name2plugin[name] is None

    def unblock(self, name: str) -> bool:
        """Unblocks a name.

        Returns whether the name was actually blocked.
        """
        if self._name2plugin.get(name, -1) is None:
            del self._name2plugin[name]
            return True
        return False

    def set_hook_impl_order(
        self,
        hook_name: str,
        *,
        before: Mapping[str, str | Iterable[str]] | None = None,
        after: Mapping[str, str | Iterable[str]] | None = None,
    ) -> None:
        """Set explicit ordering constraints between the "normal" hook
        implementations of the given hook.

        The constraints reference plugins by their registered names and
        only affect implementations which are not wrappers and are not
        marked ``tryfirst``/``trylast`` -- those keep their fixed tiers.
        Within the "normal" tier, the constraints are resolved by
        repeatedly picking, in the original call order, the next
        implementation whose predecessors have all been placed.

        This method replaces the entire ordering configuration for the
        hook. The new configuration is validated as a whole; if it is
        invalid, a :class:`HookOrderError` is raised naming the involved
        plugins and the previous configuration is kept.

        The new order takes effect on the *next* call of the hook; a call
        which is already in progress keeps its fixed order.

        :param hook_name:
            The name of the hook to order.
        :param before:
            A mapping of a plugin name to a plugin name (or iterable of
            plugin names) which it must run *before*.
        :param after:
            A mapping of a plugin name to a plugin name (or iterable of
            plugin names) which it must run *after*.

        :raises HookOrderError:
            If the hook is unknown, a referenced plugin is not registered
            or does not implement this hook, a referenced implementation
            is a wrapper or marked ``tryfirst``/``trylast``, or the
            constraints form a cycle.
        """
        edges: dict[str, set[str]] = {}
        for name, targets in _normalize_order_mapping(before).items():
            edges.setdefault(name, set()).update(targets)
        for name, targets in _normalize_order_mapping(after).items():
            for target in targets:
                edges.setdefault(target, set()).add(name)

        hook: HookCaller | None = getattr(self.hook, hook_name, None)
        if hook is None:
            raise HookOrderError(
                f"cannot set hook ordering: hook {hook_name!r} is not registered"
            )

        referenced = set(edges)
        for targets in edges.values():
            referenced.update(targets)

        unregistered = sorted(
            name for name in referenced if self.get_plugin(name) is None
        )
        if unregistered:
            raise HookOrderError(
                f"cannot set hook ordering for hook {hook_name!r}: "
                f"plugin(s) not registered: {', '.join(unregistered)}",
            )

        impls_by_name: dict[str, list[HookImpl]] = {}
        for hookimpl in hook.get_hookimpls():
            impls_by_name.setdefault(hookimpl.plugin_name, []).append(hookimpl)

        not_implementing = sorted(
            name for name in referenced if name not in impls_by_name
        )
        if not_implementing:
            raise HookOrderError(
                f"cannot set hook ordering for hook {hook_name!r}: "
                "plugin(s) do not implement this hook: "
                f"{', '.join(not_implementing)}",
            )

        not_orderable = sorted(
            name
            for name in referenced
            if any(not _is_orderable(hookimpl) for hookimpl in impls_by_name[name])
        )
        if not_orderable:
            raise HookOrderError(
                f"cannot set hook ordering for hook {hook_name!r}: "
                "ordering constraints only apply to normal hook "
                "implementations, but these plugin(s) use "
                "tryfirst/trylast/wrapper for this hook: "
                f"{', '.join(not_orderable)}",
            )

        cycle = _find_order_cycle(edges)
        if cycle is not None:
            raise HookOrderError(
                f"cannot set hook ordering for hook {hook_name!r}: "
                f"ordering constraints form a cycle: {' -> '.join(cycle)}",
            )

        hook._set_order_constraints(
            {name: frozenset(targets) for name, targets in edges.items()}
        )

    def clear_hook_impl_order(self, hook_name: str) -> None:
        """Remove all explicit ordering constraints for the given hook.

        The relative order of the hook implementations is left as it
        currently is; subsequently registered implementations are appended
        according to the default rules again.

        :raises HookOrderError:
            If the hook is unknown.
        """
        hook: HookCaller | None = getattr(self.hook, hook_name, None)
        if hook is None:
            raise HookOrderError(
                f"cannot clear hook ordering: hook {hook_name!r} is not registered"
            )
        hook._set_order_constraints({})

    def get_hook_impl_order(self, hook_name: str) -> dict[str, frozenset[str]]:
        """Return the explicit ordering constraints currently set for the
        given hook, mapping a plugin name to the plugin names it must run
        before.

        :raises HookOrderError:
            If the hook is unknown.
        """
        hook: HookCaller | None = getattr(self.hook, hook_name, None)
        if hook is None:
            raise HookOrderError(
                f"cannot get hook ordering: hook {hook_name!r} is not registered"
            )
        return hook._get_order_constraints()

    def add_hookspecs(self, module_or_class: _Namespace) -> None:
        """Add new hook specifications defined in the given ``module_or_class``.

        Functions are recognized as hook specifications if they have been
        decorated with a matching :class:`HookspecMarker`.
        """
        names = []
        for name in dir(module_or_class):
            spec_opts = self.parse_hookspec_opts(module_or_class, name)
            if spec_opts is not None:
                hc: HookCaller | None = getattr(self.hook, name, None)
                if hc is None:
                    hc = HookCaller(name, self._hookexec, module_or_class, spec_opts)
                    setattr(self.hook, name, hc)
                else:
                    # Plugins registered this hook without knowing the spec.
                    hc.set_specification(module_or_class, spec_opts)
                    for hookfunction in hc.get_hookimpls():
                        self._verify_hook(hc, hookfunction)
                names.append(name)

        if not names:
            raise ValueError(
                f"did not find any {self.project_name!r} hooks in {module_or_class!r}"
            )

    def parse_hookspec_opts(
        self, module_or_class: _Namespace, name: str
    ) -> HookspecOpts | None:
        """Try to obtain a hook specification from an item with the given name
        in the given module or class which is being searched for hook specs.

        :returns:
            The parsed hookspec options for defining a hook, or None to skip the
            given item.

        This method can be overridden by ``PluginManager`` subclasses to
        customize how hook specifications are picked up. By default, returns the
        options for items decorated with :class:`HookspecMarker`.

        Discovery uses :func:`inspect.getattr_static` so properties and other
        descriptors are not executed. Only functions, classmethods,
        staticmethods, and bound methods are considered.
        """
        found = _static_hook_attr(module_or_class, name)
        if found is None:
            return None
        return cast(
            HookspecOpts | None,
            _get_marker_opts(found[0], self.project_name + "_spec"),
        )

    def get_plugins(self) -> set[Any]:
        """Return a set of all registered plugin objects."""
        return {x for x in self._name2plugin.values() if x is not None}

    def is_registered(self, plugin: _Plugin) -> bool:
        """Return whether the plugin is already registered."""
        return any(plugin == val for val in self._name2plugin.values())

    def get_canonical_name(self, plugin: _Plugin) -> str:
        """Return a canonical name for a plugin object.

        Note that a plugin may be registered under a different name
        specified by the caller of :meth:`register(plugin, name) <register>`.
        To obtain the name of a registered plugin use :meth:`get_name(plugin)
        <get_name>` instead.
        """
        name: str | None = getattr(plugin, "__name__", None)
        return name or str(id(plugin))

    def get_plugin(self, name: str) -> Any | None:
        """Return the plugin registered under the given name, if any."""
        return self._name2plugin.get(name)

    def has_plugin(self, name: str) -> bool:
        """Return whether a plugin with the given name is registered."""
        return self.get_plugin(name) is not None

    def get_name(self, plugin: _Plugin) -> str | None:
        """Return the name the plugin is registered under, or ``None`` if
        is isn't."""
        for name, val in self._name2plugin.items():
            if plugin == val:
                return name
        return None

    def _verify_hook(self, hook: HookCaller, hookimpl: HookImpl) -> None:
        if hook.is_historic() and (hookimpl.hookwrapper or hookimpl.wrapper):
            raise PluginValidationError(
                hookimpl.plugin,
                f"Plugin {hookimpl.plugin_name!r}\nhook {hook.name!r}\n"
                "historic incompatible with yield/wrapper/hookwrapper",
            )

        assert hook.spec is not None
        if hook.spec.warn_on_impl:
            _warn_for_function(hook.spec.warn_on_impl, hookimpl.function)

        # positional arg checking
        notinspec = set(hookimpl.argnames) - set(hook.spec.argnames)
        if notinspec:
            raise PluginValidationError(
                hookimpl.plugin,
                f"Plugin {hookimpl.plugin_name!r} for hook {hook.name!r}\n"
                f"hookimpl definition: {_formatdef(hookimpl.function)}\n"
                f"Argument(s) {notinspec} are declared in the hookimpl but "
                "can not be found in the hookspec",
            )

        if hook.spec.warn_on_impl_args:
            for hookimpl_argname in hookimpl.argnames:
                argname_warning = hook.spec.warn_on_impl_args.get(hookimpl_argname)
                if argname_warning is not None:
                    _warn_for_function(argname_warning, hookimpl.function)

        if (
            hookimpl.wrapper or hookimpl.hookwrapper
        ) and not inspect.isgeneratorfunction(hookimpl.function):
            raise PluginValidationError(
                hookimpl.plugin,
                f"Plugin {hookimpl.plugin_name!r} for hook {hook.name!r}\n"
                f"hookimpl definition: {_formatdef(hookimpl.function)}\n"
                "Declared as wrapper=True or hookwrapper=True "
                "but function is not a generator function",
            )

        if hookimpl.wrapper and hookimpl.hookwrapper:
            raise PluginValidationError(
                hookimpl.plugin,
                f"Plugin {hookimpl.plugin_name!r} for hook {hook.name!r}\n"
                f"hookimpl definition: {_formatdef(hookimpl.function)}\n"
                "The wrapper=True and hookwrapper=True options are mutually exclusive",
            )

    def check_pending(self) -> None:
        """Verify that all hooks which have not been verified against a
        hook specification are optional, otherwise raise
        :exc:`PluginValidationError`."""
        for name in self.hook.__dict__:
            if name[0] == "_":
                continue
            hook: HookCaller = getattr(self.hook, name)
            if not hook.has_spec():
                for hookimpl in hook.get_hookimpls():
                    if not hookimpl.optionalhook:
                        raise PluginValidationError(
                            hookimpl.plugin,
                            f"unknown hook {name!r} in plugin {hookimpl.plugin!r}",
                        )

    def load_setuptools_entrypoints(self, group: str, name: str | None = None) -> int:
        """Load modules from querying the specified setuptools ``group``.

        :param group:
            Entry point group to load plugins.
        :param name:
            If given, loads only plugins with the given ``name``.

        :return:
            The number of plugins loaded by this call.
        """
        import importlib.metadata

        count = 0
        for dist in list(importlib.metadata.distributions()):
            for ep in dist.entry_points:
                if (
                    ep.group != group
                    or (name is not None and ep.name != name)
                    # already registered
                    or self.get_plugin(ep.name)
                    or self.is_blocked(ep.name)
                ):
                    continue
                plugin = ep.load()
                self.register(plugin, name=ep.name)
                self._plugin_distinfo.append((plugin, dist))
                count += 1
        return count

    def list_plugin_distinfo(self) -> list[tuple[_Plugin, DistFacade]]:
        """Return a list of (plugin, distinfo) pairs for all
        setuptools-registered plugins.

        .. note::
            The distinfo objects are wrapped with :class:`~pluggy._compat.DistFacade`
            for backward compatibility with the legacy pkg_resources API.
            Prefer :meth:`list_plugin_distributions` for raw
            :class:`importlib.metadata.Distribution` objects.
        """
        from ._compat import DistFacade

        return [(plugin, DistFacade(dist)) for plugin, dist in self._plugin_distinfo]

    def list_plugin_distributions(
        self,
    ) -> list[tuple[_Plugin, importlib.metadata.Distribution]]:
        """Return a list of (plugin, distribution) pairs for all plugins
        loaded via entry points.

        .. versionadded:: 1.7
        """
        return list(self._plugin_distinfo)

    def list_name_plugin(self) -> list[tuple[str, _Plugin]]:
        """Return a list of (name, plugin) pairs for all registered plugins."""
        return list(self._name2plugin.items())

    def get_hookcallers(self, plugin: _Plugin) -> list[HookCaller] | None:
        """Get all hook callers for the specified plugin.

        :returns:
            The hook callers, or ``None`` if ``plugin`` is not registered in
            this plugin manager.
        """
        if self.get_name(plugin) is None:
            return None
        hookcallers = []
        for hookcaller in self.hook.__dict__.values():
            if any(impl.plugin is plugin for impl in hookcaller.get_hookimpls()):
                hookcallers.append(hookcaller)
        return hookcallers

    def add_hookcall_monitoring(
        self, before: _BeforeTrace, after: _AfterTrace
    ) -> Callable[[], None]:
        """Add before/after tracing functions for all hooks.

        Returns an undo function which, when called, removes the added tracers.

        ``before(hook_name, hook_impls, kwargs)`` will be called ahead
        of all hook calls and receive a hookcaller instance, a list
        of HookImpl instances and the keyword arguments for the hook call.

        ``after(outcome, hook_name, hook_impls, kwargs)`` receives the
        same arguments as ``before`` but also a :class:`~pluggy.Result` object
        which represents the result of the overall hook call.
        """
        oldcall = self._inner_hookexec

        def traced_hookexec(
            hook_name: str,
            hook_impls: Sequence[HookImpl],
            caller_kwargs: Mapping[str, object],
            firstresult: bool,
        ) -> object | list[object]:
            before(hook_name, hook_impls, caller_kwargs)
            outcome = Result.from_call(
                lambda: oldcall(hook_name, hook_impls, caller_kwargs, firstresult)
            )
            after(outcome, hook_name, hook_impls, caller_kwargs)
            return outcome.get_result()

        self._inner_hookexec = traced_hookexec

        def undo() -> None:
            self._inner_hookexec = oldcall

        return undo

    def enable_tracing(self) -> Callable[[], None]:
        """Enable tracing of hook calls.

        Returns an undo function which, when called, removes the added tracing.
        """
        hooktrace = self.trace.root.get("hook")

        def before(
            hook_name: str, methods: Sequence[HookImpl], kwargs: Mapping[str, object]
        ) -> None:
            hooktrace.root.indent += 1
            hooktrace(hook_name, kwargs)

        def after(
            outcome: Result[object],
            hook_name: str,
            methods: Sequence[HookImpl],
            kwargs: Mapping[str, object],
        ) -> None:
            if outcome.exception is None:
                hooktrace("finish", hook_name, "-->", outcome.get_result())
            hooktrace.root.indent -= 1

        return self.add_hookcall_monitoring(before, after)

    def subset_hook_caller(
        self, name: str, remove_plugins: Iterable[_Plugin]
    ) -> HookCaller:
        """Return a proxy :class:`~pluggy.HookCaller` for the named hook which
        calls all registered plugins except the ones from remove_plugins."""
        orig: HookCaller = getattr(self.hook, name)
        plugins_to_remove = set(remove_plugins)
        # Optimization: discard plugins to remove which don't actually implement
        # the hook, to make checks faster.
        plugins_to_remove.intersection_update(
            hookimpl.plugin for hookimpl in orig._hookimpls
        )
        # Optimization: if none of the plugins to remove actually implement the
        # hook, avoid the subset hook caller overhead.
        if not plugins_to_remove:
            return orig
        return _SubsetHookCaller(orig, plugins_to_remove)


def _formatdef(func: Callable[..., object]) -> str:
    return f"{func.__name__}{inspect.signature(func)}"


def _normalize_order_mapping(
    mapping: Mapping[str, str | Iterable[str]] | None,
) -> dict[str, set[str]]:
    """Normalize a before/after mapping to sets of plugin names."""
    edges: dict[str, set[str]] = {}
    if mapping is None:
        return edges
    for name, targets in mapping.items():
        if isinstance(targets, str):
            targets = (targets,)
        edges.setdefault(name, set()).update(targets)
    return edges


def _find_order_cycle(edges: Mapping[str, set[str]]) -> list[str] | None:
    """Return a cycle in the ordering constraints, or None if acyclic.

    ``edges`` maps a plugin name to the plugin names it must run before.
    """
    # 0 = unvisited, 1 = on the current path, 2 = done.
    color: dict[str, int] = {}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = 1
        path.append(node)
        for nxt in edges.get(node, ()):
            state = color.get(nxt, 0)
            if state == 1:
                return path[path.index(nxt) :] + [nxt]
            if state == 0:
                cycle = visit(nxt)
                if cycle is not None:
                    return cycle
        path.pop()
        color[node] = 2
        return None

    for node in edges:
        if color.get(node, 0) == 0:
            cycle = visit(node)
            if cycle is not None:
                return cycle
    return None
