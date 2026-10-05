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
from ._hooks import _Namespace
from ._hooks import _Plugin
from ._hooks import _SubsetHookCaller
from ._hooks import HookCaller
from ._hooks import HookImpl
from ._hooks import HookimplOpts
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
        #: Explicit alias table mapping alias names to canonical plugin names.
        #: The true registration identity of every plugin is its canonical name
        #: in :attr:`_name2plugin`; aliases are only alternate handles used to
        #: query or block the same registered plugin.
        self._plugin_aliases: Final[dict[str, str]] = {}
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
        # The canonical name is the only true registration identity. If the
        # given name is an alias, resolve it to its canonical name so the
        # plugin is registered under the canonical name rather than the alias.
        plugin_name = self._resolve_alias(plugin_name)

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

    def add_alias(self, alias: str, canonical_name: str) -> None:
        """Register an explicit alias for a plugin's canonical name.

        After registering an alias, callers can use the alias (or any other
        alias) to query or block the same plugin registered under its canonical
        name. The canonical name is the only true registration identity; the
        alias table merely maps alternate handles to it.

        Alias chains are normalized before being stored: the target is
        resolved through any existing aliases to its final canonical name.
        Cycles (e.g. ``a -> b -> a``) and conflicts (the alias already points
        to a different canonical name) are rejected before any state is
        changed.

        :param alias:
            The alternate name to register.
        :param canonical_name:
            The canonical plugin name (or an existing alias) that ``alias``
            should resolve to.

        :raises ValueError:
            If adding the alias would create a cycle or conflict.
        """
        target = self._resolve_alias(canonical_name)

        # A cycle is formed when the resolved target leads back to the alias
        # itself (e.g. a -> b, b -> a).
        if target == alias:
            raise ValueError(
                f"Alias cycle detected: {alias!r} -> {canonical_name!r} "
                f"-> ... -> {alias!r}"
            )

        existing = self._plugin_aliases.get(alias)
        if existing is not None and existing != target:
            raise ValueError(
                f"Alias conflict: {alias!r} already points to {existing!r}, "
                f"cannot point to {target!r}"
            )

        self._plugin_aliases[alias] = target

    def _resolve_alias(self, name: str) -> str:
        """Resolve ``name`` to its canonical plugin name.

        Follows alias chains until a name that is not itself an alias is
        reached. Raises :class:`ValueError` if a cycle is encountered while
        resolving (defensive; cycles are normally rejected by
        :meth:`add_alias`).
        """
        seen: set[str] = set()
        current = name
        while current in self._plugin_aliases:
            if current in seen:
                raise ValueError(f"Alias cycle detected while resolving {name!r}")
            seen.add(current)
            current = self._plugin_aliases[current]
        return current

    def _aliases_for(self, canonical_name: str) -> list[str]:
        """Return all aliases that resolve to ``canonical_name``."""
        return [
            alias
            for alias, target in self._plugin_aliases.items()
            if target == canonical_name
        ]

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
        self,
        plugin: _Plugin | None = None,
        name: str | None = None,
        *,
        _cleanup_aliases: bool = True,
    ) -> Any | None:
        """Unregister a plugin and all of its hook implementations.

        The plugin can be specified either by the plugin object or the plugin
        name. If both are specified, they must agree.

        When ``name`` is an alias, it is resolved to its canonical name first;
        unregistering through any alias removes the same plugin and cleans up
        all alias mappings pointing to that canonical name. Historic hook
        playback is not replayed by this operation.

        Returns the unregistered plugin, or ``None`` if not found.
        """
        if name is None:
            assert plugin is not None, "one of name or plugin needs to be specified"
            name = self.get_name(plugin)
            assert name is not None, "plugin is not registered"

        canonical_name = self._resolve_alias(name)

        if plugin is None:
            plugin = self.get_plugin(canonical_name)
            if plugin is None:
                return None
        else:
            registered = self.get_plugin(canonical_name)
            if registered is not None and registered != plugin:
                return None

        hookcallers = self.get_hookcallers(plugin)
        if hookcallers:
            for hookcaller in hookcallers:
                hookcaller._remove_plugin(plugin)

        # if self._name2plugin[canonical_name] == None registration was
        # blocked: ignore
        if self._name2plugin.get(canonical_name):
            assert canonical_name is not None
            del self._name2plugin[canonical_name]

        # Clean up all aliases pointing to the removed canonical name.
        if _cleanup_aliases:
            for alias in self._aliases_for(canonical_name):
                del self._plugin_aliases[alias]

        return plugin

    def set_blocked(self, name: str) -> None:
        """Block registrations of the given name, unregister if already registered.

        When ``name`` is an alias, it is resolved to its canonical name first,
        so blocking takes effect by canonical identity. Alias mappings are
        preserved so that querying or blocking through an alias continues to
        resolve to the blocked canonical name.
        """
        canonical_name = self._resolve_alias(name)
        self.unregister(name=canonical_name, _cleanup_aliases=False)
        self._name2plugin[canonical_name] = None

    def is_blocked(self, name: str) -> bool:
        """Return whether the given plugin name is blocked.

        When ``name`` is an alias, it is resolved to its canonical name first.
        """
        canonical_name = self._resolve_alias(name)
        return (
            canonical_name in self._name2plugin
            and self._name2plugin[canonical_name] is None
        )

    def unblock(self, name: str) -> bool:
        """Unblocks a name.

        When ``name`` is an alias, it is resolved to its canonical name first.

        Returns whether the name was actually blocked.
        """
        canonical_name = self._resolve_alias(name)
        if self._name2plugin.get(canonical_name, -1) is None:
            del self._name2plugin[canonical_name]
            return True
        return False

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
        """Return the plugin registered under the given name, if any.

        When ``name`` is an alias, it is resolved to its canonical name first,
        so querying through an alias returns the same plugin as querying
        through the canonical name.
        """
        canonical_name = self._resolve_alias(name)
        return self._name2plugin.get(canonical_name)

    def has_plugin(self, name: str) -> bool:
        """Return whether a plugin with the given name is registered.

        When ``name`` is an alias, it is resolved to its canonical name first.
        """
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
