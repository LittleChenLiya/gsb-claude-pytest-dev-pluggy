"""
Static explanation ("dry run") of a single hook call.

Nothing in this module executes plugin code: hook implementations, wrappers,
descriptors and result callbacks are only inspected, never called.

Ordering and filtering reuse the real HookCaller data (via
PluginManager.subset_hook_caller), so a plan follows exactly the same
tryfirst/trylast/wrapper/hookwrapper ordering rules as an actual call.
"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from typing import Final
from typing import TYPE_CHECKING
from typing import TypeAlias

from ._hooks import HookCaller
from ._hooks import HookImpl


if TYPE_CHECKING:
    from ._manager import PluginManager


_KwargsNames: TypeAlias = Collection[str]

#: Diagnostic code for a hook name with no HookCaller.
UNKNOWN_HOOK: Final = "unknown_hook"
#: Diagnostic code for arguments required by the spec or an implementation
#: which are not present in the planned call arguments.
MISSING_ARGUMENTS: Final = "missing_arguments"
#: Diagnostic code for a remove_plugins entry that is not a registered
#: plugin name or plugin object.
INVALID_REMOVE_PLUGIN: Final = "invalid_remove_plugin"

_ABSENT: Final = object()


def _function_location(function: object) -> str | None:
    """Return "path:lineno" for the function, without executing it."""
    func = inspect.unwrap(function)  # type: ignore[arg-type]
    code = getattr(func, "__code__", None)
    if code is None:
        return None
    return f"{code.co_filename}:{code.co_firstlineno}"


def _function_name(function: object) -> str:
    name = getattr(function, "__name__", None)
    if name is not None:
        return str(name)
    return type(function).__name__


@dataclass(frozen=True)
class HookExplainStep:
    """A single entry, implementation or teardown step of a hook call plan.

    Step instances are immutable snapshots: later registrations,
    unregistrations or plugin object changes do not affect them.
    """

    #: The name the owning plugin is registered under.
    plugin_name: str
    #: The function name (__name__) of the hook implementation.
    function_name: str
    #: "impl", "wrapper" (new-style) or "hookwrapper" (old-style).
    kind: str
    #: Source location of the function, as "path:lineno", or None when the
    #: function has no Python code object.
    location: str | None
    #: Call argument names this step would receive, in binding order.
    argnames: tuple[str, ...]
    #: Arguments from argnames missing from the planned call; a real call
    #: would raise HookCallError when reaching this step.
    missing_args: tuple[str, ...]
    #: Whether the implementation is marked tryfirst.
    tryfirst: bool
    #: Whether the implementation is marked trylast.
    trylast: bool
    #: Whether the implementation is marked optionalhook.
    optionalhook: bool
    #: Whether this step only runs conditionally. For firstresult hooks,
    #: implementations after a possible short-circuit point are conditional:
    #: they run only if every earlier implementation returns None. The plan
    #: never predicts which implementation actually wins.
    conditional: bool
    #: Whether this step may short-circuit later implementations. True for
    #: every normal implementation of a firstresult hook: if it returns a
    #: non-None value, execution halts before the following impl steps.
    may_short_circuit: bool


@dataclass(frozen=True)
class ExplainDiagnostic:
    """A problem detected while planning the hook call.

    Diagnostics never rely on executing plugin code. code is one of
    UNKNOWN_HOOK, MISSING_ARGUMENTS or INVALID_REMOVE_PLUGIN.
    """

    #: Machine-readable diagnostic code.
    code: str
    #: Human-readable explanation.
    message: str


@dataclass(frozen=True)
class HookCallPlan:
    """An independent snapshot of how one hook call would proceed.

    The plan is immutable and detached from the HookCaller: registering or
    unregistering plugins after obtaining it does not change any steps.

    Wrappers execute in three moments:

    1. enter_steps -- pre-yield code, outermost last, in execution order;
    2. impl_steps -- the normal (non-wrapper) implementations;
    3. exit_steps -- wrapper teardown (post-yield) code, in reverse order.
    """

    #: Name of the hook.
    hook_name: str
    #: Whether the hook spec declares firstresult.
    firstresult: bool
    #: Whether the hook is historic.
    historic: bool
    #: Whether the hook has a registered specification.
    has_spec: bool
    #: Snapshot of the call argument names passed to explain_hook.
    kwargs: tuple[str, ...]
    #: Wrapper entry steps (pre-yield), in execution order.
    enter_steps: tuple[HookExplainStep, ...]
    #: Normal implementation steps, in execution order.
    impl_steps: tuple[HookExplainStep, ...]
    #: Wrapper teardown steps (post-yield), in execution order.
    exit_steps: tuple[HookExplainStep, ...]
    #: All detected problems; the plan may still be usable when this is
    #: non-empty (e.g. it shows where a missing argument would raise).
    diagnostics: tuple[ExplainDiagnostic, ...]


def _build_step(
    hook_impl: HookImpl,
    kwargs_names: _KwargsNames,
    *,
    conditional: bool,
    may_short_circuit: bool,
) -> HookExplainStep:
    if hook_impl.hookwrapper:
        kind = "hookwrapper"
    elif hook_impl.wrapper:
        kind = "wrapper"
    else:
        kind = "impl"
    missing_args = tuple(
        argname for argname in hook_impl.argnames if argname not in kwargs_names
    )
    return HookExplainStep(
        plugin_name=hook_impl.plugin_name,
        function_name=_function_name(hook_impl.function),
        kind=kind,
        location=_function_location(hook_impl.function),
        argnames=tuple(hook_impl.argnames),
        missing_args=missing_args,
        tryfirst=hook_impl.tryfirst,
        trylast=hook_impl.trylast,
        optionalhook=hook_impl.optionalhook,
        conditional=conditional,
        may_short_circuit=may_short_circuit,
    )


def build_hook_call_plan(
    manager: PluginManager,
    name: str,
    kwargs_names: _KwargsNames,
    remove_plugins: Collection[object],
) -> HookCallPlan:
    """:meta private:"""
    diagnostics: list[ExplainDiagnostic] = []

    removed: set[object] = set()
    for item in remove_plugins:
        if isinstance(item, str):
            registered = manager._name2plugin.get(item, _ABSENT)
            if registered is _ABSENT or registered is None:
                diagnostics.append(
                    ExplainDiagnostic(
                        INVALID_REMOVE_PLUGIN,
                        f"remove_plugins entry {item!r} is not a registered "
                        "plugin name",
                    )
                )
            else:
                removed.add(registered)
        elif manager.is_registered(item):
            removed.add(item)
        else:
            diagnostics.append(
                ExplainDiagnostic(
                    INVALID_REMOVE_PLUGIN,
                    f"remove_plugins entry {item!r} is not a registered plugin",
                )
            )

    hook_caller: HookCaller | None = manager.hook.__dict__.get(name)
    kwargs_snapshot = tuple(kwargs_names)
    if hook_caller is None:
        diagnostics.append(ExplainDiagnostic(UNKNOWN_HOOK, f"unknown hook {name!r}"))
        return HookCallPlan(
            hook_name=name,
            firstresult=False,
            historic=False,
            has_spec=False,
            kwargs=kwargs_snapshot,
            enter_steps=(),
            impl_steps=(),
            exit_steps=(),
            diagnostics=tuple(diagnostics),
        )

    # Reuse the real HookCaller rules for ordering and filtering.
    effective_caller: HookCaller = (
        manager.subset_hook_caller(name, removed) if removed else hook_caller
    )
    spec = hook_caller.spec
    firstresult = bool(spec.opts.get("firstresult", False)) if spec else False

    if spec is not None:
        missing_spec_args = tuple(
            argname for argname in spec.argnames if argname not in kwargs_names
        )
        if missing_spec_args:
            diagnostics.append(
                ExplainDiagnostic(
                    MISSING_ARGUMENTS,
                    "hook call is missing required spec argument(s): "
                    + ", ".join(repr(a) for a in missing_spec_args),
                )
            )

    wrappers: list[HookImpl] = []
    nonwrappers: list[HookImpl] = []
    # Multicall iterates the stored list in reverse; get_hookimpls() returns
    # the same defensive copy HookCaller.__call__ passes to the hook executor.
    for hook_impl in reversed(effective_caller.get_hookimpls()):
        if hook_impl.wrapper or hook_impl.hookwrapper:
            wrappers.append(hook_impl)
        else:
            nonwrappers.append(hook_impl)

    # For firstresult hooks every normal implementation is a potential
    # short-circuit point; everything after the first one is conditional.
    # No winner is ever predicted.
    enter_steps = tuple(
        _build_step(
            hook_impl,
            kwargs_names,
            conditional=False,
            may_short_circuit=False,
        )
        for hook_impl in wrappers
    )
    impl_steps_list: list[HookExplainStep] = []
    short_circuit_possible = False
    for hook_impl in nonwrappers:
        impl_steps_list.append(
            _build_step(
                hook_impl,
                kwargs_names,
                conditional=short_circuit_possible,
                may_short_circuit=firstresult,
            )
        )
        if firstresult:
            short_circuit_possible = True
    impl_steps = tuple(impl_steps_list)
    # Teardown runs in reverse setup order.
    exit_steps = tuple(
        _build_step(
            hook_impl,
            kwargs_names,
            conditional=False,
            may_short_circuit=False,
        )
        for hook_impl in reversed(wrappers)
    )

    for step in (*enter_steps, *impl_steps, *exit_steps):
        if step.missing_args:
            diagnostics.append(
                ExplainDiagnostic(
                    MISSING_ARGUMENTS,
                    f"implementation {step.plugin_name!r} {step.function_name} "
                    "is missing call argument(s): "
                    + ", ".join(repr(a) for a in step.missing_args),
                )
            )

    return HookCallPlan(
        hook_name=name,
        firstresult=firstresult,
        historic=hook_caller.is_historic(),
        has_spec=spec is not None,
        kwargs=kwargs_snapshot,
        enter_steps=enter_steps,
        impl_steps=impl_steps,
        exit_steps=exit_steps,
        diagnostics=tuple(diagnostics),
    )


def normalize_kwargs_names(
    kwargs: Iterable[str] | Mapping[str, object] | None,
) -> _KwargsNames:
    """Normalize the explain_hook argument-name input."""
    if kwargs is None:
        return ()
    if isinstance(kwargs, Mapping):
        return tuple(kwargs.keys())
    return tuple(kwargs)
