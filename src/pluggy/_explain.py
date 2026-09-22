"""
Static, non-executing explanation of hook calls.

This module powers :meth:`pluggy.PluginManager.explain_hook`. It builds a
structured plan describing how a hook call *would* flow through the
registered hook implementations -- enter (wrapper setup), regular
implementations, and exit (wrapper teardown) phases -- without executing
any hook implementation, wrapper, descriptor, or result callback.

The ordering and filtering reuse the real :class:`~pluggy.HookCaller`
machinery (``HookCaller._hookimpls`` and
:meth:`~pluggy.PluginManager.subset_hook_caller`), so the plan matches what
an actual call would do, including ``tryfirst``/``trylast`` ordering,
new-style ``wrapper``\\s, old-style ``hookwrapper``\\s, ``firstresult``
short-circuiting and ``optionalhook`` implementations.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final
from typing import Literal
from typing import TYPE_CHECKING

from ._hooks import _Plugin
from ._hooks import HookCaller
from ._hooks import HookImpl


if TYPE_CHECKING:
    from ._manager import PluginManager


__all__ = [
    "HookPlan",
    "HookPlanDiagnostic",
    "HookPlanStep",
]


#: Diagnostic kinds emitted in :attr:`HookPlan.diagnostics`.
DiagnosticKind = Literal[
    "unknown-hook",
    "no-spec",
    "missing-arguments",
    "missing-impl-arguments",
    "invalid-plugin-filter",
]

#: Which phase of a hook call a plan step belongs to.
PhaseKind = Literal["enter", "impl", "exit"]

#: The flavor of a hook implementation.
StepKind = Literal["wrapper", "hookwrapper", "impl"]


@dataclass(frozen=True)
class HookPlanDiagnostic:
    """A single diagnostic found while building a :class:`HookPlan`.

    .. versionadded:: 1.7
    """

    #: Machine-readable category, one of ``"unknown-hook"``, ``"no-spec"``,
    #: ``"missing-arguments"``, ``"missing-impl-arguments"`` or
    #: ``"invalid-plugin-filter"``.
    kind: DiagnosticKind
    #: Human-readable description of the problem.
    message: str


@dataclass(frozen=True)
class HookPlanStep:
    """One step of a :class:`HookPlan`.

    A step is a plain snapshot: it holds no references to the plugin, the
    hook implementation or the plugin manager, so later (un)registrations
    cannot mutate it.

    .. versionadded:: 1.7
    """

    #: The phase this step belongs to: ``"enter"`` (wrapper setup),
    #: ``"impl"`` (regular implementation) or ``"exit"`` (wrapper teardown).
    phase: PhaseKind
    #: The name the plugin is registered under.
    plugin_name: str
    #: The name of the hook implementation function.
    function_name: str
    #: ``"path/to/file.py:lineno"`` of the function, or ``"<unknown>"``.
    location: str
    #: The hook argument names the function receives.
    argnames: tuple[str, ...]
    #: ``"wrapper"`` (new-style), ``"hookwrapper"`` (old-style) or ``"impl"``.
    kind: StepKind
    #: Whether the implementation is marked ``tryfirst``.
    tryfirst: bool
    #: Whether the implementation is marked ``trylast``.
    trylast: bool
    #: Whether the implementation is marked ``optionalhook``.
    optionalhook: bool
    #: Whether this step only runs conditionally. For ``firstresult`` hooks,
    #: every implementation after the first is skipped as soon as an earlier
    #: one returns a non-``None`` result; which one wins cannot be known
    #: statically, so these steps are marked conditional.
    conditional: bool
    #: Whether a non-``None`` result produced at this step short-circuits
    #: (skips) the remaining implementations. Only true for ``"impl"`` steps
    #: of ``firstresult`` hooks. Wrapper teardown steps can always alter the
    #: result or raise, which is implied by their ``kind``.
    may_shortcut: bool


@dataclass(frozen=True)
class HookPlan:
    """A static, immutable plan of how a hook call would execute.

    Produced by :meth:`pluggy.PluginManager.explain_hook`. The plan is an
    independent snapshot: later registrations or unregistrations do not
    affect it.

    .. versionadded:: 1.7
    """

    #: The name of the hook.
    hook_name: str
    #: Whether the hook has a registered hook specification.
    has_spec: bool
    #: Whether the hook is a ``firstresult`` hook.
    firstresult: bool
    #: Whether the hook is historic.
    historic: bool
    #: Wrapper setup steps, in execution order (outermost first).
    enter: tuple[HookPlanStep, ...]
    #: Regular (non-wrapper) implementation steps, in execution order.
    impls: tuple[HookPlanStep, ...]
    #: Wrapper teardown steps, in execution order (innermost first).
    exit: tuple[HookPlanStep, ...]
    #: Diagnostics discovered while planning, e.g. an unknown hook, missing
    #: required arguments or invalid plugin filters.
    diagnostics: tuple[HookPlanDiagnostic, ...]

    @property
    def steps(self) -> tuple[HookPlanStep, ...]:
        """All steps in execution order: enter, then impls, then exit."""
        return self.enter + self.impls + self.exit

    @property
    def ok(self) -> bool:
        """Whether the plan has no diagnostics."""
        return not self.diagnostics


def _function_location(hookimpl: HookImpl) -> str:
    code = getattr(hookimpl.function, "__code__", None)
    if code is None:
        return "<unknown>"
    return f"{code.co_filename}:{code.co_firstlineno}"


def _make_step(
    hookimpl: HookImpl,
    phase: PhaseKind,
    *,
    conditional: bool = False,
    may_shortcut: bool = False,
) -> HookPlanStep:
    if hookimpl.wrapper:
        kind: StepKind = "wrapper"
    elif hookimpl.hookwrapper:
        kind = "hookwrapper"
    else:
        kind = "impl"
    return HookPlanStep(
        phase=phase,
        plugin_name=hookimpl.plugin_name,
        function_name=getattr(hookimpl.function, "__name__", "<unknown>"),
        location=_function_location(hookimpl),
        argnames=tuple(hookimpl.argnames),
        kind=kind,
        tryfirst=hookimpl.tryfirst,
        trylast=hookimpl.trylast,
        optionalhook=hookimpl.optionalhook,
        conditional=conditional,
        may_shortcut=may_shortcut,
    )


def explain_hook(
    pm: PluginManager,
    hook_name: str,
    argnames: Iterable[str] = (),
    remove_plugins: Iterable[_Plugin | str] = (),
) -> HookPlan:
    """Build a :class:`HookPlan` for a hook call without executing anything.

    See :meth:`pluggy.PluginManager.explain_hook` for documentation.
    """
    diagnostics: list[HookPlanDiagnostic] = []
    provided: Final[frozenset[str]] = frozenset(argnames)

    caller: HookCaller | None = getattr(pm.hook, hook_name, None)
    if caller is None:
        diagnostics.append(
            HookPlanDiagnostic(
                kind="unknown-hook",
                message=f"unknown hook {hook_name!r}: no hook caller is registered "
                "under this name and no plugin implements it",
            )
        )
        return HookPlan(
            hook_name=hook_name,
            has_spec=False,
            firstresult=False,
            historic=False,
            enter=(),
            impls=(),
            exit=(),
            diagnostics=tuple(diagnostics),
        )

    # Resolve the plugin filter, reusing the real subset filtering for the
    # plugins which are valid.
    plugins_to_remove: list[_Plugin] = []
    for entry in remove_plugins:
        if isinstance(entry, str):
            plugin = pm.get_plugin(entry)
            if plugin is None:
                diagnostics.append(
                    HookPlanDiagnostic(
                        kind="invalid-plugin-filter",
                        message=f"no plugin registered under name {entry!r}",
                    )
                )
                continue
        else:
            plugin = entry
            if pm.get_name(plugin) is None:
                diagnostics.append(
                    HookPlanDiagnostic(
                        kind="invalid-plugin-filter",
                        message=f"plugin {plugin!r} is not registered",
                    )
                )
                continue
        plugins_to_remove.append(plugin)
    if plugins_to_remove:
        caller = pm.subset_hook_caller(hook_name, plugins_to_remove)

    spec = caller.spec
    firstresult = bool(spec and spec.opts.get("firstresult", False))
    historic = caller.is_historic()

    if spec is None:
        non_optional = [
            impl.plugin_name for impl in caller.get_hookimpls() if not impl.optionalhook
        ]
        if non_optional:
            diagnostics.append(
                HookPlanDiagnostic(
                    kind="no-spec",
                    message=f"hook {hook_name!r} has no hookspec and non-optional "
                    f"implementations from plugins {non_optional!r} would fail "
                    "PluginManager.check_pending()",
                )
            )
    else:
        missing = [name for name in spec.argnames if name not in provided]
        if missing:
            diagnostics.append(
                HookPlanDiagnostic(
                    kind="missing-arguments",
                    message=f"argument(s) {missing!r} declared in the hookspec of "
                    f"{hook_name!r} are not provided by this call",
                )
            )

    # The real execution order: HookCaller iterates its impl list in reverse.
    hookimpls = caller.get_hookimpls()
    exec_order: Sequence[HookImpl] = list(reversed(hookimpls))

    for impl in exec_order:
        missing_impl_args = [name for name in impl.argnames if name not in provided]
        if missing_impl_args:
            diagnostics.append(
                HookPlanDiagnostic(
                    kind="missing-impl-arguments",
                    message=f"implementation from plugin {impl.plugin_name!r} "
                    f"requires argument(s) {missing_impl_args!r} which are not "
                    "provided by this call; a real call would raise HookCallError",
                )
            )

    wrappers = [impl for impl in exec_order if impl.wrapper or impl.hookwrapper]
    regulars = [impl for impl in exec_order if not (impl.wrapper or impl.hookwrapper)]

    enter = tuple(_make_step(impl, "enter") for impl in wrappers)
    impl_steps = tuple(
        _make_step(
            impl,
            "impl",
            # With firstresult, any earlier non-None result halts the loop,
            # so only the first implementation runs unconditionally.
            conditional=firstresult and index > 0,
            may_shortcut=firstresult,
        )
        for index, impl in enumerate(regulars)
    )
    # Teardowns run in the reverse order of their setup.
    exit_steps = tuple(_make_step(impl, "exit") for impl in reversed(wrappers))

    return HookPlan(
        hook_name=hook_name,
        has_spec=spec is not None,
        firstresult=firstresult,
        historic=historic,
        enter=enter,
        impls=impl_steps,
        exit=exit_steps,
        diagnostics=tuple(diagnostics),
    )
