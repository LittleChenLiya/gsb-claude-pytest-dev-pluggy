"""
Optional auditing of exception/result transitions across hook wrapper layers.

The audit is *observational only*: it never calls wrapper functions itself and
never compares exception messages.  Each recorded stage keeps only the layer
identity (hook name, plugin name, implementation name, layer kind) and the
*types* of the incoming/outgoing exceptions, so enabling the audit does not
change the final result or exception of a hook call.
"""

from __future__ import annotations

from typing import Final
from typing import final
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._hooks import HookImpl


#: Layer kind of a regular (non-wrapper) hook implementation.
LAYER_IMPL: Final = "impl"
#: Layer kind of a new-style hook wrapper (``wrapper=True``).
LAYER_WRAPPER: Final = "wrapper"
#: Layer kind of an old-style hook wrapper (``hookwrapper=True``).
LAYER_HOOKWRAPPER: Final = "hookwrapper"

#: The layer turned a result into an exception.
ACTION_RAISED: Final = "raised"
#: The layer propagated the very same exception instance unchanged.
ACTION_PRESERVED: Final = "preserved"
#: The layer replaced the incoming exception with a different one.
ACTION_REPLACED: Final = "replaced"
#: The layer recovered from an exception by (re)turning a result, i.e. it
#: "took over" the outcome of the hook call.
ACTION_RECOVERED: Final = "recovered"
#: The layer received a result and produced a result.
ACTION_PASSED: Final = "passed"


@final
class WrapperAuditStage:
    """A single observed transition at one wrapper layer of a hook call.

    Stages are collected in real execution order.  Only types and layer
    identities are kept - never exception instances or messages.
    """

    __slots__ = (
        "action",
        "hook_name",
        "impl_name",
        "incoming_exception",
        "layer_kind",
        "outgoing_exception",
        "plugin_name",
    )

    def __init__(
        self,
        hook_name: str,
        *,
        layer_kind: str,
        plugin_name: str,
        impl_name: str,
        incoming_exception: type[BaseException] | None,
        outgoing_exception: type[BaseException] | None,
        action: str,
    ) -> None:
        """:meta private:"""
        #: Name of the hook being called.
        self.hook_name: Final = hook_name
        #: Kind of layer: ``"impl"``, ``"wrapper"`` or ``"hookwrapper"``.
        self.layer_kind: Final = layer_kind
        #: Name of the plugin providing this layer.
        self.plugin_name: Final = plugin_name
        #: Name of the hook implementation function at this layer.
        self.impl_name: Final = impl_name
        #: Type of the exception arriving at this layer, or ``None`` when a
        #: result arrived.
        self.incoming_exception: Final = incoming_exception
        #: Type of the exception leaving this layer, or ``None`` when a
        #: result left the layer.
        self.outgoing_exception: Final = outgoing_exception
        #: What the layer did: one of ``"raised"``, ``"preserved"``,
        #: ``"replaced"``, ``"recovered"`` or ``"passed"``.
        self.action: Final = action

    def __repr__(self) -> str:
        incoming = (
            self.incoming_exception.__name__
            if self.incoming_exception is not None
            else "result"
        )
        outgoing = (
            self.outgoing_exception.__name__
            if self.outgoing_exception is not None
            else "result"
        )
        return (
            f"<WrapperAuditStage {self.hook_name!r} layer={self.layer_kind} "
            f"plugin={self.plugin_name!r} impl={self.impl_name!r} "
            f"{incoming} -> {outgoing} action={self.action!r}>"
        )


@final
class WrapperAuditReport:
    """The audit trail of a single hook call.

    Contains one :class:`WrapperAuditStage` per observed layer transition, in
    real execution order, plus the final outcome of the call (kept as an
    exception *type* only).
    """

    __slots__ = ("hook_name", "outcome_exception", "stages")

    def __init__(self, hook_name: str) -> None:
        """:meta private:"""
        #: Name of the hook which was called.
        self.hook_name: Final = hook_name
        #: The observed layer transitions, in real execution order.
        self.stages: Final[list[WrapperAuditStage]] = []
        #: The type of the exception the hook call finished with, or ``None``
        #: when the hook call returned a result.
        self.outcome_exception: type[BaseException] | None = None

    @property
    def raised(self) -> bool:
        """Whether the hook call finished with an exception."""
        return self.outcome_exception is not None

    def _record_stage(
        self,
        hook_impl: HookImpl,
        incoming: BaseException | None,
        outgoing: BaseException | None,
    ) -> None:
        # Observational only: exceptions are compared by identity (never by
        # message) and only their types are stored.
        if incoming is None:
            action = ACTION_RAISED if outgoing is not None else ACTION_PASSED
        elif outgoing is None:
            action = ACTION_RECOVERED
        elif incoming is outgoing:
            action = ACTION_PRESERVED
        else:
            action = ACTION_REPLACED
        if hook_impl.hookwrapper:
            layer_kind = LAYER_HOOKWRAPPER
        elif hook_impl.wrapper:
            layer_kind = LAYER_WRAPPER
        else:
            layer_kind = LAYER_IMPL
        function = hook_impl.function
        impl_name = (
            getattr(function, "__qualname__", None)
            or getattr(function, "__name__", None)
            or type(function).__qualname__
        )
        self.stages.append(
            WrapperAuditStage(
                self.hook_name,
                layer_kind=layer_kind,
                plugin_name=hook_impl.plugin_name,
                impl_name=impl_name,
                incoming_exception=type(incoming) if incoming is not None else None,
                outgoing_exception=type(outgoing) if outgoing is not None else None,
                action=action,
            )
        )

    def _set_outcome(self, exception: BaseException | None) -> None:
        self.outcome_exception = type(exception) if exception is not None else None

    def __repr__(self) -> str:
        outcome = (
            self.outcome_exception.__name__
            if self.outcome_exception is not None
            else "result"
        )
        return (
            f"<WrapperAuditReport {self.hook_name!r} "
            f"stages={len(self.stages)} outcome={outcome}>"
        )


@final
class WrapperAuditFailure:
    """An independent report of a failure of the wrapper-audit log callback.

    A failing audit callback never changes the hook result; the failure is
    collected here instead, see
    :attr:`pluggy.PluginManager.wrapper_audit_failures`.
    """

    __slots__ = ("error", "hook_name")

    def __init__(self, hook_name: str, error: Exception) -> None:
        """:meta private:"""
        #: Name of the hook which was being audited.
        self.hook_name: Final = hook_name
        #: The exception raised by the audit callback.
        self.error: Final = error

    def __repr__(self) -> str:
        return f"<WrapperAuditFailure {self.hook_name!r} error={self.error!r}>"
