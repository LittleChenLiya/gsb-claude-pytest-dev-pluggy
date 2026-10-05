"""
Wrapper audit: observe how exceptions are retained, replaced or restored
to results at each wrapper layer.

The audit is *passive*: it never calls a wrapper itself and never compares
exception strings.  It only records, per wrapper layer, the exception type
(or that a result was passed) and the layer identity (the plugin name).  The
recorded entries are collected in real execution order -- innermost wrapper
first, since that is the order in which the post-yield blocks actually run.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import final


# ---------------------------------------------------------------------------
# Audit action labels.
#
# The vocabulary mirrors the task description: an exception is either
# retained (保留), replaced (替换) or restored to a result (恢复, displayed
# as 已接管).  A result may be passed through (放行) or turned into an
# exception (抛出).
# ---------------------------------------------------------------------------

#: An exception was passed through unchanged (same object identity).
RETAINED: str = "保留"

#: An exception was replaced by a different exception.
REPLACED: str = "替换"

#: An exception was restored to a result -- the wrapper "took over" (已接管).
RESTORED: str = "恢复"

#: A result was passed through unchanged.
RETURNED: str = "放行"

#: A result was turned into an exception.
BROKEN: str = "抛出"


@final
class WrapperAuditEntry:
    """Audit record for a single wrapper layer's handling of an outcome.

    The report only retains the exception *type* and the layer identity
    (``plugin_name``); exception values and strings are never recorded.
    """

    __slots__ = (
        "action",
        "input_is_exception",
        "input_type",
        "output_is_exception",
        "output_type",
        "plugin_name",
        "wrapper",
    )

    def __init__(
        self,
        plugin_name: str,
        wrapper: str,
        action: str,
        input_is_exception: bool,
        input_type: str | None,
        output_is_exception: bool,
        output_type: str | None,
    ) -> None:
        #: The name of the plugin which defined this wrapper (layer identity).
        self.plugin_name: str = plugin_name
        #: The wrapper protocol: ``"wrapper"`` (new-style) or
        #: ``"hookwrapper"`` (old-style).
        self.wrapper: str = wrapper
        #: The action performed by the wrapper -- one of the
        #: :data:`RETAINED`, :data:`REPLACED`, :data:`RESTORED`,
        #: :data:`RETURNED`, :data:`BROKEN` constants.
        self.action: str = action
        #: Whether the input to the wrapper was an exception.
        self.input_is_exception: bool = input_is_exception
        #: The exception type name of the input, or ``None`` for a result.
        self.input_type: str | None = input_type
        #: Whether the output of the wrapper was an exception.
        self.output_is_exception: bool = output_is_exception
        #: The exception type name of the output, or ``None`` for a result.
        self.output_type: str | None = output_type

    def __repr__(self) -> str:
        return (
            f"<WrapperAuditEntry wrapper={self.wrapper!r} "
            f"plugin={self.plugin_name!r} action={self.action!r} "
            f"input={self.input_type if self.input_is_exception else 'result'} "
            f"output={self.output_type if self.output_is_exception else 'result'}>"
        )


@final
class WrapperAudit:
    """Collects wrapper audit entries for a single hook call.

    Entries are appended in real execution order -- the innermost wrapper
    first, because that is the order in which the wrapper post-yield blocks
    actually execute.
    """

    __slots__ = ("entries", "hook_name")

    def __init__(self, hook_name: str) -> None:
        #: The name of the hook being audited.
        self.hook_name: str = hook_name
        #: The collected audit entries, in real execution order.
        self.entries: list[WrapperAuditEntry] = []

    def record(self, entry: WrapperAuditEntry) -> None:
        """Append an audit entry."""
        self.entries.append(entry)

    def __repr__(self) -> str:
        return f"<WrapperAudit hook={self.hook_name!r} entries={self.entries!r}>"


#: Context variable holding the current hook call's audit collector, or
#: ``None`` when wrapper auditing is disabled.  Set by
#: :meth:`PluginManager._hookexec` and read by :func:`_multicall`.
_current_audit: ContextVar[WrapperAudit | None] = ContextVar(
    "pluggy_wrapper_audit", default=None
)
