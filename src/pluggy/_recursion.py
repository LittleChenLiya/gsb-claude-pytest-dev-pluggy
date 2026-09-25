"""
Optional hook recursion protection.

This module implements a per-call-context stack of hook calls which can be
used to detect (and reject) excessive nesting or re-entry of selected hooks.
The stack is stored in a :class:`contextvars.ContextVar`, so every thread and
every asyncio task observes its own independent call chain: identical hook
calls running concurrently in different contexts are never mistaken for
recursion.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from contextvars import ContextVar
from typing import NamedTuple
from typing import TYPE_CHECKING

from ._result import HookCallError


if TYPE_CHECKING:
    from ._hooks import HookImpl


class HookCallFrame(NamedTuple):
    """One entry of the hook call stack tracked by recursion protection."""

    #: Name of the hook being called.
    hook_name: str
    #: Names of the plugins providing implementations for this call.
    plugin_names: tuple[str, ...]
    #: 1-based depth of this call within the current call chain.
    depth: int


# The hook call stack of the current call context. Each thread and each
# asyncio task sees an independent stack, so concurrent calls of the same
# hook in different contexts are not mistaken for recursion.
_call_stack: ContextVar[tuple[HookCallFrame, ...]] = ContextVar(
    "pluggy_hook_call_stack", default=()
)

#: Default maximum number of entries kept in a trimmed call ring.
DEFAULT_MAX_RING_ENTRIES = 20


def _trim_call_ring(
    ring: Sequence[HookCallFrame],
    max_entries: int,
    keep_index: int,
) -> tuple[tuple[HookCallFrame, ...], bool]:
    """Trim ``ring`` to at most ``max_entries`` entries.

    The frame at ``keep_index`` (the first re-entry point) and the last
    frame (the triggering call) are always preserved. Returns the trimmed
    ring and whether trimming occurred.
    """
    if len(ring) <= max_entries:
        return tuple(ring), False
    tail_count = max_entries - 1
    tail = ring[len(ring) - tail_count :]
    if keep_index >= len(ring) - tail_count:
        # The frame to keep is already part of the tail.
        return tuple(tail), True
    return (ring[keep_index], *tail), True


def format_call_ring(ring: Sequence[HookCallFrame], truncated: bool) -> str:
    """Render a call ring as a human readable, numbered chain."""
    lines = []
    for i, frame in enumerate(ring):
        if truncated and i == 1:
            lines.append("  ...")
        plugins = ", ".join(frame.plugin_names) if frame.plugin_names else "<none>"
        lines.append(f"  {frame.depth}. {frame.hook_name} (plugins: {plugins})")
    return "\n".join(lines)


class HookRecursionError(HookCallError):
    """A hook call was rejected by the recursion protection.

    Raised when recursion protection is enabled via
    :meth:`PluginManager.enable_recursion_protection` and a hook call
    either exceeds the configured maximum call-chain depth or re-enters a
    hook which was declared non-reentrant (directly or indirectly).

    :param message: The error message.
    :param call_ring:
        The trimmed :class:`HookCallFrame` chain which led to this
        error. The first re-entry point and the final triggering call are
        always preserved, even when the chain had to be trimmed.
    """

    def __init__(self, message: str, call_ring: tuple[HookCallFrame, ...]) -> None:
        super().__init__(message)
        #: The trimmed call ring which led to this error.
        self.call_ring: tuple[HookCallFrame, ...] = call_ring


class RecursionGuard:
    """Tracks the hook call stack of the current context and enforces
    recursion limits.

    :meta private:
    """

    __slots__ = ("_max_depth", "_max_ring_entries", "_no_reentry")

    def __init__(
        self,
        max_depth: int | None,
        no_reentry: frozenset[str],
        max_ring_entries: int,
    ) -> None:
        self._max_depth = max_depth
        self._no_reentry = no_reentry
        self._max_ring_entries = max_ring_entries

    def guarded_call(
        self,
        inner: Callable[
            [str, Sequence[HookImpl], Mapping[str, object], bool],
            object | list[object],
        ],
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
    ) -> object | list[object]:
        """Run ``inner`` under recursion protection.

        Pushes a :class:`HookCallFrame` for this call before invoking
        the inner hook executor and pops it in a ``finally`` block, so
        the stack is cleaned up consistently no matter how the call ends --
        normal return, ``firstresult`` early exit, wrapper teardown
        exception or historic replay.
        """
        __tracebackhide__ = True
        stack = _call_stack.get()
        depth = len(stack) + 1
        frame = HookCallFrame(
            hook_name=hook_name,
            plugin_names=tuple(impl.plugin_name for impl in hook_impls),
            depth=depth,
        )
        ring = (*stack, frame)
        if self._max_depth is not None and depth > self._max_depth:
            trimmed, truncated = _trim_call_ring(
                ring, self._max_ring_entries, keep_index=0
            )
            raise HookRecursionError(
                f"hook call chain exceeded the maximum depth of {self._max_depth} "
                f"while calling hook {hook_name!r}:\n"
                f"{format_call_ring(trimmed, truncated)}",
                trimmed,
            )
        if hook_name in self._no_reentry:
            for index, entry in enumerate(stack):
                if entry.hook_name == hook_name:
                    trimmed, truncated = _trim_call_ring(
                        ring, self._max_ring_entries, keep_index=index
                    )
                    raise HookRecursionError(
                        f"hook {hook_name!r} was re-entered directly or indirectly "
                        f"but is declared non-reentrant:\n"
                        f"{format_call_ring(trimmed, truncated)}",
                        trimmed,
                    )
        token = _call_stack.set(ring)
        try:
            return inner(hook_name, hook_impls, caller_kwargs, firstresult)
        finally:
            _call_stack.reset(token)

