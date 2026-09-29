"""
Optional hook recursion protection.

This module implements an opt-in guard which tracks the active hook call
chain per execution context (thread / async task) and can reject calls
which nest too deeply or re-enter a forbidden hook, directly or
indirectly.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Final
from typing import final

from ._result import HookCallError


if TYPE_CHECKING:
    from ._hooks import _HookExec
    from ._hooks import HookImpl


#: Default maximum total depth of a guarded hook call chain.
DEFAULT_MAX_DEPTH: Final = 50

#: Maximum number of entries kept in the call chain attached to a
#: :class:`HookRecursionError`. The first re-entry point and the last
#: triggering frame are always preserved when trimming.
MAX_CHAIN_ENTRIES: Final = 20


@final
class HookCallFrame:
    """A single entry in a guarded hook call chain.

    Records which hook was entered, which plugins provide the
    implementations being called, and the nesting depth of the call.

    .. versionadded:: 1.7
    """

    __slots__ = ("depth", "hook_name", "plugin_names")

    def __init__(
        self, hook_name: str, plugin_names: tuple[str, ...], depth: int
    ) -> None:
        """:meta private:"""
        #: Name of the hook being called.
        self.hook_name: Final = hook_name
        #: Names of the plugins providing the implementations of this call.
        self.plugin_names: Final = plugin_names
        #: Nesting depth of this call within the chain, starting at 1.
        self.depth: Final = depth

    def __repr__(self) -> str:
        return (
            f"<HookCallFrame #{self.depth} {self.hook_name!r} "
            f"plugins={self.plugin_names!r}>"
        )

    def _format(self) -> str:
        plugins = ", ".join(self.plugin_names) or "<no plugins>"
        return f"#{self.depth} {self.hook_name} ({plugins})"


class HookRecursionError(HookCallError):
    """A hook call was rejected by the recursion guard.

    Raised when a guarded hook call chain exceeds the configured maximum
    depth, or when a hook marked as non-reentrant is re-entered directly
    or indirectly.

    :param message: A description of the violation, including the
        (possibly trimmed) call chain.
    :param call_chain: The trimmed :class:`HookCallFrame` chain leading to
        the violation. The first re-entry point and the last triggering
        frame are always preserved.

    .. versionadded:: 1.7
    """

    def __init__(
        self, message: str, call_chain: Sequence[HookCallFrame]
    ) -> None:
        super().__init__(message)
        #: The (possibly trimmed) call chain leading to this error.
        self.call_chain: Final = tuple(call_chain)


def _trim_call_chain(
    chain: Sequence[HookCallFrame],
    keep_from: int,
    limit: int = MAX_CHAIN_ENTRIES,
) -> list[HookCallFrame]:
    """Trim ``chain`` to at most ``limit`` entries.

    The frame at ``keep_from`` (the original re-entry point) and the last
    frame (the triggering call) are always preserved; the remaining budget
    is filled from the tail of the chain.
    """
    chain = list(chain)
    if len(chain) <= limit:
        return chain
    tail = chain[-(limit - 1) :]
    if keep_from >= len(chain) - len(tail):
        # The frame to keep is already part of the tail.
        return tail
    return [chain[keep_from], *tail]


def _format_call_chain(chain: Sequence[HookCallFrame]) -> str:
    parts: list[str] = []
    prev_depth: int | None = None
    for frame in chain:
        if prev_depth is not None and frame.depth != prev_depth + 1:
            parts.append("...")
        parts.append(frame._format())
        prev_depth = frame.depth
    return " -> ".join(parts)


@final
class _RecursionGuard:
    """Tracks the active hook call chain per execution context.

    Uses a :class:`~contextvars.ContextVar` so independent call chains in
    other threads or async tasks are never mistaken for recursion.
    """

    def __init__(self, *, max_depth: int, no_reentry: Iterable[str]) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.max_depth: Final = max_depth
        self.no_reentry: Final = frozenset(no_reentry)
        self._stack: Final[ContextVar[tuple[HookCallFrame, ...]]] = ContextVar(
            "pluggy_hook_call_chain", default=()
        )

    def current_chain(self) -> tuple[HookCallFrame, ...]:
        """Return the call chain active in the current context."""
        return self._stack.get()

    def hookexec(
        self,
        hook_name: str,
        hook_impls: Sequence[HookImpl],
        caller_kwargs: Mapping[str, object],
        firstresult: bool,
        inner_hookexec: _HookExec,
    ) -> object | list[object]:
        stack = self._stack.get()
        depth = len(stack) + 1
        frame = HookCallFrame(
            hook_name,
            tuple(impl.plugin_name for impl in hook_impls),
            depth,
        )
        chain = (*stack, frame)
        if depth > self.max_depth:
            trimmed = _trim_call_chain(chain, keep_from=0)
            raise HookRecursionError(
                f"hook call chain exceeded the maximum depth of "
                f"{self.max_depth}; call chain: {_format_call_chain(trimmed)}",
                trimmed,
            )
        if hook_name in self.no_reentry:
            for i, entered in enumerate(stack):
                if entered.hook_name == hook_name:
                    trimmed = _trim_call_chain(chain, keep_from=i)
                    raise HookRecursionError(
                        f"hook {hook_name!r} may not be re-entered directly "
                        f"or indirectly; call chain: "
                        f"{_format_call_chain(trimmed)}",
                        trimmed,
                    )
        token = self._stack.set(chain)
        try:
            return inner_hookexec(hook_name, hook_impls, caller_kwargs, firstresult)
        finally:
            # Consistent cleanup for all exit paths: normal returns,
            # firstresult early exits, historic replays and exceptions.
            self._stack.reset(token)

