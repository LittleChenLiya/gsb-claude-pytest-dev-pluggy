"""
Optional per-call profiling of hook executions.
"""

from __future__ import annotations

from collections.abc import Callable
import contextvars
import threading
import time
from typing import Final
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import final


if TYPE_CHECKING:
    from ._hooks import HookImpl


#: A callable returning a monotonically increasing time in seconds.
Clock: TypeAlias = Callable[[], float]

# The profile of the hook call currently executing in this thread/context,
# or None when no profiled hook call is active. Stored in a ContextVar so
# that recursive calls link to their parent call and concurrent threads
# never see each other's call.
_current_profile: contextvars.ContextVar[HookCallProfile | None] = (
    contextvars.ContextVar("pluggy_current_call_profile", default=None)
)


@final
class ImplTiming:
    """Timing and outcome of a single hook implementation execution.

    Only implementations which were actually invoked during a hook call get
    an :class:`ImplTiming`; implementations skipped because of
    :ref:`firstresult` or an earlier exception are not recorded at all.

    No argument values or return values are stored, only the plugin name,
    the outcome and the measured durations.
    """

    __slots__ = (
        "inner_duration",
        "is_wrapper",
        "own_duration",
        "plugin_name",
        "status",
    )

    def __init__(self, plugin_name: str, is_wrapper: bool) -> None:
        #: The name of the plugin which provides the implementation.
        self.plugin_name: Final = plugin_name
        #: Whether the implementation is a (new- or old-style) hook wrapper.
        self.is_wrapper: Final = is_wrapper
        #: The outcome: "running", "ok" or "exception".
        self.status = "running"
        #: Seconds spent in the implementation itself. For wrappers this is
        #: the setup plus teardown time, excluding the wrapped inner calls.
        self.own_duration = 0.0
        #: Seconds spent in the inner calls wrapped by this implementation.
        #: Always 0.0 for non-wrapper implementations.
        self.inner_duration = 0.0

    @property
    def total_duration(self) -> float:
        """Total seconds between entering and exiting the implementation."""
        return self.own_duration + self.inner_duration

    def __repr__(self) -> str:
        return (
            f"<ImplTiming {self.plugin_name!r} status={self.status!r} "
            f"own={self.own_duration:.6f}s inner={self.inner_duration:.6f}s>"
        )


class _Pending:
    """In-flight profiling state for one hook implementation execution."""

    __slots__ = ("entered", "inner_started", "teardown_started", "timing")

    def __init__(self, timing: ImplTiming, entered: float) -> None:
        self.timing = timing
        self.entered = entered
        self.inner_started = 0.0
        self.teardown_started = 0.0


@final
class HookCallProfile:
    """Profile of a single hook call (one 1:N hook execution).

    A new profile is created for every hook call made while profiling is
    enabled. A hook call made from inside another hook call (recursion)
    gets its own profile which is linked to the enclosing call through
    :attr:`parent` and :attr:`children`; timings of the nested call are
    never mixed into the records of the parent call.
    """

    __slots__ = (
        "_dead",
        "_profiler",
        "_start",
        "children",
        "duration",
        "hook_name",
        "impl_timings",
        "parent",
        "status",
    )

    def __init__(
        self,
        profiler: CallProfiler,
        hook_name: str,
        parent: HookCallProfile | None,
    ) -> None:
        self._profiler = profiler
        #: The name of the called hook.
        self.hook_name: Final = hook_name
        #: The profile of the enclosing hook call if this hook was called
        #: from within another hook call, else None.
        self.parent: Final = parent
        #: The profiles of hook calls made from within this hook call.
        self.children: Final[list[HookCallProfile]] = []
        #: The timing of each implementation which was actually executed,
        #: in the order the implementations were entered.
        self.impl_timings: Final[list[ImplTiming]] = []
        #: The outcome of the hook call: "running", "ok" or "exception".
        self.status = "running"
        #: Total wall-clock seconds of the hook call.
        self.duration = 0.0
        self._start = 0.0
        self._dead = False

    def __repr__(self) -> str:
        return (
            f"<HookCallProfile {self.hook_name!r} status={self.status!r} "
            f"duration={self.duration:.6f}s impls={len(self.impl_timings)}>"
        )

    def _abort(self, exc: BaseException) -> None:
        self._dead = True
        self._profiler._fail(exc)

    # The impl_* and wrapper_* methods below are called from _multicall.
    # They never raise: a recorder failure aborts this profile and disables
    # the profiler, leaving the hook call itself unaffected.

    def impl_enter(self, hook_impl: HookImpl, is_wrapper: bool) -> _Pending | None:
        if self._dead:
            return None
        try:
            timing = ImplTiming(hook_impl.plugin_name, is_wrapper)
            pending = _Pending(timing, self._profiler.clock())
            self.impl_timings.append(timing)
        except BaseException as exc:
            self._abort(exc)
            return None
        return pending

    def impl_exit(self, pending: _Pending | None, ok: bool) -> None:
        if pending is None or self._dead:
            return
        try:
            pending.timing.own_duration += self._profiler.clock() - pending.entered
            pending.timing.status = "ok" if ok else "exception"
        except BaseException as exc:
            self._abort(exc)

    def wrapper_setup_done(self, pending: _Pending | None) -> None:
        if pending is None or self._dead:
            return
        try:
            now = self._profiler.clock()
            pending.timing.own_duration += now - pending.entered
            pending.inner_started = now
        except BaseException as exc:
            self._abort(exc)

    def wrapper_teardown_enter(self, pending: _Pending | None) -> None:
        if pending is None or self._dead:
            return
        try:
            now = self._profiler.clock()
            pending.timing.inner_duration += now - pending.inner_started
            pending.teardown_started = now
        except BaseException as exc:
            self._abort(exc)

    def wrapper_teardown_exit(self, pending: _Pending | None, ok: bool) -> None:
        if pending is None or self._dead:
            return
        try:
            now = self._profiler.clock()
            pending.timing.own_duration += now - pending.teardown_started
            pending.timing.status = "ok" if ok else "exception"
        except BaseException as exc:
            self._abort(exc)


@final
class CallProfiler:
    """Records a :class:`HookCallProfile` for every hook call.

    Enable it on a :class:`~pluggy.PluginManager` with
    :py:meth:`pluggy.PluginManager.enable_call_profiling`.

    :param clock:
        A callable returning the current time in seconds, used for all
        measurements. Defaults to :func:`time.perf_counter`. A fake clock
        can be injected for testing.

    The profiler is optional and never changes hook behavior: if the
    recorder itself fails (for example the clock raises), monitoring is
    disabled entirely, the failure is stored in :attr:`error`, and hook
    calls continue unaffected.
    """

    def __init__(self, clock: Clock = time.perf_counter) -> None:
        #: The clock used for all measurements.
        self.clock: Final = clock
        #: The profiles of all completed top-level hook calls, in the order
        #: they finished. Nested (recursive) calls are found in
        #: :attr:`HookCallProfile.children` of their parent call.
        self.calls: Final[list[HookCallProfile]] = []
        #: The first failure of the recorder itself, if any. When set,
        #: monitoring has been disabled.
        self.error: BaseException | None = None
        #: Whether monitoring is still active. Set to False after a
        #: recorder failure.
        self.active = True
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Drop all recorded calls and re-enable monitoring after a failure."""
        with self._lock:
            self.calls.clear()
            self.error = None
            self.active = True

    def _fail(self, exc: BaseException) -> None:
        # Policy: a recorder failure disables monitoring entirely; the hook
        # call in progress and all later calls continue unaffected.
        with self._lock:
            if self.error is None:
                self.error = exc
            self.active = False

    def _begin_call(self, hook_name: str) -> HookCallProfile | None:
        if not self.active:
            return None
        try:
            parent = _current_profile.get()
            profile = HookCallProfile(self, hook_name, parent)
            profile._start = self.clock()
        except BaseException as exc:
            self._fail(exc)
            return None
        if parent is not None:
            parent.children.append(profile)
        return profile

    def _end_call(self, profile: HookCallProfile, ok: bool) -> None:
        try:
            profile.duration = self.clock() - profile._start
            profile.status = "ok" if ok else "exception"
        except BaseException as exc:
            self._fail(exc)
            return
        if profile.parent is None:
            with self._lock:
                self.calls.append(profile)
