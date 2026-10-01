"""
Tests for the optional per-call hook profiler.

See pluggy._profiling and PluginManager.enable_call_profiling.
"""

from __future__ import annotations

import threading

import pytest

from pluggy import CallProfiler
from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class FakeClock:
    """A manually advanced clock for deterministic duration assertions."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Spec:
    @hookspec
    def outer(self, x: int) -> None: ...

    @hookspec
    def inner(self) -> None: ...

    @hookspec(firstresult=True)
    def first(self, x: int) -> None: ...


def make_pm() -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Spec)
    return pm


def test_own_and_inner_durations_with_fake_clock() -> None:
    """A wrapper's own time is setup+teardown; the wrapped inner calls are
    accounted separately, including a recursive inner hook call."""
    clock = FakeClock()
    pm = make_pm()

    class WrapperPlugin:
        @hookimpl(wrapper=True)
        def outer(self, x: int):
            clock.advance(2)  # setup work
            result = yield
            clock.advance(5)  # teardown work
            return result

    class PlainPlugin:
        @hookimpl
        def outer(self, x: int) -> int:
            clock.advance(3)
            pm.hook.inner()
            return x

    class InnerPlugin:
        @hookimpl
        def inner(self) -> None:
            clock.advance(7)

    pm.register(InnerPlugin(), name="inner-p")
    pm.register(PlainPlugin(), name="plain")
    pm.register(WrapperPlugin(), name="wrap")

    profiler = CallProfiler(clock=clock)
    pm.enable_call_profiling(profiler)

    assert pm.hook.outer(x=10) == [10]

    (outer,) = profiler.calls
    assert outer.hook_name == "outer"
    assert outer.status == "ok"
    assert outer.duration == 17

    wrapper, plain = outer.impl_timings
    assert wrapper.is_wrapper
    assert not plain.is_wrapper
    # wrapper: own = setup(2) + teardown(5); inner = plain(3) + inner hook(7)
    assert wrapper.own_duration == 7
    assert wrapper.inner_duration == 10
    assert wrapper.total_duration == 17
    # plain impl: its span includes the nested inner hook call
    assert plain.own_duration == 10
    assert plain.inner_duration == 0
    assert plain.total_duration == 10

    # The recursive call is linked to the parent call but timed separately.
    (inner,) = outer.children
    assert inner.parent is outer
    assert inner.hook_name == "inner"
    assert inner.status == "ok"
    assert inner.duration == 7
    (inner_impl,) = inner.impl_timings
    assert inner_impl.plugin_name == "inner-p"
    assert inner_impl.own_duration == 7


def test_results_and_call_order_identical_with_and_without_profiling() -> None:
    pm = make_pm()
    order: list[str] = []

    class P1:
        @hookimpl
        def outer(self, x: int) -> int:
            order.append("p1")
            return x + 1

    class P2:
        @hookimpl(tryfirst=True)
        def outer(self, x: int) -> None:
            order.append("p2")

    class W1:
        @hookimpl(wrapper=True)
        def outer(self, x: int):
            order.append("w1-setup")
            result = yield
            order.append("w1-teardown")
            return result

    class W2:
        @hookimpl(wrapper=True, trylast=True)
        def outer(self, x: int):
            order.append("w2-setup")
            result = yield
            order.append("w2-teardown")
            return result

    pm.register(P1(), name="p1")
    pm.register(P2(), name="p2")
    pm.register(W1(), name="w1")
    pm.register(W2(), name="w2")

    result_off = pm.hook.outer(x=1)
    order_off = list(order)

    profiler = CallProfiler(clock=FakeClock())
    undo = pm.enable_call_profiling(profiler)
    order.clear()
    result_on = pm.hook.outer(x=1)
    order_on = list(order)
    undo()

    assert result_off == [2]
    assert result_on == result_off
    assert order_on == order_off
    assert order_on == [
        "w1-setup",
        "w2-setup",
        "p2",
        "p1",
        "w2-teardown",
        "w1-teardown",
    ]

    (profile,) = profiler.calls
    assert [t.plugin_name for t in profile.impl_timings] == ["w1", "w2", "p2", "p1"]
    assert all(t.status == "ok" for t in profile.impl_timings)

    # After undo, nothing new is recorded.
    pm.hook.outer(x=5)
    assert len(profiler.calls) == 1


def test_firstresult_stops_and_skips_uncalled_impls() -> None:
    pm = make_pm()
    calls: list[str] = []

    class P1:
        @hookimpl
        def first(self, x: int) -> int:
            calls.append("p1")
            return x

    class P2:
        @hookimpl
        def first(self, x: int) -> int:
            calls.append("p2")
            return x * 2

    class P3:
        @hookimpl
        def first(self, x: int) -> None:
            calls.append("p3")

    pm.register(P1(), name="p1")
    pm.register(P2(), name="p2")
    pm.register(P3(), name="p3")

    # Baseline without profiling: p3 returns None, p2 wins, p1 never runs.
    assert pm.hook.first(x=21) == 42
    assert calls == ["p3", "p2"]

    profiler = CallProfiler(clock=FakeClock())
    pm.enable_call_profiling(profiler)
    calls.clear()
    assert pm.hook.first(x=21) == 42
    assert calls == ["p3", "p2"]

    (profile,) = profiler.calls
    # p1 was never called and must not appear as an executed impl.
    assert [t.plugin_name for t in profile.impl_timings] == ["p3", "p2"]
    assert all(t.status == "ok" for t in profile.impl_timings)


def test_impl_exception_is_recorded() -> None:
    pm = make_pm()

    class Boom:
        @hookimpl
        def outer(self, x: int) -> None:
            raise ValueError("boom")

    class Ok:
        @hookimpl(tryfirst=True)
        def outer(self, x: int) -> int:
            return x

    pm.register(Ok(), name="ok")
    pm.register(Boom(), name="boom")

    profiler = CallProfiler(clock=FakeClock())
    pm.enable_call_profiling(profiler)

    with pytest.raises(ValueError, match="boom"):
        pm.hook.outer(x=1)

    (profile,) = profiler.calls
    assert profile.status == "exception"
    # "ok" ran before "boom" raised; no impl is left out or invented.
    ok, boom = profile.impl_timings
    assert boom.plugin_name == "boom"
    assert boom.status == "exception"
    assert ok.plugin_name == "ok"
    assert ok.status == "ok"


def test_wrapper_recovers_exception() -> None:
    pm = make_pm()

    class Boom:
        @hookimpl
        def outer(self, x: int) -> None:
            raise ValueError("boom")

    class RecoveringWrapper:
        @hookimpl(wrapper=True)
        def outer(self, x: int):
            try:
                yield
            except ValueError:
                return "recovered"

    pm.register(Boom(), name="boom")
    pm.register(RecoveringWrapper(), name="recover")

    # Baseline without profiling.
    assert pm.hook.outer(x=1) == "recovered"

    profiler = CallProfiler(clock=FakeClock())
    pm.enable_call_profiling(profiler)
    assert pm.hook.outer(x=1) == "recovered"

    (profile,) = profiler.calls
    assert profile.status == "ok"
    wrapper, boom = profile.impl_timings
    assert boom.status == "exception"
    # The wrapper recovered the exception, so it exited cleanly.
    assert wrapper.status == "ok"


def test_wrapper_rewrites_exception() -> None:
    pm = make_pm()

    class Boom:
        @hookimpl
        def outer(self, x: int) -> None:
            raise ValueError("boom")

    class RewritingWrapper:
        @hookimpl(wrapper=True)
        def outer(self, x: int):
            try:
                yield
            except ValueError as exc:
                raise RuntimeError("rewritten") from exc

    pm.register(Boom(), name="boom")
    pm.register(RewritingWrapper(), name="rewrite")

    # Baseline without profiling.
    with pytest.raises(RuntimeError, match="rewritten"):
        pm.hook.outer(x=1)

    profiler = CallProfiler(clock=FakeClock())
    pm.enable_call_profiling(profiler)
    with pytest.raises(RuntimeError, match="rewritten"):
        pm.hook.outer(x=1)

    (profile,) = profiler.calls
    assert profile.status == "exception"
    wrapper, boom = profile.impl_timings
    assert boom.status == "exception"
    # The wrapper itself raised a new exception during teardown.
    assert wrapper.status == "exception"


def test_old_style_hookwrapper_is_timed_as_wrapper() -> None:
    clock = FakeClock()
    pm = make_pm()

    class OldWrapper:
        @hookimpl(hookwrapper=True)
        def outer(self, x: int):
            clock.advance(1)
            outcome = yield
            clock.advance(4)
            outcome.force_result([10])

    class Plain:
        @hookimpl
        def outer(self, x: int) -> int:
            clock.advance(3)
            return x

    pm.register(Plain(), name="plain")
    pm.register(OldWrapper(), name="old")

    profiler = CallProfiler(clock=clock)
    pm.enable_call_profiling(profiler)
    assert pm.hook.outer(x=10) == [10]

    (profile,) = profiler.calls
    old, plain = profile.impl_timings
    assert old.is_wrapper
    assert old.own_duration == 5
    assert old.inner_duration == 3
    assert plain.own_duration == 3


def test_concurrent_calls_from_two_threads() -> None:
    pm = make_pm()

    class P:
        @hookimpl
        def outer(self, x: int) -> int:
            return x * 2

    pm.register(P(), name="p")
    profiler = CallProfiler()
    pm.enable_call_profiling(profiler)

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(50):
                assert pm.hook.outer(x=i) == [i * 2]
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    # Each thread's calls are recorded as independent top-level calls.
    assert len(profiler.calls) == 100
    for profile in profiler.calls:
        assert profile.status == "ok"
        assert profile.parent is None
        assert profile.children == []
        assert profile.duration >= 0
        assert [t.plugin_name for t in profile.impl_timings] == ["p"]


def test_recorder_failure_disables_monitoring_but_hooks_keep_working() -> None:
    pm = make_pm()

    class P:
        @hookimpl
        def outer(self, x: int) -> int:
            return x

    pm.register(P(), name="p")

    class FlakyClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            if self.calls > 2:
                raise RuntimeError("clock is broken")
            return float(self.calls)

    profiler = CallProfiler(clock=FlakyClock())
    pm.enable_call_profiling(profiler)

    # The failing clock must not break the hook call.
    assert pm.hook.outer(x=1) == [1]
    assert isinstance(profiler.error, RuntimeError)
    assert profiler.error is not None
    assert "clock is broken" in str(profiler.error)
    assert not profiler.active
    # The broken call's profile is dropped.
    assert profiler.calls == []

    # Monitoring stays off; hooks keep working and nothing is recorded.
    assert pm.hook.outer(x=2) == [2]
    assert profiler.calls == []

    # reset() re-enables monitoring; the still-broken clock fails again
    # without affecting the hook call.
    profiler.reset()
    assert profiler.active
    assert pm.hook.outer(x=3) == [3]
    assert isinstance(profiler.error, RuntimeError)
    assert not profiler.active


def test_no_arguments_or_results_are_stored() -> None:
    pm = make_pm()
    sentinel = object()

    class P:
        @hookimpl
        def outer(self, x: object) -> str:
            return "secret-result"

    pm.register(P(), name="p")
    profiler = CallProfiler(clock=FakeClock())
    pm.enable_call_profiling(profiler)
    pm.hook.outer(x=sentinel)

    (profile,) = profiler.calls
    assert repr(sentinel) not in repr(profile)
    assert "secret-result" not in repr(profile)
    for timing in profile.impl_timings:
        assert repr(sentinel) not in repr(timing)
        assert "secret-result" not in repr(timing)
