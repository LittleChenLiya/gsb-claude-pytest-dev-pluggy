"""
Tests for the optional wrapper audit.

The wrapper audit records how exceptions are retained, replaced or restored
to results at each wrapper layer, in real execution order.  It is purely
observational: it never calls a wrapper, never compares exception strings,
and never changes the hook result.
"""

from __future__ import annotations

import warnings

import pytest

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy import WrapperAudit
from pluggy import WrapperAuditEntry
from pluggy._audit import BROKEN
from pluggy._audit import REPLACED
from pluggy._audit import RESTORED
from pluggy._audit import RETAINED
from pluggy._audit import RETURNED
from pluggy._warnings import PluggyWrapperAuditWarning


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class _Hooks:
    @hookspec
    def he_method1(self, arg: object) -> None:
        pass


def _make_pm(audit_cb=None) -> PluginManager:
    pm = PluginManager("example", wrapper_audit=audit_cb)
    pm.add_hookspecs(_Hooks)
    return pm


def _collect(audits: list[WrapperAudit]):
    def cb(audit: WrapperAudit) -> None:
        audits.append(audit)

    return cb


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_audit_disabled_by_default() -> None:
    """Without ``wrapper_audit`` the hook behaves exactly as before and no
    audit is produced."""
    pm = _make_pm()

    class P:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            return arg

    pm.register(P(), name="P")
    assert pm.hook.he_method1(arg=1) == 1


def test_audit_collected_in_real_execution_order() -> None:
    """Inner wrappers run first (post-yield), so entries are appended
    innermost-first -- the real execution order."""
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class Inner:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            return "inner"

    class Outer:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            return "outer"

    pm.register(Inner(), name="inner")
    pm.register(Outer(), name="outer")

    pm.hook.he_method1(arg=1)

    assert len(audits) == 1
    assert [e.plugin_name for e in audits[0].entries] == ["inner", "outer"]


def test_audit_entry_repr() -> None:
    entry = WrapperAuditEntry(
        plugin_name="p",
        wrapper="wrapper",
        action=RETAINED,
        input_is_exception=True,
        input_type="ValueError",
        output_is_exception=True,
        output_type="ValueError",
    )
    assert "ValueError" in repr(entry)
    assert "p" in repr(entry)


# ---------------------------------------------------------------------------
# Retain / replace / restore
# ---------------------------------------------------------------------------


def test_inner_throws_e1_outer_replaces_e2_two_phases() -> None:
    """内层抛E1、外层替换E2时报告两阶段.

    The inner wrapper retains E1; the outer wrapper replaces it with E2.
    The report has two entries (two phases), innermost first.
    """
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class Inner:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except Exception:  # noqa: TRY203
                raise  # retain E1

    class Outer:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except Exception:
                raise OSError("E2")  # replace E1 -> E2

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(Inner(), name="inner")
    pm.register(Outer(), name="outer")
    pm.register(Impl(), name="impl")

    with pytest.raises(OSError, match="E2"):
        pm.hook.he_method1(arg=1)

    assert len(audits) == 1
    entries = audits[0].entries
    assert len(entries) == 2
    # phase 1: inner retains E1
    assert entries[0].plugin_name == "inner"
    assert entries[0].action == RETAINED
    assert entries[0].input_type == "ValueError"
    assert entries[0].output_type == "ValueError"
    # phase 2: outer replaces E1 with E2
    assert entries[1].plugin_name == "outer"
    assert entries[1].action == REPLACED
    assert entries[1].input_type == "ValueError"
    assert entries[1].output_type == "OSError"


def test_wrapper_restores_value_shows_taken_over() -> None:
    """另一wrapper恢复值时显示已接管.

    A wrapper that turns an exception into a result is recorded as
    "恢复" (restored / 已接管).
    """
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class Restorer:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except Exception:
                return "taken-over"  # restore

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(Restorer(), name="restorer")
    pm.register(Impl(), name="impl")

    assert pm.hook.he_method1(arg=1) == "taken-over"

    entries = audits[0].entries
    assert len(entries) == 1
    assert entries[0].action == RESTORED
    assert entries[0].input_is_exception is True
    assert entries[0].input_type == "ValueError"
    assert entries[0].output_is_exception is False
    assert entries[0].output_type is None


def test_result_passthrough() -> None:
    """A result passed through a wrapper is recorded as 放行 (returned)."""
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class W:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            return arg

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            return arg + 1

    pm.register(W(), name="w")
    pm.register(Impl(), name="impl")

    # The wrapper's return value becomes the hook result.
    assert pm.hook.he_method1(arg=1) == 1

    entries = audits[0].entries
    assert len(entries) == 1
    assert entries[0].action == RETURNED
    assert entries[0].input_is_exception is False
    assert entries[0].output_is_exception is False


def test_result_turned_into_exception() -> None:
    """A wrapper that turns a result into an exception is recorded as
    抛出 (broken)."""
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class W:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            raise OSError("broken")

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            return 1

    pm.register(W(), name="w")
    pm.register(Impl(), name="impl")

    with pytest.raises(OSError, match="broken"):
        pm.hook.he_method1(arg=1)

    entries = audits[0].entries
    assert len(entries) == 1
    assert entries[0].action == BROKEN
    assert entries[0].input_is_exception is False
    assert entries[0].output_is_exception is True
    assert entries[0].output_type == "OSError"


# ---------------------------------------------------------------------------
# Old-style hookwrappers
# ---------------------------------------------------------------------------


def test_old_style_hookwrapper_retain() -> None:
    """Old-style hookwrappers are handled per their original protocol (a
    ``Result`` object) and recorded the same way."""
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class W:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            yield
            # retain: do nothing

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(W(), name="w")
    pm.register(Impl(), name="impl")

    with pytest.raises(ValueError, match="E1"):
        pm.hook.he_method1(arg=1)

    entries = audits[0].entries
    assert len(entries) == 1
    assert entries[0].wrapper == "hookwrapper"
    assert entries[0].action == RETAINED


def test_old_style_hookwrapper_replace() -> None:
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class W:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            result = yield
            try:
                result.get_result()
            except BaseException:
                result.force_exception(OSError("E2"))

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(W(), name="w")
    pm.register(Impl(), name="impl")

    with pytest.raises(OSError, match="E2"):
        pm.hook.he_method1(arg=1)

    entries = audits[0].entries
    assert len(entries) == 1
    assert entries[0].wrapper == "hookwrapper"
    assert entries[0].action == REPLACED
    assert entries[0].input_type == "ValueError"
    assert entries[0].output_type == "OSError"


def test_old_style_hookwrapper_restore() -> None:
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class W:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            result = yield
            try:
                result.get_result()
            except BaseException:
                result.force_result("taken-over")

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(W(), name="w")
    pm.register(Impl(), name="impl")

    assert pm.hook.he_method1(arg=1) == "taken-over"

    entries = audits[0].entries
    assert len(entries) == 1
    assert entries[0].wrapper == "hookwrapper"
    assert entries[0].action == RESTORED


# ---------------------------------------------------------------------------
# Transparency: audit does not change hook behaviour
# ---------------------------------------------------------------------------


def test_audit_does_not_change_hook_result() -> None:
    """The final result or exception is identical with and without audit."""

    def build(audits):
        pm = _make_pm(_collect(audits) if audits is not None else None)

        class W1:
            @hookimpl(wrapper=True)
            def he_method1(self, arg):
                try:
                    yield
                except Exception:  # noqa: TRY203
                    raise

        class W2:
            @hookimpl(wrapper=True)
            def he_method1(self, arg):
                try:
                    yield
                except Exception:
                    raise OSError("E2")

        class Impl:
            @hookimpl
            def he_method1(self, arg):
                raise ValueError("E1")

        pm.register(W1(), name="w1")
        pm.register(W2(), name="w2")
        pm.register(Impl(), name="impl")
        return pm

    pm1 = build(None)
    with pytest.raises(OSError, match="E2"):
        pm1.hook.he_method1(arg=1)

    pm2 = build([])
    with pytest.raises(OSError, match="E2"):
        pm2.hook.he_method1(arg=1)


def test_audit_does_not_increase_call_count() -> None:
    """调用计数不增加.

    The audit never calls a wrapper, so the number of wrapper/impl
    invocations is the same with and without auditing.
    """
    counts = {"wrapper": 0, "impl": 0}

    class W:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            counts["wrapper"] += 1
            yield
            return arg

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            counts["impl"] += 1
            return arg + 1

    def run(audit_cb):
        pm = _make_pm(audit_cb)
        pm.register(W(), name="w")
        pm.register(Impl(), name="impl")
        return pm.hook.he_method1(arg=1)

    without = run(None)
    counts_without = dict(counts)
    counts["wrapper"] = 0
    counts["impl"] = 0
    with_audit = run(lambda a: None)
    counts_with = dict(counts)

    # The wrapper's return value (arg) becomes the hook result.
    assert without == with_audit == 1
    assert counts_without == counts_with


# ---------------------------------------------------------------------------
# Audit failures are reported independently
# ---------------------------------------------------------------------------


def test_audit_callback_failure_does_not_change_hook_result() -> None:
    """日志回调失败不改变hook结果.

    A failing audit callback is reported via a warning; the hook result is
    returned unchanged.
    """

    def bad_cb(a):
        raise RuntimeError("callback boom")

    pm = _make_pm(bad_cb)

    class W:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            return 42

    pm.register(W(), name="w")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pm.hook.he_method1(arg=1) == 42

    audit_warnings = [
        w for w in caught if issubclass(w.category, PluggyWrapperAuditWarning)
    ]
    assert len(audit_warnings) == 1
    assert "callback boom" in str(audit_warnings[0].message)


def test_audit_callback_failure_on_exception_path() -> None:
    """Even when the hook itself raises, a failing audit callback does not
    replace the hook exception."""

    def bad_cb(a):
        raise RuntimeError("callback boom")

    pm = _make_pm(bad_cb)

    class W:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except Exception:  # noqa: TRY203
                raise

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(W(), name="w")
    pm.register(Impl(), name="impl")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="E1"):
            pm.hook.he_method1(arg=1)

    audit_warnings = [
        w for w in caught if issubclass(w.category, PluggyWrapperAuditWarning)
    ]
    assert len(audit_warnings) == 1


# ---------------------------------------------------------------------------
# Three-layer example
# ---------------------------------------------------------------------------


def test_three_layer_example() -> None:
    """三层示例.

    Three wrapper layers around an implementation that raises E1:

    * inner wrapper retains E1 (保留);
    * middle wrapper restores E1 to a result (恢复 / 已接管);
    * outer wrapper passes the result through (放行).

    The report has three entries in real execution order (innermost first).
    """
    audits: list[WrapperAudit] = []
    pm = _make_pm(_collect(audits))

    class Inner:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except Exception:  # noqa: TRY203
                raise  # retain

    class Middle:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except Exception:
                return "taken-over"  # restore / 已接管

    class Outer:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            return "outer-result"  # pass through

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise ValueError("E1")

    pm.register(Inner(), name="inner")
    pm.register(Middle(), name="middle")
    pm.register(Outer(), name="outer")
    pm.register(Impl(), name="impl")

    assert pm.hook.he_method1(arg=1) == "outer-result"

    entries = audits[0].entries
    assert len(entries) == 3
    assert [e.plugin_name for e in entries] == ["inner", "middle", "outer"]
    assert entries[0].action == RETAINED
    assert entries[1].action == RESTORED
    assert entries[2].action == RETURNED
    # The restored entry shows "已接管" behaviour: exception in, result out.
    assert entries[1].input_is_exception is True
    assert entries[1].output_is_exception is False
