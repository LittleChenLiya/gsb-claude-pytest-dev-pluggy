"""
Tests for the optional wrapper audit (``PluginManager.enable_wrapper_audit``).

Run only these targeted tests with:

    pytest testing/test_wrapper_audit.py -v

or a single acceptance case, e.g.:

    pytest testing/test_wrapper_audit.py -v -k three_layer
"""

from __future__ import annotations

import pytest

from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager
from pluggy import WrapperAuditReport
from pluggy import WrapperAuditStage


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class E1(Exception):
    pass


class E2(Exception):
    pass


class Hooks:
    @hookspec
    def he_method1(self, arg):
        pass


def make_pm(*plugins: object) -> PluginManager:
    pm = PluginManager("example")
    pm.add_hookspecs(Hooks)
    for plugin in plugins:
        pm.register(plugin, name=type(plugin).__name__)
    return pm


def collect(pm: PluginManager) -> list[WrapperAuditReport]:
    reports: list[WrapperAuditReport] = []
    pm.enable_wrapper_audit(reports.append)
    return reports


def test_inner_raises_outer_replaces_reports_two_stages() -> None:
    """Inner wrapper raises E1, outer wrapper replaces it with E2."""

    class Inner:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            yield
            raise E1("inner")

    class Outer:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except E1:
                raise E2("outer")

    pm = make_pm(Inner(), Outer())
    reports = collect(pm)

    with pytest.raises(E2):
        pm.hook.he_method1(arg=1)

    (report,) = reports
    assert len(report.stages) == 2
    inner, outer = report.stages
    # Real execution order: the inner wrapper teardown runs first.
    assert inner.layer_kind == "wrapper"
    assert inner.incoming_exception is None  # a result arrived
    assert inner.outgoing_exception is E1
    assert inner.action == "raised"
    assert outer.layer_kind == "wrapper"
    assert outer.incoming_exception is E1
    assert outer.outgoing_exception is E2
    assert outer.action == "replaced"
    # Only types are kept, never instances.
    assert report.outcome_exception is E2


def test_wrapper_recovers_value_shows_taken_over() -> None:
    """A wrapper restoring a result is reported as recovered (took over)."""

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("boom")

    class Recovering:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except E1:
                return 42

    pm = make_pm(Impl(), Recovering())
    reports = collect(pm)

    assert pm.hook.he_method1(arg=1) == 42

    (report,) = reports
    assert [(s.layer_kind, s.action) for s in report.stages] == [
        ("impl", "raised"),
        ("wrapper", "recovered"),
    ]
    recovered = report.stages[-1]
    assert recovered.incoming_exception is E1
    assert recovered.outgoing_exception is None  # a result left the layer
    assert report.outcome_exception is None
    assert not report.raised


def test_old_style_hookwrapper_follows_original_protocol() -> None:
    """Old-style hookwrappers are observed via the Result object protocol."""

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("boom")

    class ReplacingOldWrapper:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            result = yield
            result.force_exception(E2("replacement"))

    pm = make_pm(Impl(), ReplacingOldWrapper())
    reports = collect(pm)

    with pytest.raises(E2):
        pm.hook.he_method1(arg=1)

    (report,) = reports
    assert [(s.layer_kind, s.action) for s in report.stages] == [
        ("impl", "raised"),
        ("hookwrapper", "replaced"),
    ]
    assert report.stages[-1].incoming_exception is E1
    assert report.stages[-1].outgoing_exception is E2


def test_old_style_hookwrapper_preserve_and_recover() -> None:
    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("boom")

    class PreservingOldWrapper:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            yield  # leave the Result untouched

    pm = make_pm(Impl(), PreservingOldWrapper())
    reports = collect(pm)

    with pytest.raises(E1):
        pm.hook.he_method1(arg=1)

    (report,) = reports
    assert report.stages[-1].action == "preserved"
    assert report.stages[-1].incoming_exception is E1
    assert report.stages[-1].outgoing_exception is E1

    class RecoveringOldWrapper:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            result = yield
            result.force_result([7])

    pm2 = make_pm(Impl(), RecoveringOldWrapper())
    reports2 = collect(pm2)

    assert pm2.hook.he_method1(arg=1) == [7]
    (report2,) = reports2
    assert report2.stages[-1].action == "recovered"
    assert report2.stages[-1].layer_kind == "hookwrapper"


def test_three_layer_example() -> None:
    """Three wrapper layers, mixed protocols, in real execution order.

    Layer 1 (innermost, old-style hookwrapper) preserves E1.
    Layer 2 (middle, new-style wrapper) replaces E1 with E2.
    Layer 3 (outermost, new-style wrapper) recovers a result (takes over).
    """

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("from impl")

    # Registered first -> innermost wrapper layer.
    class Layer1OldStyle:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            yield  # preserve the exception

    class Layer2NewStyle:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except E1:
                raise E2("replaced")

    # Registered last -> outermost wrapper layer.
    class Layer3NewStyle:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            try:
                yield
            except E2:
                return "taken-over"

    pm = make_pm(Impl(), Layer1OldStyle(), Layer2NewStyle(), Layer3NewStyle())
    reports = collect(pm)

    assert pm.hook.he_method1(arg=1) == "taken-over"

    (report,) = reports
    assert [(s.layer_kind, s.action) for s in report.stages] == [
        ("impl", "raised"),
        ("hookwrapper", "preserved"),
        ("wrapper", "replaced"),
        ("wrapper", "recovered"),
    ]
    assert [s.outgoing_exception for s in report.stages] == [E1, E1, E2, None]
    # Layer identity is recorded for every stage.
    assert {s.plugin_name for s in report.stages} == {
        "Impl",
        "Layer1OldStyle",
        "Layer2NewStyle",
        "Layer3NewStyle",
    }
    assert report.outcome_exception is None


def test_audit_does_not_change_result_or_exception() -> None:
    """Final result/exception is identical with and without auditing."""

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            return arg + 1

    class Wrapper:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            res = yield
            return [x * 10 for x in res]

    plain = make_pm(Impl(), Wrapper())
    audited = make_pm(Impl(), Wrapper())
    reports = collect(audited)

    assert audited.hook.he_method1(arg=1) == plain.hook.he_method1(arg=1) == [20]
    (report,) = reports
    assert [s.action for s in report.stages] == ["passed"]

    class FailingImpl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("boom")

    plain2 = make_pm(FailingImpl())
    audited2 = make_pm(FailingImpl())
    collect(audited2)

    with pytest.raises(E1) as excinfo_plain:
        plain2.hook.he_method1(arg=1)
    with pytest.raises(E1) as excinfo_audited:
        audited2.hook.he_method1(arg=1)
    assert type(excinfo_audited.value) is type(excinfo_plain.value)


def test_audit_does_not_increase_call_counts() -> None:
    """Enabling the audit must not call impls or wrappers more often."""
    counts = {"impl": 0, "wrapper": 0, "hookwrapper": 0}

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            counts["impl"] += 1
            raise E1("boom")

    class OldWrapper:
        @hookimpl(hookwrapper=True)
        def he_method1(self, arg):
            counts["hookwrapper"] += 1
            yield

    class NewWrapper:
        @hookimpl(wrapper=True)
        def he_method1(self, arg):
            counts["wrapper"] += 1
            try:
                yield
            except E1:
                return "recovered"

    plain = make_pm(Impl(), OldWrapper(), NewWrapper())
    before = counts.copy()
    assert plain.hook.he_method1(arg=1) == "recovered"
    plain_counts = {k: counts[k] - before[k] for k in counts}

    audited = make_pm(Impl(), OldWrapper(), NewWrapper())
    reports = collect(audited)
    before = counts.copy()
    assert audited.hook.he_method1(arg=1) == "recovered"
    audited_counts = {k: counts[k] - before[k] for k in counts}

    assert (
        audited_counts
        == plain_counts
        == {
            "impl": 1,
            "wrapper": 1,
            "hookwrapper": 1,
        }
    )
    assert len(reports) == 1


def test_report_keeps_only_types_and_layer_identity() -> None:
    class Impl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("secret message")

    pm = make_pm(Impl())
    reports = collect(pm)

    with pytest.raises(E1):
        pm.hook.he_method1(arg=1)

    (report,) = reports
    (stage,) = report.stages
    assert isinstance(stage, WrapperAuditStage)
    assert stage.outgoing_exception is E1  # a type, not an instance
    assert not isinstance(stage.outgoing_exception, E1)
    assert stage.layer_kind == "impl"
    assert stage.plugin_name == "Impl"
    assert "he_method1" in stage.impl_name
    assert "secret message" not in repr(report)


def test_auditor_failure_is_reported_independently() -> None:
    """A failing log callback neither changes the hook result nor is lost."""

    class Impl:
        @hookimpl
        def he_method1(self, arg):
            return arg

    pm = make_pm(Impl())

    def bad_auditor(report: WrapperAuditReport) -> None:
        raise RuntimeError("log backend down")

    pm.enable_wrapper_audit(bad_auditor)

    # Hook result is unaffected by the failing auditor.
    assert pm.hook.he_method1(arg=5) == [5]

    (failure,) = pm.wrapper_audit_failures
    assert failure.hook_name == "he_method1"
    assert isinstance(failure.error, RuntimeError)
    assert "log backend down" in str(failure.error)

    # Also holds when the hook itself raises.
    class FailingImpl:
        @hookimpl
        def he_method1(self, arg):
            raise E1("boom")

    pm2 = make_pm(FailingImpl())
    pm2.enable_wrapper_audit(bad_auditor)
    with pytest.raises(E1):
        pm2.hook.he_method1(arg=1)
    assert len(pm2.wrapper_audit_failures) == 1


def test_undo_disables_audit() -> None:
    class Impl:
        @hookimpl
        def he_method1(self, arg):
            return arg

    pm = make_pm(Impl())
    reports: list[WrapperAuditReport] = []
    undo = pm.enable_wrapper_audit(reports.append)
    assert pm.hook.he_method1(arg=1) == [1]
    assert len(reports) == 1

    undo()
    assert pm.hook.he_method1(arg=1) == [1]
    assert len(reports) == 1
