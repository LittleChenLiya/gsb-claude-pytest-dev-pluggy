"""
Three-layer wrapper audit example.

This example shows how the optional wrapper audit records the way an
exception is handled at each wrapper layer, in real execution order.

Run it with::

    python docs/examples/wrapper-audit-example.py

Expected output (the inner wrapper runs first, post-yield, so the audit
entries are collected innermost-first):

    hook result: outer-result
    audit for hook 'myhook':
      1. wrapper=wrapper      plugin=inner   action=保留  in=ValueError  out=ValueError
      2. wrapper=wrapper      plugin=middle  action=恢复  in=ValueError  out=result
      3. wrapper=wrapper      plugin=outer   action=放行  in=result      out=result
"""

import pluggy

hookspec = pluggy.HookspecMarker("myproject")
hookimpl = pluggy.HookimplMarker("myproject")


class MySpec:
    """Hook specification namespace."""

    @hookspec
    def myhook(self, arg):
        """My special little hook."""


# A simple logger that prints the audit report.
def log_audit(audit: pluggy.WrapperAudit) -> None:
    print(f"audit for hook {audit.hook_name!r}:")
    for i, entry in enumerate(audit.entries, start=1):
        inp = entry.input_type if entry.input_is_exception else "result"
        out = entry.output_type if entry.output_is_exception else "result"
        print(
            f"  {i}. wrapper={entry.wrapper:<12} "
            f"plugin={entry.plugin_name:<8} action={entry.action}  "
            f"in={inp}  out={out}"
        )


pm = pluggy.PluginManager("myproject", wrapper_audit=log_audit)
pm.add_hookspecs(MySpec)


class Inner:
    """Retains the exception (保留)."""

    @hookimpl(wrapper=True)
    def myhook(self, arg):
        try:
            yield
        except Exception:
            raise


class Middle:
    """Restores the exception to a result -- takes over (恢复 / 已接管)."""

    @hookimpl(wrapper=True)
    def myhook(self, arg):
        try:
            yield
        except Exception:
            return "taken-over"


class Outer:
    """Passes the result through (放行)."""

    @hookimpl(wrapper=True)
    def myhook(self, arg):
        yield
        return "outer-result"


class Impl:
    """Raises an exception that the wrappers handle."""

    @hookimpl
    def myhook(self, arg):
        raise ValueError("boom")


pm.register(Inner(), name="inner")
pm.register(Middle(), name="middle")
pm.register(Outer(), name="outer")
pm.register(Impl(), name="impl")

result = pm.hook.myhook(arg=1)
print("hook result:", result)
