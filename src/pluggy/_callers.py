"""
Call loop machinery
"""

from __future__ import annotations

from collections.abc import Generator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import cast
from typing import NoReturn
from typing import TYPE_CHECKING
from typing import TypeAlias
import warnings

from ._audit import _current_audit
from ._audit import BROKEN
from ._audit import REPLACED
from ._audit import RESTORED
from ._audit import RETAINED
from ._audit import RETURNED
from ._audit import WrapperAudit
from ._audit import WrapperAuditEntry
from ._hooks import HookImpl
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning


# Need to distinguish between old- and new-style hook wrappers.
# Wrapping with a tuple is the fastest type-safe way I found to do it.
Teardown: TypeAlias = Generator[None, object, object]


def run_old_style_hookwrapper(
    hook_impl: HookImpl, hook_name: str, args: Sequence[object]
) -> Teardown:
    """
    backward compatibility wrapper to run a old style hookwrapper as a wrapper
    """
    if TYPE_CHECKING:
        teardown = cast(Teardown, hook_impl.function(*args))
    else:
        teardown = hook_impl.function(*args)
    try:
        next(teardown)
    except StopIteration:
        _raise_wrapfail(teardown, "did not yield")
    try:
        res = yield
        result = Result(res, None)
    except BaseException as exc:
        result = Result(None, exc)
    try:
        teardown.send(result)
    except StopIteration:
        pass
    except BaseException as e:
        _warn_teardown_exception(hook_name, hook_impl, e)
        raise
    else:
        _raise_wrapfail(teardown, "has second yield")
    finally:
        teardown.close()
    return result.get_result()


def _raise_wrapfail(
    wrap_controller: Generator[None, object, object],
    msg: str,
) -> NoReturn:
    co = wrap_controller.gi_code  # type: ignore[attr-defined]
    raise RuntimeError(
        f"wrap_controller at {co.co_name!r} {co.co_filename}:{co.co_firstlineno} {msg}"
    )


def _warn_teardown_exception(
    hook_name: str, hook_impl: HookImpl, e: BaseException
) -> None:
    msg = (
        f"A plugin raised an exception during an old-style hookwrapper teardown.\n"
        f"Plugin: {hook_impl.plugin_name}, Hook: {hook_name}\n"
        f"{type(e).__name__}: {e}\n"
        f"For more information see https://pluggy.readthedocs.io/en/stable/api_reference.html#pluggy.PluggyTeardownRaisedWarning"
    )
    warnings.warn(PluggyTeardownRaisedWarning(msg), stacklevel=6)


def _audit_teardown(
    audit: WrapperAudit | None,
    hook_impl: HookImpl,
    wrapper_kind: str,
    before_exc: BaseException | None,
    after_exc: BaseException | None,
) -> None:
    """Record how a single wrapper layer handled the outcome.

    The audit is passive: it never calls the wrapper and never compares
    exception strings.  It only observes the outcome *before* and *after*
    the wrapper's post-yield block ran.  Exceptions are compared by identity
    (``is``), so a retained exception is the very same object that was
    passed in.  The report only retains the exception type and the layer
    identity (``plugin_name``); result values are never recorded.
    """
    if audit is None:
        return
    if before_exc is not None:
        if after_exc is not None:
            # Same object passed through unchanged -> retained (保留);
            # a different exception -> replaced (替换).
            action = RETAINED if after_exc is before_exc else REPLACED
        else:
            # An exception was turned into a result -> restored (恢复),
            # i.e. the wrapper "took over" (已接管).
            action = RESTORED
    else:
        if after_exc is not None:
            # A result was turned into an exception -> thrown (抛出).
            action = BROKEN
        else:
            # A result was passed through -> returned (放行).
            action = RETURNED
    audit.record(
        WrapperAuditEntry(
            plugin_name=hook_impl.plugin_name,
            wrapper=wrapper_kind,
            action=action,
            input_is_exception=before_exc is not None,
            input_type=type(before_exc).__name__ if before_exc is not None else None,
            output_is_exception=after_exc is not None,
            output_type=type(after_exc).__name__ if after_exc is not None else None,
        )
    )


def _multicall(
    hook_name: str,
    hook_impls: Sequence[HookImpl],
    caller_kwargs: Mapping[str, object],
    firstresult: bool,
) -> object | list[object]:
    """Execute a call into multiple python functions/methods and return the
    result(s).

    ``caller_kwargs`` comes from HookCaller.__call__().
    """
    __tracebackhide__ = True
    results: list[object] = []
    exception = None
    # Each teardown is paired with the hook implementation that produced it
    # and the wrapper protocol it uses, so the wrapper audit can attribute
    # an outcome to the correct layer.
    teardowns: list[tuple[Teardown, HookImpl, str]] = []
    audit = _current_audit.get()
    try:  # run impl and wrapper setup functions in a loop
        for hook_impl in reversed(hook_impls):
            try:
                args = [caller_kwargs[argname] for argname in hook_impl.argnames]
            except KeyError as e:
                raise HookCallError(
                    f"hook call must provide argument {e.args[0]!r}"
                ) from e

            if hook_impl.hookwrapper:
                function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)

                next(function_gen)  # first yield
                teardowns.append((function_gen, hook_impl, "hookwrapper"))

            elif hook_impl.wrapper:
                res = hook_impl.function(*args)
                # If this cast is not valid, a type error is raised below,
                # which is the desired response.
                if TYPE_CHECKING:
                    function_gen = cast(Generator[None, object, object], res)
                else:
                    function_gen = res
                try:
                    next(function_gen)  # first yield
                except StopIteration:
                    _raise_wrapfail(function_gen, "did not yield")
                teardowns.append((function_gen, hook_impl, "wrapper"))
            else:
                res = hook_impl.function(*args)
                if res is not None:
                    results.append(res)
                    if firstresult:  # halt further impl calls
                        break
    except BaseException as exc:
        exception = exc
    finally:
        if firstresult:  # first result hooks return a single value
            result = results[0] if results else None
        else:
            result = results

        # run all wrapper post-yield blocks
        for teardown, hook_impl, wrapper_kind in reversed(teardowns):
            before_exc = exception
            try:
                if exception is not None:
                    try:
                        teardown.throw(exception)
                    except RuntimeError as re:
                        # StopIteration from generator causes RuntimeError
                        # even for coroutine usage - see #544
                        if (
                            isinstance(exception, StopIteration)
                            and re.__cause__ is exception
                        ):
                            teardown.close()
                            _audit_teardown(
                                audit,
                                hook_impl,
                                wrapper_kind,
                                before_exc,
                                exception,
                            )
                            continue
                        else:
                            raise
                else:
                    teardown.send(result)
                # Following is unreachable for a well behaved hook wrapper.
                # Try to force finalizers otherwise postponed till GC action.
                # Note: close() may raise if generator handles GeneratorExit.
                teardown.close()
            except StopIteration as si:
                result = si.value
                exception = None
                _audit_teardown(audit, hook_impl, wrapper_kind, before_exc, None)
                continue
            except BaseException as e:
                exception = e
                _audit_teardown(audit, hook_impl, wrapper_kind, before_exc, e)
                continue
            _raise_wrapfail(teardown, "has second yield")

    if exception is not None:
        raise exception
    else:
        return result
