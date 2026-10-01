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

from ._hooks import HookImpl
from ._profiling import _current_profile
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning


if TYPE_CHECKING:
    from ._profiling import _Pending


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
    teardowns: list[Teardown] = []
    # The profile of this hook call when call profiling is enabled, else
    # None. See pluggy._profiling and PluginManager.enable_call_profiling.
    profile = _current_profile.get()
    # In-flight profiling state for each teardown, parallel to teardowns.
    pendings: list[_Pending | None] = []
    try:  # run impl and wrapper setup functions in a loop
        for hook_impl in reversed(hook_impls):
            try:
                args = [caller_kwargs[argname] for argname in hook_impl.argnames]
            except KeyError as e:
                raise HookCallError(
                    f"hook call must provide argument {e.args[0]!r}"
                ) from e

            if hook_impl.hookwrapper:
                pending = (
                    profile.impl_enter(hook_impl, is_wrapper=True)
                    if profile is not None
                    else None
                )
                try:
                    function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
                    next(function_gen)  # first yield
                except BaseException:
                    if profile is not None:
                        profile.impl_exit(pending, ok=False)
                    raise
                if profile is not None:
                    profile.wrapper_setup_done(pending)
                teardowns.append(function_gen)
                pendings.append(pending)

            elif hook_impl.wrapper:
                pending = (
                    profile.impl_enter(hook_impl, is_wrapper=True)
                    if profile is not None
                    else None
                )
                try:
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
                except BaseException:
                    if profile is not None:
                        profile.impl_exit(pending, ok=False)
                    raise
                if profile is not None:
                    profile.wrapper_setup_done(pending)
                teardowns.append(function_gen)
                pendings.append(pending)
            else:
                if profile is not None:
                    pending = profile.impl_enter(hook_impl, is_wrapper=False)
                    try:
                        res = hook_impl.function(*args)
                    except BaseException:
                        profile.impl_exit(pending, ok=False)
                        raise
                    profile.impl_exit(pending, ok=True)
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
        for i in range(len(teardowns) - 1, -1, -1):
            teardown = teardowns[i]
            pending = pendings[i]
            if profile is not None:
                profile.wrapper_teardown_enter(pending)
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
                            if profile is not None:
                                profile.wrapper_teardown_exit(pending, ok=False)
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
                if profile is not None:
                    profile.wrapper_teardown_exit(pending, ok=True)
                continue
            except BaseException as e:
                exception = e
                if profile is not None:
                    profile.wrapper_teardown_exit(pending, ok=False)
                continue
            if profile is not None:
                profile.wrapper_teardown_exit(pending, ok=False)
            _raise_wrapfail(teardown, "has second yield")

    if exception is not None:
        raise exception
    else:
        return result
