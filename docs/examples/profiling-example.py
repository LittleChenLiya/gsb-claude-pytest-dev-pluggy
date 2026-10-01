"""Demo: profiling hook calls with pluggy.CallProfiler.

Prints, for one hook call, each plugin's name and how much time it spent.
Wrappers show their own time separately from the time of the calls they
wrap. Run with:

    python docs/examples/profiling-example.py
"""

import time

from pluggy import CallProfiler
from pluggy import HookCallProfile
from pluggy import HookimplMarker
from pluggy import HookspecMarker
from pluggy import PluginManager


hookspec = HookspecMarker("example")
hookimpl = HookimplMarker("example")


class Spec:
    @hookspec
    def render(self, text: str) -> None: ...


class SlowPlugin:
    @hookimpl
    def render(self, text: str) -> str:
        time.sleep(0.05)
        return text.upper()


class TimingWrapper:
    @hookimpl(wrapper=True)
    def render(self, text: str):
        time.sleep(0.01)  # setup work
        result = yield
        time.sleep(0.02)  # teardown work
        return result


def show(profile: HookCallProfile, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}hook {profile.hook_name}: {profile.duration:.4f}s ({profile.status})")
    for timing in profile.impl_timings:
        if timing.is_wrapper:
            print(
                f"{pad}  {timing.plugin_name}: own={timing.own_duration:.4f}s "
                f"inner={timing.inner_duration:.4f}s ({timing.status})"
            )
        else:
            print(
                f"{pad}  {timing.plugin_name}: "
                f"own={timing.own_duration:.4f}s ({timing.status})"
            )
    for child in profile.children:
        show(child, indent + 1)


def main() -> None:
    pm = PluginManager("example")
    pm.add_hookspecs(Spec)
    pm.register(SlowPlugin(), name="slow-plugin")
    pm.register(TimingWrapper(), name="timing-wrapper")

    profiler = CallProfiler()
    pm.enable_call_profiling(profiler)

    pm.hook.render(text="hello pluggy")

    for profile in profiler.calls:
        show(profile)


if __name__ == "__main__":
    main()
