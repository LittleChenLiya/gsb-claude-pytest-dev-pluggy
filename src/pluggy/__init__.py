__all__ = [
    "HookCallError",
    "HookCaller",
    "HookImpl",
    "HookRelay",
    "HookimplMarker",
    "HookimplOpts",
    "HookspecMarker",
    "HookspecOpts",
    "PluggyTeardownRaisedWarning",
    "PluggyWarning",
    "PluggyWrapperAuditWarning",
    "PluginManager",
    "PluginValidationError",
    "Result",
    "WrapperAudit",
    "WrapperAuditEntry",
    "__version__",
]
from ._audit import WrapperAudit
from ._audit import WrapperAuditEntry
from ._hooks import HookCaller
from ._hooks import HookImpl
from ._hooks import HookimplMarker
from ._hooks import HookimplOpts
from ._hooks import HookRelay
from ._hooks import HookspecMarker
from ._hooks import HookspecOpts
from ._manager import PluginManager
from ._manager import PluginValidationError
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning
from ._warnings import PluggyWarning
from ._warnings import PluggyWrapperAuditWarning


def __getattr__(name: str) -> str:
    if name == "__version__":
        from importlib.metadata import version

        return version("pluggy")

    raise AttributeError(f"module {__name__} has no attribute {name!r}")
