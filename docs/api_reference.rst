:orphan:

.. _`api-reference`:

API Reference
=============

.. autoclass:: pluggy.PluginManager
    :members:

.. autoclass:: pluggy.PluginValidationError
    :show-inheritance:
    :members:

.. autodecorator:: pluggy.HookspecMarker

.. autodecorator:: pluggy.HookimplMarker

.. autoclass:: pluggy.HookRelay()
    :members:

    .. data:: <hook name>

        :type: HookCaller

        The caller for the hook with the given name.

.. autoclass:: pluggy.HookCaller()
    :members:
    :special-members: __call__

.. autoclass:: pluggy.HookCallError()
    :show-inheritance:
    :members:

.. autoclass:: pluggy.Result()
    :show-inheritance:
    :members:

.. autoclass:: pluggy.HookImpl()
    :members:

.. autoclass:: pluggy.HookspecOpts()
    :show-inheritance:
    :members:

.. autoclass:: pluggy.HookimplOpts()
    :show-inheritance:
    :members:


Wrapper audit
-------------

The optional wrapper audit records how exceptions are retained, replaced or
restored to results at each wrapper layer, in real execution order.  It is
purely observational: it never calls a wrapper, never compares exception
strings, and never changes the hook result.  Enable it by passing a
``wrapper_audit`` callback to :class:`~pluggy.PluginManager`.

.. autoclass:: pluggy.WrapperAudit()
    :members:

.. autoclass:: pluggy.WrapperAuditEntry()
    :members:


Warnings
--------

Custom warnings generated in some situations such as improper usage or deprecated features.

.. autoclass:: pluggy.PluggyWarning()
    :show-inheritance:

.. autoclass:: pluggy.PluggyTeardownRaisedWarning()
    :show-inheritance:

.. autoclass:: pluggy.PluggyWrapperAuditWarning()
    :show-inheritance:
