:orphan:

.. _`api-reference`:

API Reference
=============

.. autoclass:: pluggy.PluginManager
    :members:

Plugin aliases
--------------

A plugin alias is an alternate name that resolves to a plugin's canonical
name. The canonical name is the only true registration identity; aliases are
merely handles used to query or block the same registered plugin.

Use :meth:`~pluggy.PluginManager.add_alias` to register an alias. Alias chains
are normalized before being stored, so an alias may point to another alias.
Cycles (e.g. ``a -> b -> a``) and conflicts (the alias already points to a
different canonical name) are rejected with a :class:`ValueError`.

Once an alias is registered, :meth:`~pluggy.PluginManager.get_plugin`,
:meth:`~pluggy.PluginManager.has_plugin`, :meth:`~pluggy.PluginManager.is_blocked`,
:meth:`~pluggy.PluginManager.set_blocked`, :meth:`~pluggy.PluginManager.unblock`
and :meth:`~pluggy.PluginManager.unregister` all resolve aliases to their
canonical name first. Blocking therefore takes effect by canonical identity:
blocking through an alias blocks the canonical name too, and vice versa.

Registering a plugin under an alias also resolves the alias to its canonical
name, so the plugin is registered under the canonical name rather than the
alias.

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


Warnings
--------

Custom warnings generated in some situations such as improper usage or deprecated features.

.. autoclass:: pluggy.PluggyWarning()
    :show-inheritance:

.. autoclass:: pluggy.PluggyTeardownRaisedWarning()
    :show-inheritance:
