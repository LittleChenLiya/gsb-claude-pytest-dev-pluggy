Added :meth:`PluginManager.set_hook_impl_order() <pluggy.PluginManager.set_hook_impl_order>`
which allows declaring explicit call-order constraints (``before``/``after``)
between the normal hook implementations of a hook, referenced by plugin name.
Invalid configurations (unknown plugins, plugins not implementing the hook,
tryfirst/trylast/wrapper implementations, cycles) are rejected with
:class:`pluggy.HookOrderError` while keeping the previous configuration.
