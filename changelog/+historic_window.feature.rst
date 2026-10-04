Added support for bounding the memory of historic hooks:
:meth:`pluggy.HookCaller.set_history_limit` retains only the most recent
``N`` historic call records and reports the trimmed sequence range via
:meth:`pluggy.HookCaller.history_dropped_range`, without re-invoking
trimmed ``result_callback``\ s. Late-registered plugins replay only the
retained window by default; :meth:`pluggy.PluginManager.register` gained
a ``require_complete_history`` flag which refuses registration up front
(without calling the plugin) when records were trimmed.
:meth:`pluggy.HookCaller.call_historic` gained an explicit ``snapshot``
function parameter so recorded arguments are immune to later mutation of
the caller's kwargs mapping; the default (no snapshot) behavior is
unchanged.
