"""The game shell (materials-engine-spec sec 11, M6) — the interactive front-end.

A thin, UI-agnostic game layer ON TOP of the mature engine: it *imports* the engine, the
machine layer, and (later) chemistry, and is never imported by them — the same one-way
layering discipline the machine layer follows. The game adds only what the engine lacks: a
player's session (an inventory of discovered materials), a progression of goals, and a
terminal UI. Every material, property, and machine performance is still produced by the
engine/machines — the shell just makes discovering and using them a loop.

Layout:

* ``naming``   — deterministic coined names for discovered materials (reproducible alchemy).
* ``catalog``  — the buildable machines (role order + the headline figure of merit), a thin
                 generic wrapper over ``machines/``.
* ``goals``    — the progression: discovery and build objectives that auto-complete.
* ``state``    — :class:`GameState`: the registry-backed session + save/load (JSON).
* ``session``  — :class:`Session`: the UI-agnostic command API (discover/build/inspect/goals).
* ``shell``    — the terminal REPL (``python -m game``).
"""
