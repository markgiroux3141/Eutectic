"""Session: the UI-agnostic command API for the game (materials-engine-spec sec 11).

Every command returns a list of plain ASCII lines (the Windows console is cp1252; never emit
unicode here). The terminal shell just reads a command and prints these lines, so the same API
could back a different UI later. All game logic lives here and in :mod:`game.state`; the shell is
purely I/O.
"""

from __future__ import annotations

from pathlib import Path

from . import catalog, goals as goals_mod
from .state import GameState

# Properties surfaced in the compact inventory row (the rest show in `inspect`).
_HEADLINE_PROPS = ("conductivity", "magnetism", "density", "strength", "melting_temperature")


class Session:
    """Holds a :class:`GameState` and turns commands into printable lines."""

    def __init__(self, state: GameState | None = None) -> None:
        self.state = state or GameState()

    # --- helpers ----------------------------------------------------------------------

    def _goal_lines(self, newly: list[goals_mod.Goal]) -> list[str]:
        out = []
        for g in newly:
            out.append(f"  *** GOAL COMPLETE: {g.title} -- {g.description}")
        if newly and goals_mod.active_goals(self.state.completed):
            out.append("  (type 'goals' to see what opened up)")
        return out

    @staticmethod
    def _props_str(mat) -> str:
        bits = []
        for k in _HEADLINE_PROPS:
            if k in mat.properties:
                bits.append(f"{k[:4]}={mat.properties[k]:.3g}")
        return "  ".join(bits)

    # --- commands ---------------------------------------------------------------------

    def help(self) -> list[str]:
        return [
            "commands:",
            "  inventory [all]         list your materials (elements + discovered alloys)",
            "  inspect <handle>        full measured properties + lineage of a material",
            "  discover <h1> <h2>      combine two materials into a new one",
            "  machines                list buildable machines and their roles",
            "  build <machine> <h...>  build a machine from materials (in role order)",
            "  goals                   active objectives + what you've completed",
            "  save <file> / load <file>   persist or restore a session",
            "  help / quit",
        ]

    def machines(self) -> list[str]:
        out = ["buildable machines:"]
        for spec in catalog.CATALOG.values():
            out.append(f"  {spec.name:<14} roles: {', '.join(spec.roles)}")
            out.append(f"  {'':<14} {spec.blurb}")
        return out

    def inventory(self, show_all: bool = False) -> list[str]:
        items = self.state.inventory()
        elems = [(h, m) for h, m in items if m.is_root]
        combos = [(h, m) for h, m in items if not m.is_root]
        out = [f"inventory: {len(elems)} elements, {len(combos)} discovered alloys"]
        if show_all:
            out.append("-- elements --")
            for h, m in elems:
                out.append(f"  {h:<14} {self._props_str(m)}")
        out.append("-- discovered alloys --" if combos else "  (no alloys yet -- try: discover <h1> <h2>)")
        for h, m in combos:
            out.append(f"  {h:<14} {self._props_str(m)}")
        if not show_all:
            out.append(f"  (+{len(elems)} elements; 'inventory all' to list them)")
        return out

    def inspect(self, handle: str) -> list[str]:
        try:
            mat = self.state.resolve(handle)
        except KeyError as exc:
            return [str(exc)]
        kind = "element" if mat.is_root else "alloy"
        out = [f"{handle}  ({kind})  id={mat.id}"]
        if not mat.is_root:
            parents = "  +  ".join(self.state.handle_of(p) for p in mat.lineage)
            out.append(f"  lineage: {parents}")
        out.append("  properties:")
        for k in sorted(mat.properties):
            out.append(f"    {k:<26} {mat.properties[k]}")
        return out

    def discover(self, handle_a: str, handle_b: str) -> list[str]:
        try:
            child, is_new = self.state.discover(handle_a, handle_b)
        except (KeyError, ValueError) as exc:
            return [str(exc)]
        handle = self.state.handle_of(child.id)
        if not is_new:
            return [f"already known: {handle_a} + {handle_b} -> {handle}",
                    f"  {self._props_str(child)}"]
        newly = self.state.refresh_goals()
        out = [f"discovered {handle}!   ({handle_a} + {handle_b})",
               f"  {self._props_str(child)}"]
        return out + self._goal_lines(newly)

    def build(self, machine: str, *handles: str) -> list[str]:
        try:
            spec = catalog.get(machine)
        except KeyError as exc:
            return [str(exc)]
        try:
            mats = [self.state.resolve(h) for h in handles]
        except KeyError as exc:
            return [str(exc)]
        try:
            perf = spec.perform(mats)
        except ValueError as exc:
            return [str(exc),
                    f"  {machine} roles (in order): {', '.join(spec.roles)}"]
        score = spec.score(perf)
        self.state.record_build(machine, score)
        newly = self.state.refresh_goals()
        out = [f"built a {machine}:  {spec.headline_label} = {score:.4f}"]
        out += ["  " + ln for ln in perf.summary().splitlines()]
        return out + self._goal_lines(newly)

    def goals(self) -> list[str]:
        done = self.state.completed
        active = goals_mod.active_goals(done)
        out = [f"goals: {len(done)}/{len(goals_mod.GOALS)} complete"]
        out.append("-- active --" if active else "-- all available goals complete! --")
        for g in active:
            out.append(f"  [ ] {g.title:<18} {g.description}")
        if done:
            out.append("-- completed --")
            for gid in sorted(done):
                out.append(f"  [x] {goals_mod.BY_ID[gid].title}")
        return out

    def save(self, path: str) -> list[str]:
        self.state.save(path)
        return [f"saved to {path}"]

    def load(self, path: str) -> list[str]:
        if not Path(path).exists():
            return [f"no save file at {path}"]
        self.state = GameState.load(path)
        return [f"loaded {path}",
                f"  {len(list(self.state.combos()))} alloys, "
                f"{len(self.state.completed)}/{len(goals_mod.GOALS)} goals"]
