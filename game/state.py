"""GameState: the player's session, backed by the engine registry (materials-engine-spec sec 11).

The session is a thin layer over :class:`engine.registry.Registry` — the registry already does
deterministic discovery, caching, and lineage; the game adds the player-facing bits: typeable
handles for materials, the set of completed goals, and the best machine each player has built.

Everything is **derivable from a tiny save**: the universe seed, the seeded elements, and the
ordered list of combinations. Because the engine is deterministic, replaying that list rebuilds
the exact same materials (and the same coined handles), so saves are small and reproducible — no
lattices are serialized. Handles, too, are recomputed deterministically on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from engine import elements as elements_mod
from engine.lattice import DEFAULT_SHAPE_2D
from engine.material import Material
from engine.registry import Registry
from engine.rng import UNIVERSE_SEED

from . import goals as goals_mod
from .naming import coin_name


class GameState:
    """A registry-backed game session: inventory, handles, goals, and built machines."""

    def __init__(
        self,
        *,
        universe_seed: int = UNIVERSE_SEED,
        shape: tuple[int, ...] = DEFAULT_SHAPE_2D,
        seed_elements: bool = True,
    ) -> None:
        self.registry = Registry(universe_seed=universe_seed, shape=shape)
        self._handle_to_id: dict[str, str] = {}
        self._id_to_handle: dict[str, str] = {}
        self._combo_ops: list[tuple[str, str]] = []      # ordered (a_id, b_id) for replay
        self.completed: set[str] = set()
        self._best_machines: dict[str, float] = {}
        if seed_elements:
            for mat in self.registry.seed_elements():
                self._assign_handle(mat.id)

    # --- handles ----------------------------------------------------------------------

    def _assign_handle(self, material_id: str) -> str:
        """Give a material a stable, unique, typeable handle (roots: element id; combos: coined)."""
        if material_id in self._id_to_handle:
            return self._id_to_handle[material_id]
        mat = self.registry.get(material_id)
        base = material_id if mat.is_root else coin_name(material_id)
        handle = base
        n = 2
        while handle in self._handle_to_id:        # deterministic collision resolution
            handle = f"{base}{n}"
            n += 1
        self._handle_to_id[handle] = material_id
        self._id_to_handle[material_id] = handle
        return handle

    def handle_of(self, material_id: str) -> str:
        return self._id_to_handle[material_id]

    def resolve(self, handle: str) -> Material:
        """Material for a handle (or raise a friendly KeyError)."""
        try:
            return self.registry.get(self._handle_to_id[handle])
        except KeyError:
            raise KeyError(f"no material with handle {handle!r}") from None

    # --- inventory views --------------------------------------------------------------

    def inventory(self) -> list[tuple[str, Material]]:
        """(handle, material) for everything discovered, elements first then combos, sorted."""
        items = [(self._id_to_handle[mid], self.registry.get(mid)) for mid in self.registry.all_ids()]
        items.sort(key=lambda hm: (not hm[1].is_root, hm[0]))
        return items

    def combos(self) -> Iterator[Material]:
        """Discovered combinations only (lineage length 2) — what discovery goals check."""
        for mid in self.registry.all_ids():
            mat = self.registry.get(mid)
            if not mat.is_root:
                yield mat

    # --- discovery + building ---------------------------------------------------------

    def discover(self, handle_a: str, handle_b: str) -> tuple[Material, bool]:
        """Combine two materials; return (child, is_new). Deterministic via the registry."""
        a = self.resolve(handle_a)
        b = self.resolve(handle_b)
        child = self.registry.combine(a.id, b.id)   # deterministic + cached by the registry
        is_new = child.id not in self._id_to_handle
        if is_new:
            self._assign_handle(child.id)
            self._combo_ops.append((a.id, b.id))
        return child, is_new

    def best_machine(self, name: str) -> float | None:
        return self._best_machines.get(name)

    def record_build(self, name: str, score: float) -> None:
        prev = self._best_machines.get(name)
        if prev is None or score > prev:
            self._best_machines[name] = score

    # --- goals ------------------------------------------------------------------------

    def refresh_goals(self) -> list[goals_mod.Goal]:
        """Complete any active goal whose predicate now holds; return the newly completed ones."""
        newly: list[goals_mod.Goal] = []
        # loop to a fixed point: completing a goal can activate (and immediately satisfy) another
        while True:
            progressed = False
            for goal in goals_mod.active_goals(self.completed):
                try:
                    done = goal.check(self)
                except Exception:
                    done = False
                if done:
                    self.completed.add(goal.id)
                    newly.append(goal)
                    progressed = True
            if not progressed:
                break
        return newly

    # --- persistence (small + deterministic; replays the combo list) ------------------

    def to_dict(self) -> dict:
        return {
            "universe_seed": self.registry.universe_seed,
            "shape": list(self.registry.shape),
            "seeded_elements": [m.id for m in self.registry.materials() if m.is_root],
            "combos": [list(op) for op in self._combo_ops],
            "completed": sorted(self.completed),
            "best_machines": self._best_machines,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GameState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        state = cls(
            universe_seed=int(data["universe_seed"]),
            shape=tuple(data["shape"]),
            seed_elements=False,
        )
        # re-seed exactly the saved elements (handles for roots = element id)
        for eid in data["seeded_elements"]:
            mat = state.registry.add_element(eid)
            state._assign_handle(mat.id)
        # replay combinations in order -> identical materials + coined handles (determinism)
        for a_id, b_id in data["combos"]:
            child = state.registry.combine(a_id, b_id)
            state._assign_handle(child.id)
            state._combo_ops.append((a_id, b_id))
        state.completed = set(data.get("completed", []))
        state._best_machines = {k: float(v) for k, v in data.get("best_machines", {}).items()}
        return state
