"""Discovered-materials registry + lineage graph (spec §4.5, §10).

The :class:`Registry` owns the set of materials a session has produced and the cache that
guarantees a combination is only ever computed once (spec §4.5). It also fixes the
per-session config — ``universe_seed``, lattice ``shape``, ``relax_steps`` — so every
material in one registry is built under identical rules.

Combination is keyed by the canonical (sorted) pair of parent ids, so ``combine(a, b)``
and ``combine(b, a)`` hit the same cache entry.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from . import elements as elements_mod
from . import material as material_mod
from .elements import Element
from .lattice import DEFAULT_SHAPE_2D, RELAX_STEPS
from .material import Material
from .rng import UNIVERSE_SEED


class Registry:
    """A material store + lineage graph for one universe configuration."""

    def __init__(
        self,
        *,
        universe_seed: int = UNIVERSE_SEED,
        shape: tuple[int, ...] = DEFAULT_SHAPE_2D,
        relax_steps: int = RELAX_STEPS,
    ) -> None:
        self.universe_seed = universe_seed
        self.shape = tuple(int(s) for s in shape)
        self.relax_steps = relax_steps
        self._materials: dict[str, Material] = {}
        # canonical (sorted) parent-id tuple -> child material id
        self._combo_cache: dict[tuple[str, str], str] = {}

    # --- population -------------------------------------------------------------------

    def add_element(self, element: Element | str) -> Material:
        """Add a root material from an Element (or element id). Idempotent."""
        el = elements_mod.get(element) if isinstance(element, str) else element
        if el.id in self._materials:
            return self._materials[el.id]
        mat = material_mod.from_element(
            el, shape=self.shape, universe_seed=self.universe_seed
        )
        self._materials[mat.id] = mat
        return mat

    def seed_elements(self, ids: Iterable[str] | None = None) -> list[Material]:
        """Add every root element (or a given subset). Returns the root materials."""
        chosen = elements_mod.all_ids() if ids is None else list(ids)
        return [self.add_element(eid) for eid in chosen]

    # --- access -----------------------------------------------------------------------

    def get(self, material_id: str) -> Material:
        try:
            return self._materials[material_id]
        except KeyError:
            raise KeyError(f"unknown material {material_id!r}") from None

    def __contains__(self, material_id: str) -> bool:
        return material_id in self._materials

    def __len__(self) -> int:
        return len(self._materials)

    def all_ids(self) -> list[str]:
        return sorted(self._materials)

    def materials(self) -> Iterator[Material]:
        for mid in self.all_ids():
            yield self._materials[mid]

    # --- combination ------------------------------------------------------------------

    def combine(self, a_id: str, b_id: str) -> Material:
        """Combine two registered materials, caching by canonical parent pair (spec §4.5)."""
        a = self.get(a_id)
        b = self.get(b_id)
        key = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        cached = self._combo_cache.get(key)
        if cached is not None:
            return self._materials[cached]

        child = material_mod.combine(
            a, b, universe_seed=self.universe_seed, relax_steps=self.relax_steps
        )
        self._materials[child.id] = child
        self._combo_cache[key] = child.id
        return child
