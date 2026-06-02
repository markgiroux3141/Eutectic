"""Root elements + their base lattice generation (spec §3.1, §4.1).

Root elements are the leaves of the lineage graph. Each has a hand-authored
``signature`` (a stable seed) and a small set of ``base_affinities`` that bias how its
base lattice is generated and, later, how it merges/relaxes.

The affinities are deliberately a *small* set of scalars in [0, 1] (spec §3.1):

* ``bond_energy``         — fill density / structural rigidity (higher -> denser lattice,
                            stronger but more likely to percolate; feeds melting point).
* ``magnetic_tendency``   — initial spin alignment bias (feeds Ising magnetism).
* ``conduction_tendency`` — how uniformly mass concentrates (feeds percolation /
                            conductivity).

~15-30 elements authored by hand to start (spec §3.1). The values below are a first pass
chosen to spread elements across the interesting thresholds; tune them in the explorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .lattice import DEFAULT_SHAPE_2D, Lattice, generate_base
from .rng import UNIVERSE_SEED, hash_str, mix


@dataclass(frozen=True)
class Element:
    """A root material (spec §3.1)."""

    id: str
    display_name: str
    atomic_mass: float
    signature: int
    base_affinities: dict[str, float] = field(default_factory=dict)

    def lattice(
        self,
        *,
        shape: Sequence[int] = DEFAULT_SHAPE_2D,
        universe_seed: int = UNIVERSE_SEED,
    ) -> Lattice:
        """Generate this element's base lattice deterministically (spec §4.1).

        The seed folds ``signature`` with ``UNIVERSE_SEED`` so flipping the universe seed
        reshuffles the whole material space while staying deterministic within a save.
        """
        seed = mix(self.signature, universe_seed)
        return generate_base(
            seed,
            shape=shape,
            affinities=self.base_affinities,
            mass_per_atom=self.atomic_mass,
        )


def _sig(element_id: str) -> int:
    """Derive a stable signature seed from the element id (spec §3.1)."""
    return hash_str(element_id)


def _aff(bond_energy: float, magnetic_tendency: float, conduction_tendency: float) -> dict:
    return {
        "bond_energy": bond_energy,
        "magnetic_tendency": magnetic_tendency,
        "conduction_tendency": conduction_tendency,
    }


# --- Hand-authored root element table -------------------------------------------------
# atomic_mass is roughly real (feeds density); affinities are gameplay knobs in [0,1].
# Signatures are derived from id so they are stable and collision-resistant.

_ELEMENT_TABLE: tuple[tuple[str, str, float, dict], ...] = (
    # id            display          atomic_mass  affinities(bond, magnetic, conduct)
    ("hydrogen",    "Hydrogen",        1.008,  _aff(0.10, 0.05, 0.20)),
    ("lithium",     "Lithium",         6.94,   _aff(0.30, 0.10, 0.80)),
    ("carbon",      "Carbon",         12.011,  _aff(0.85, 0.05, 0.55)),
    ("oxygen",      "Oxygen",         15.999,  _aff(0.25, 0.10, 0.15)),
    ("sodium",      "Sodium",         22.990,  _aff(0.30, 0.10, 0.85)),
    ("magnesium",   "Magnesium",      24.305,  _aff(0.45, 0.10, 0.70)),
    ("aluminium",   "Aluminium",      26.982,  _aff(0.55, 0.08, 0.90)),
    ("silicon",     "Silicon",        28.085,  _aff(0.70, 0.05, 0.50)),  # semiconductor-ish
    ("sulfur",      "Sulfur",         32.06,   _aff(0.35, 0.05, 0.20)),
    ("titanium",    "Titanium",       47.867,  _aff(0.75, 0.15, 0.60)),
    ("chromium",    "Chromium",       51.996,  _aff(0.70, 0.40, 0.65)),
    ("iron",        "Iron",           55.845,  _aff(0.65, 0.85, 0.70)),  # ferromagnet
    ("cobalt",      "Cobalt",         58.933,  _aff(0.68, 0.80, 0.68)),  # ferromagnet
    ("nickel",      "Nickel",         58.693,  _aff(0.66, 0.78, 0.72)),  # ferromagnet
    ("copper",      "Copper",         63.546,  _aff(0.60, 0.05, 0.97)),  # great conductor
    ("zinc",        "Zinc",           65.38,   _aff(0.50, 0.05, 0.75)),
    ("silver",      "Silver",        107.868,  _aff(0.58, 0.05, 0.98)),  # best conductor
    ("tin",         "Tin",           118.710,  _aff(0.45, 0.05, 0.78)),
    ("tungsten",    "Tungsten",      183.84,   _aff(0.95, 0.20, 0.60)),  # high melting pt
    ("platinum",    "Platinum",      195.084,  _aff(0.80, 0.10, 0.85)),
    ("gold",        "Gold",          196.967,  _aff(0.62, 0.05, 0.95)),
    ("mercury",     "Mercury",       200.592,  _aff(0.20, 0.05, 0.70)),
    ("lead",        "Lead",          207.2,    _aff(0.40, 0.05, 0.65)),
    ("uranium",     "Uranium",       238.029,  _aff(0.78, 0.25, 0.55)),
    ("niobium",     "Niobium",        92.906,  _aff(0.72, 0.15, 0.88)),  # superconductor-leaning
)


ELEMENTS: dict[str, Element] = {
    eid: Element(
        id=eid,
        display_name=name,
        atomic_mass=mass,
        signature=_sig(eid),
        base_affinities=aff,
    )
    for (eid, name, mass, aff) in _ELEMENT_TABLE
}


def get(element_id: str) -> Element:
    """Look up a root element by id, with a helpful error on typo."""
    try:
        return ELEMENTS[element_id]
    except KeyError:
        raise KeyError(
            f"unknown element {element_id!r}; known: {', '.join(sorted(ELEMENTS))}"
        ) from None


def all_ids() -> list[str]:
    """All root element ids, sorted (sorted for determinism — spec §6.2)."""
    return sorted(ELEMENTS)
