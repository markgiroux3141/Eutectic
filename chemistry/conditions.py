"""Chemical conditions: ``ChemConditions(T, P, concentration, catalysts)`` (spec §13).

Extends :class:`engine.Conditions` (the ``(T, P, H)`` dial set the materials engine already
threads through ``measure(structure, conditions)``) with the two dials chemistry adds:

* **concentration** — activity of the reacting species; shifts equilibria and rates
  (Le Chatelier, mass action).
* **catalysts** — a frozen set of species present that lower specific activation barriers
  (C4); carried here so the same conditions object threads thermo (C3) and kinetics (C4).

``T`` and ``P`` are inherited unchanged (already live in the materials engine). The
quantize-before-seeding discipline (spec §14.4) is extended: :meth:`seed_key` folds the two
new dials into the deterministic seed so a reaction measured at the same conditions twice is
byte-identical.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace

from engine.conditions import _SEED_SCALE, Conditions

# Standard chemical state: unit activity, no catalysts (the reference for ΔG° / K).
STANDARD_CONCENTRATION: float = 1.0


class Phase(enum.Enum):
    """Aggregation state of a species (spec §4). Sets the entropy a species contributes."""

    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"

    @property
    def is_gas(self) -> bool:
        return self is Phase.GAS


@dataclass(frozen=True)
class ChemConditions(Conditions):
    """A point in chemical dial-space: ``(T, P, H)`` + concentration + catalysts (spec §13).

    Frozen value object; ``catalysts`` is a ``frozenset`` so the dataclass stays hashable.
    ``concentration`` is a single scalar activity (a coarse stand-in for per-species
    activities — enough for the Le Chatelier keystone; richer per-species activities are
    future work).
    """

    concentration: float = STANDARD_CONCENTRATION
    catalysts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.concentration <= 0.0:
            # ln(activity) appears in ΔG; a non-positive activity is undefined.
            raise ValueError(f"concentration must be > 0, got {self.concentration}")

    def seed_key(self) -> tuple[int, ...]:
        """Quantized dials as integers (spec §14.4), extending the ``(T, P, H)`` base key.

        Appends the quantized concentration and a stable hash of the sorted catalyst set, so
        two ``ChemConditions`` that quantize equal seed identically.
        """
        from engine.rng import hash_str

        base = super().seed_key()
        conc = int(round(self.concentration * _SEED_SCALE))
        cats = hash_str("|".join(sorted(self.catalysts)))
        return (*base, conc, cats)

    def with_temperature(self, temperature: float) -> "ChemConditions":
        """A copy at a different temperature (the common move when sweeping T)."""
        return replace(self, temperature=temperature)

    def with_pressure(self, pressure: float) -> "ChemConditions":
        """A copy at a different pressure (the Le Chatelier P sweep)."""
        return replace(self, pressure=pressure)

    def with_concentration(self, concentration: float) -> "ChemConditions":
        """A copy at a different concentration (the Le Chatelier / mass-action sweep)."""
        return replace(self, concentration=concentration)

    def has_catalyst(self, species: str) -> bool:
        return species in self.catalysts


# The reference chemical state (ΔG° / K are defined here): standard T, P, unit activity.
STANDARD_CHEM = ChemConditions()
