"""The buildable machines — a thin generic wrapper over ``machines/`` for the game.

The machine layer already does the physics; the game only needs a uniform way to (a) know what
roles a machine has and in what order, (b) build it from assigned materials, and (c) read its
single headline figure of merit (for goals and the UI). This table is that adapter and nothing
more — it imports ``machines/`` and never reaches into the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from machines import armor, cable, electromagnet, heatsink, motor


@dataclass(frozen=True)
class MachineSpec:
    """One buildable machine: its roles (in build order) and its headline metric."""

    name: str
    roles: tuple[str, ...]
    build: Callable[..., Any]          # build(*materials_in_role_order) -> performance object
    headline: str                       # attribute on the performance object (higher = better)
    headline_label: str
    blurb: str

    def perform(self, materials: list[Any]) -> Any:
        """Build the machine from materials assigned in role order; return the performance object."""
        if len(materials) != len(self.roles):
            raise ValueError(f"{self.name} needs {len(self.roles)} materials "
                             f"({', '.join(self.roles)}), got {len(materials)}")
        return self.build(*materials)

    def score(self, performance: Any) -> float:
        return float(getattr(performance, self.headline))


CATALOG: dict[str, MachineSpec] = {
    "motor": MachineSpec(
        "motor", ("core", "coil_wire", "shaft"), motor.build_motor,
        "torque", "torque",
        "an electric motor: magnetic core + conductive coil + strong shaft -> torque",
    ),
    "electromagnet": MachineSpec(
        "electromagnet", ("core", "coil"), electromagnet.build_electromagnet,
        "lift_force", "lift",
        "an electromagnet: a core and a coil -> lifting force (lift ~ I^2)",
    ),
    "cable": MachineSpec(
        "cable", ("conductor",), cable.build_cable,
        "efficiency", "efficiency",
        "a power cable: one conductor -> transmission efficiency (low loss)",
    ),
    "heatsink": MachineSpec(
        "heatsink", ("fin",), heatsink.build_heat_sink,
        "conductance", "conductance",
        "a heat sink: one fin material -> thermal conductance (the diamond divergence)",
    ),
    "armor": MachineSpec(
        "armor", ("hard_face", "ductile_backing"), armor.build_armor,
        "protection", "protection",
        "composite armor: a hard face + a ductile backing -> protection (the M8 dilemma, solved)",
    ),
}


def get(name: str) -> MachineSpec:
    try:
        return CATALOG[name]
    except KeyError:
        raise KeyError(f"unknown machine {name!r}; known: {', '.join(CATALOG)}") from None
