"""Power-cable worked example (spec §8) — the electrical-transmission machine.

A power cable is a one-role assembly: a ``conductor`` that carries power over a *distance*. Unlike
the motor coil (a fixed lump), the cable's resistance grows with length, so the new physics is
transmission loss and ampacity over distance, plus the weight/sag of a heavy conductor.

* ``conductivity_continuous`` — sets resistance (and so loss). M3.
* ``melting_temperature`` + ``thermal_conductivity`` — the ampacity (max current before it melts),
  via the shared I²R-burnout math (:mod:`machines._electrical`). M5 + M6b.
* ``ductility`` — manufacturability (can it be drawn into cable). M8.
* ``density`` — **lower is better**: a heavy cable sags and needs more support. M2.

Performance (deliver power ``P`` at voltage ``V`` over distance ``L``)::

    R          = R0 * L / sigma                       # resistance grows with distance
    current    = P / V                                # current needed to push power P
    efficiency = max(0, 1 - current * R / V)          # 1 - I^2 R / P : fraction not lost as heat
    ampacity   = burnout_current(R, kappa, T_melt, ...)   # melts above this
    overheated = current > ampacity
    mass       = density * L                          # weight ~ length (heavy -> sag)

Story: a high-σ conductor (tungsten, titanium, copper) transmits efficiently; a non-conductor
(carbon, lead — σ = 0) gives ``R = inf`` → efficiency 0 (useless), and a conductor with no thermal
headroom (melting point at/below ambient) has zero ampacity and overheats. The light + conductive
sweet spot (titanium) beats the best raw conductor (tungsten) once weight matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._electrical import burnout_current, wire_resistance
from .roles import Blueprint, Requirement, Role, _props

CABLE = Blueprint(
    name="power_cable",
    roles=(
        Role(
            name="conductor",
            description="carries power over distance with low loss; light + conductive + high-melting wins",
            requirements=(
                Requirement("conductivity_continuous", ref=0.12, description="low resistance -> low loss"),
                Requirement("melting_temperature", ref=2.5, description="ampacity headroom"),
                Requirement("thermal_conductivity", ref=0.13, description="sheds I^2R heat"),
                Requirement("ductility", ref=0.40, description="manufacturability: drawn into cable"),
                Requirement("density", ref=40.0, higher_is_better=False, description="lighter -> less sag"),
            ),
        ),
    ),
)

# Fixed design/calibration scales (not per-material dials).
CABLE_R0: float = 1.0       # resistance scale: R = CABLE_R0 * length / sigma
BURNOUT_K: float = 0.15     # same I^2R-burnout scale as the motor coil


@dataclass(frozen=True)
class TransmissionLoad:
    """The operating point: deliver ``power`` at ``voltage`` over ``distance``, at an ambient T."""

    power: float = 0.03
    distance: float = 1.0
    voltage: float = 1.0
    ambient_temperature: float = 0.5


@dataclass(frozen=True)
class CablePerformance:
    resistance: float
    current: float                # P / V, the current the line must carry
    efficiency: float             # fraction of sent power delivered (not lost as I^2R heat)
    loss_fraction: float
    ampacity: float               # max current before the cable melts
    overheated: bool
    mass: float                   # ~ density * length (sag/weight)
    suitability: float

    def summary(self) -> str:
        flag = "  OVERHEATED" if self.overheated else ""
        return (
            f"efficiency={self.efficiency:.3f}  loss={self.loss_fraction:.3f}  "
            f"current={self.current:.4f}{flag}\n"
            f"resistance={self.resistance:.4f}  ampacity={self.ampacity:.4f}  mass={self.mass:.2f}\n"
            f"suitability: conductor={self.suitability:.2f}"
        )


def build_cable(conductor: Any, load: TransmissionLoad = TransmissionLoad()) -> CablePerformance:
    """Build a power cable from one material; compute transmission performance (spec §8).

    Pure function of ``(conductor, load)``; consumes ``Material.properties`` only.
    """
    props = _props(conductor)
    sigma = float(props.get("conductivity_continuous", 0.0))
    resistance = wire_resistance(sigma, r0=CABLE_R0, length=load.distance)

    current = load.power / load.voltage if load.voltage > 0.0 else 0.0
    if resistance == float("inf"):
        efficiency = 0.0
    else:
        efficiency = max(0.0, 1.0 - current * resistance / load.voltage)
    loss_fraction = 1.0 - efficiency

    ampacity = burnout_current(
        resistance=resistance,
        thermal_conductivity=float(props.get("thermal_conductivity", 0.0)),
        melting_temperature=float(props.get("melting_temperature", 0.0)),
        ambient_temperature=load.ambient_temperature,
        burnout_k=BURNOUT_K,
    )
    overheated = current > ampacity
    mass = float(props.get("density", 0.0)) * load.distance

    return CablePerformance(
        resistance=resistance,
        current=current,
        efficiency=efficiency,
        loss_fraction=loss_fraction,
        ampacity=ampacity,
        overheated=overheated,
        mass=mass,
        suitability=CABLE.role("conductor").suitability(props),
    )
