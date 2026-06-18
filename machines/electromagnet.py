"""Electromagnet (lifting magnet) worked example (spec §8) — the magnetic machine.

Two roles, like a stripped-down motor without the shaft, but the output is a **holding/lift force**
rather than torque — and the force is *quadratic* in current (Maxwell stress ∝ B²), which is the
new physics versus the motor's linear torque.

* ``core`` — ``magnetism`` (permeability) and ``curie_temperature`` (must stay magnetic at the
  operating temperature). M3 + M4. Lower ``density`` is better for lift-per-weight.
* ``coil`` — ``conductivity_continuous`` + ``melting_temperature`` + ``thermal_conductivity``
  (current and its I²R burnout limit, via :mod:`machines._electrical`). M3 + M5 + M6b.

Performance::

    current  = coil current, burnout-limited (a bare coil: no external load resistance)
    flux     = magnetism * max(0, 1 - T_ambient / T_curie)    # demagnetizes near Tc; 0 if non-magnetic
    field    = FIELD_K * flux * current                       # ampere-turns x permeability
    lift     = field ** 2                                     # Maxwell stress ~ B^2 (quadratic in I)
    specific_lift = lift / core_density                       # lift per unit core mass

No gates: a non-ferromagnetic core (no Curie point) gives zero flux → zero field → zero lift, and a
non-conducting coil carries no current → zero field. The quadratic ``lift ∝ I²`` makes a
low-resistance, high-burnout coil pay off even harder than in the motor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._electrical import coil_current
from .roles import Blueprint, Requirement, Role, _props

ELECTROMAGNET = Blueprint(
    name="electromagnet",
    roles=(
        Role(
            name="core",
            description="the magnetic core the field rides on; high permeability, stays magnetic hot",
            requirements=(
                Requirement("magnetism", ref=0.60, description="permeability / flux"),
                Requirement("curie_temperature", ref=2.0, description="stays magnetic at operating T"),
            ),
        ),
        Role(
            name="coil",
            description="drives the ampere-turns; low resistance + high burnout headroom",
            requirements=(
                Requirement("conductivity_continuous", ref=0.12, description="more current"),
                Requirement("melting_temperature", ref=2.5, description="burnout headroom"),
                Requirement("thermal_conductivity", ref=0.13, description="sheds I^2R heat"),
            ),
        ),
    ),
)

# Fixed design/calibration scales (not per-material dials).
COIL_R0: float = 1.0       # coil resistance scale: R = COIL_R0 / sigma
FIELD_K: float = 20.0      # field constant: field = FIELD_K * flux * current
BURNOUT_K: float = 0.15    # same I^2R-burnout scale as the motor coil


@dataclass(frozen=True)
class OperatingPoint:
    """Where the magnet is energized — supply ``voltage`` at an ``ambient_temperature``."""

    voltage: float = 1.0
    ambient_temperature: float = 0.5


@dataclass(frozen=True)
class ElectromagnetPerformance:
    lift_force: float
    field: float
    current: float
    flux: float
    specific_lift: float          # lift per unit core mass — the figure of merit
    burnout_current: float
    limiting_factor: str          # "ohmic" | "burnout"
    demagnetized: bool
    suitabilities: dict[str, float]

    def summary(self) -> str:
        lim = self.limiting_factor + (", core DEMAGNETIZED" if self.demagnetized else "")
        suit = "  ".join(f"{k}={v:.2f}" for k, v in self.suitabilities.items())
        return (
            f"lift_force={self.lift_force:.4f}  field={self.field:.4f}  current={self.current:.4f}\n"
            f"flux={self.flux:.4f}  specific_lift (per mass) = {self.specific_lift:.5f}\n"
            f"limited by: {lim}\nsuitability: {suit}"
        )


def build_electromagnet(
    core: Any, coil: Any, operating_point: OperatingPoint = OperatingPoint()
) -> ElectromagnetPerformance:
    """Build a lifting magnet from a core + coil; compute its lift force (spec §8).

    Pure function of ``(core, coil, operating_point)``; consumes ``Material.properties`` only.
    """
    core_p = _props(core)
    coil_p = _props(coil)
    op = operating_point

    # Bare coil: no external load resistance, so the ceiling is its own resistance + burnout.
    state = coil_current(
        coil_p, voltage=op.voltage, ambient_temperature=op.ambient_temperature,
        r0=COIL_R0, load_resistance=0.0, burnout_k=BURNOUT_K,
    )

    magnetism = float(core_p.get("magnetism", 0.0))
    t_curie = float(core_p.get("curie_temperature", 0.0))
    if t_curie <= 0.0:
        flux, demagnetized = 0.0, True
    else:
        retained = max(0.0, 1.0 - op.ambient_temperature / t_curie)
        flux, demagnetized = magnetism * retained, retained <= 0.0

    field = FIELD_K * flux * state.current
    lift_force = field ** 2
    core_density = float(core_p.get("density", 0.0))
    specific_lift = lift_force / core_density if core_density > 0.0 else 0.0

    return ElectromagnetPerformance(
        lift_force=lift_force,
        field=field,
        current=state.current,
        flux=flux,
        specific_lift=specific_lift,
        burnout_current=state.burnout_current,
        limiting_factor=state.limiting_factor,
        demagnetized=demagnetized,
        suitabilities=ELECTROMAGNET.suitabilities({"core": core_p, "coil": coil_p}),
    )
