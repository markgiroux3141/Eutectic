"""Electric-motor worked example (spec §8) — the machine-layer payoff loop.

A motor is an assembly of three roles, each reading measured ``Material.properties``:

* ``core``      — ``magnetism`` (flux/permeability) and ``curie_temperature`` (the core must
  stay magnetic at the operating temperature, or it demagnetizes). M3 + M4.
* ``coil_wire`` — ``conductivity_continuous`` (→ coil resistance), ``melting_temperature`` and
  ``thermal_conductivity`` (the I²R burnout limit), and ``ductility`` (manufacturability — can
  it be drawn into wire). M5 + M6b + M8.
* ``shaft``     — ``strength`` (shear modulus; caps the torque it can transmit before it
  yields) and ``density`` (rotor inertia, reported but not in the torque equation). M8.

Performance equations (real-ish; consume only ``Material.properties``)::

    R_wire     = WIRE_R0 / sigma                 # sigma = conductivity_continuous; 0 -> open circuit
    I_ohm      = V / (R_wire + R_LOAD)           # coil resistance in series with the rest of the circuit
    P_burn     = BURNOUT_K * kappa * (T_melt - T_ambient)   # heat the wire can shed before melting
    I_max      = sqrt(P_burn / R_wire)           # current at which I^2 R_wire == P_burn
    I          = min(I_ohm, I_max)               # burnout-limited when I_ohm > I_max
    flux       = magnetism * max(0, 1 - T_ambient / T_curie)   # core demagnetizes near Tc; 0 if non-ferromagnetic
    turns      = clamp(ductility / DUCT_REF, 0, 1)             # brittle wire -> fewer usable turns (soft)
    torque     = TORQUE_K * flux * I * turns,  capped at SHAFT_K * strength   # shaft yields above the cap
    efficiency = R_LOAD / (R_wire + R_LOAD)      # fraction of input power NOT lost in the coil

The requirements are **emergent, never gated** (see :mod:`machines.roles`): a non-conducting
wire gives ``R_wire -> inf -> I = 0 -> torque = 0``; a non-magnetic core gives ``flux = 0``;
a core whose Curie point is below the operating temperature demagnetizes. A rare material that
is genuinely low-resistance, high-melt, ductile and a good heat-shedder (a great wire) plus a
high-magnetism, high-Curie core and a strong shaft visibly builds a better motor — and the
``limiting_factor`` tells you *why* it tops out. That payoff loop is the point of the layer.

The capital-letter constants are fixed display/calibration scales (the motor's "design
constants" — supply circuit, geometry), chosen so element-built motors land in legible
ranges. Like ``mechanical.MODULUS_SCALE``, they rescale every motor identically and change no
ordering; they are **not** tunable physics dials on any material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .roles import Blueprint, Requirement, Role, _props

# --- The motor blueprint: roles + what each one wants (suitability is a soft readout) ------
MOTOR = Blueprint(
    name="electric_motor",
    roles=(
        Role(
            name="core",
            description="magnetic core — carries the flux the torque rides on",
            requirements=(
                Requirement("magnetism", ref=0.60, description="permeability / flux"),
                Requirement("curie_temperature", ref=2.0,
                            description="stays magnetic at operating temperature"),
            ),
        ),
        Role(
            name="coil_wire",
            description="the winding — carries current with low loss, survives I^2R heating",
            requirements=(
                Requirement("conductivity_continuous", ref=0.12,
                            description="low resistance -> more current, less loss"),
                Requirement("melting_temperature", ref=2.5,
                            description="high burnout headroom"),
                Requirement("thermal_conductivity", ref=0.13,
                            description="sheds I^2R heat -> tolerates more current"),
                Requirement("ductility", ref=0.40,
                            description="manufacturability: can be drawn into wire"),
            ),
        ),
        Role(
            name="shaft",
            description="transmits the torque to the load without yielding",
            requirements=(
                Requirement("strength", ref=0.10, description="shear modulus -> torque it can carry"),
            ),
        ),
    ),
)

# --- Fixed design/calibration constants (NOT per-material dials; see module docstring) -----
WIRE_R0: float = 1.0      # coil-geometry resistance scale: R_wire = WIRE_R0 / sigma
R_LOAD: float = 5.0       # the rest of the circuit (the electromechanical load), in series
BURNOUT_K: float = 0.15   # heat-shedding scale: P_burn = BURNOUT_K * kappa * (T_melt - T_amb)
DUCT_REF: float = 0.40    # ductility at which the wire is fully manufacturable (turns -> 1)
TORQUE_K: float = 20.0    # torque constant: torque = TORQUE_K * flux * current * turns
SHAFT_K: float = 5.0      # shaft yield scale: max transmissible torque = SHAFT_K * strength
_EPS: float = 1e-12       # guards divide-by-zero for a perfect (zero-resistance) wire


@dataclass(frozen=True)
class OperatingPoint:
    """Where the motor is run — the machine-layer analogue of :class:`engine.conditions.Conditions`.

    ``voltage`` is the supply driving the coil; ``ambient_temperature`` is the surrounding
    temperature in the same reduced units as ``curie_temperature`` / ``melting_temperature``
    (so they are directly comparable). Performance is a *curve* over this point — raise the
    voltage and torque climbs until the wire hits its burnout limit, exactly as a real motor.
    """

    voltage: float = 1.0
    ambient_temperature: float = 0.5


@dataclass(frozen=True)
class MotorPerformance:
    """Computed motor stats plus the intermediate readings that explain them."""

    torque: float
    efficiency: float
    current: float            # operating current = min(ohmic, burnout)
    ohmic_current: float      # what Ohm's law alone would draw
    burnout_current: float    # the current at which the coil melts
    flux: float               # effective core flux (after demagnetization)
    wire_resistance: float
    turns_factor: float       # manufacturability multiplier from ductility
    shaft_torque_cap: float   # torque above which the shaft yields
    limiting_factor: str      # "ohmic" | "burnout" — what caps the current
    shaft_limited: bool       # torque is clipped by the shaft, not the electromagnetics
    demagnetized: bool        # operating temperature >= core Curie point -> no flux
    suitabilities: dict[str, float]  # soft 0..1 fit score per role (legibility only)

    def summary(self) -> str:
        lim = self.limiting_factor + (", shaft-limited" if self.shaft_limited else "")
        if self.demagnetized:
            lim += ", core DEMAGNETIZED"
        suit = "  ".join(f"{k}={v:.2f}" for k, v in self.suitabilities.items())
        return (
            f"torque={self.torque:.4f}  efficiency={self.efficiency:.3f}  "
            f"current={self.current:.4f} (ohmic={self.ohmic_current:.4f}, "
            f"burnout={self.burnout_current:.4f})\n"
            f"flux={self.flux:.4f}  R_wire={self.wire_resistance:.3f}  "
            f"turns={self.turns_factor:.3f}  shaft_cap={self.shaft_torque_cap:.4f}\n"
            f"limited by: {lim}\nsuitability: {suit}"
        )


def build_motor(
    core: Any,
    coil_wire: Any,
    shaft: Any,
    operating_point: OperatingPoint = OperatingPoint(),
) -> MotorPerformance:
    """Build a motor from three materials and compute its performance (spec §8).

    Each argument is a :class:`engine.material.Material` (duck-typed: has ``.properties``) or a
    bare properties dict. Pure function of the four inputs — deterministic, side-effect-free.
    Consumes ``Material.properties`` only; never touches the engine.
    """
    core_p = _props(core)
    wire_p = _props(coil_wire)
    shaft_p = _props(shaft)
    op = operating_point

    # --- Coil: resistance, then the two current limits (Ohm vs burnout) -------------------
    sigma = float(wire_p.get("conductivity_continuous", 0.0))
    wire_resistance = WIRE_R0 / sigma if sigma > _EPS else math.inf
    ohmic_current = op.voltage / (wire_resistance + R_LOAD)  # -> 0 as R_wire -> inf

    t_melt = float(wire_p.get("melting_temperature", 0.0))
    kappa = float(wire_p.get("thermal_conductivity", 0.0))
    headroom = max(0.0, t_melt - op.ambient_temperature)
    p_burn = BURNOUT_K * kappa * headroom
    if math.isfinite(wire_resistance) and wire_resistance > _EPS:
        burnout_current = math.sqrt(p_burn / wire_resistance)
    else:
        # No coil (open circuit): nothing flows, so nothing burns out either.
        burnout_current = 0.0 if not math.isfinite(wire_resistance) else math.inf

    current = min(ohmic_current, burnout_current)
    limiting_factor = "burnout" if burnout_current < ohmic_current else "ohmic"

    # --- Core: flux, with demagnetization as the operating temperature nears Curie --------
    magnetism = float(core_p.get("magnetism", 0.0))
    t_curie = float(core_p.get("curie_temperature", 0.0))
    if t_curie <= 0.0:
        flux, demagnetized = 0.0, True  # not ferromagnetic at standard conditions -> no flux
    else:
        retained = max(0.0, 1.0 - op.ambient_temperature / t_curie)
        flux, demagnetized = magnetism * retained, retained <= 0.0

    # --- Manufacturability: a brittle wire yields fewer usable turns (soft, never a gate) --
    ductility = float(wire_p.get("ductility", 0.0))
    turns_factor = max(0.0, min(1.0, ductility / DUCT_REF))

    # --- Torque, then the shaft yield cap -------------------------------------------------
    raw_torque = TORQUE_K * flux * current * turns_factor
    strength = float(shaft_p.get("strength", 0.0))
    shaft_torque_cap = SHAFT_K * strength
    shaft_limited = raw_torque > shaft_torque_cap
    torque = min(raw_torque, shaft_torque_cap)

    # --- Efficiency: fraction of supplied power not lost as heat in the coil --------------
    efficiency = R_LOAD / (wire_resistance + R_LOAD) if math.isfinite(wire_resistance) else 0.0

    return MotorPerformance(
        torque=torque,
        efficiency=efficiency,
        current=current,
        ohmic_current=ohmic_current,
        burnout_current=burnout_current,
        flux=flux,
        wire_resistance=wire_resistance,
        turns_factor=turns_factor,
        shaft_torque_cap=shaft_torque_cap,
        limiting_factor=limiting_factor,
        shaft_limited=shaft_limited,
        demagnetized=demagnetized,
        suitabilities=MOTOR.suitabilities(
            {"core": core_p, "coil_wire": wire_p, "shaft": shaft_p}
        ),
    )
