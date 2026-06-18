"""Shared electrical math for the machine layer — coil/conductor current and its limits.

Three machines drive current through a conductor and hit the same two ceilings: Ohm's law
(``I = V / R``) and an **I²R burnout** limit (the conductor melts if it can't shed the heat).
Rather than repeat that in :mod:`machines.motor`, :mod:`machines.electromagnet` and
:mod:`machines.cable`, the physics lives here once. It consumes ``Material.properties``
(``conductivity_continuous``, ``thermal_conductivity``, ``melting_temperature``) only — like the
rest of the machine layer, it never touches the engine.

The current limits compose: a coil draws ``min(ohmic, burnout)`` and reports which bound binds.
``load_resistance`` is the rest of the series circuit (the motor's electromechanical load); set
it to ``0`` for a bare coil/conductor whose only ceiling is its own resistance + burnout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

EPS: float = 1e-12  # guards divide-by-zero for a perfect (zero-resistance) conductor


def wire_resistance(sigma: float, *, r0: float, length: float = 1.0) -> float:
    """Conductor resistance ``r0 * length / sigma``; ``inf`` for a non-conductor (sigma → 0)."""
    return r0 * length / sigma if sigma > EPS else math.inf


def burnout_current(
    *,
    resistance: float,
    thermal_conductivity: float,
    melting_temperature: float,
    ambient_temperature: float,
    burnout_k: float,
) -> float:
    """Current at which I²R heating equals the heat the conductor can shed before melting.

    ``P_burn = burnout_k * kappa * max(0, T_melt - T_ambient)`` is the dissipation the conductor
    tolerates (a better heat-shedder / higher-melting wire tolerates more); the burnout current
    is ``sqrt(P_burn / R)``. A non-conductor (``R = inf``) carries nothing, so nothing burns out
    → ``0.0``.
    """
    headroom = max(0.0, melting_temperature - ambient_temperature)
    p_burn = burnout_k * thermal_conductivity * headroom
    if math.isfinite(resistance) and resistance > EPS:
        return math.sqrt(p_burn / resistance)
    return 0.0 if not math.isfinite(resistance) else math.inf


@dataclass(frozen=True)
class CoilState:
    """The current a coil/conductor carries and the two limits that bound it."""

    current: float            # operating current = min(ohmic, burnout)
    resistance: float
    ohmic_current: float      # what Ohm's law alone would draw
    burnout_current: float    # the current at which the conductor melts
    limiting_factor: str      # "ohmic" | "burnout"


def coil_current(
    props: Mapping[str, float],
    *,
    voltage: float,
    ambient_temperature: float,
    r0: float,
    load_resistance: float,
    burnout_k: float,
    length: float = 1.0,
) -> CoilState:
    """Resolve a conductor's operating current from its measured properties and the drive.

    ``min`` of the Ohmic draw ``V / (R + load_resistance)`` and the burnout ceiling. A
    non-conducting material gives ``R = inf`` → zero current (open circuit), the emergent
    "requirement" that a coil must actually conduct.
    """
    sigma = float(props.get("conductivity_continuous", 0.0))
    resistance = wire_resistance(sigma, r0=r0, length=length)
    ohmic = voltage / (resistance + load_resistance) if math.isfinite(resistance) else 0.0
    burnout = burnout_current(
        resistance=resistance,
        thermal_conductivity=float(props.get("thermal_conductivity", 0.0)),
        melting_temperature=float(props.get("melting_temperature", 0.0)),
        ambient_temperature=ambient_temperature,
        burnout_k=burnout_k,
    )
    current = min(ohmic, burnout)
    limiting_factor = "burnout" if burnout < ohmic else "ohmic"
    return CoilState(current, resistance, ohmic, burnout, limiting_factor)
