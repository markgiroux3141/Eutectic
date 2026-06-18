"""Heat sink worked example (spec §8) — the thermal machine, where the diamond divergence pays off.

A heat sink is a one-role assembly: a ``fin`` that conducts a heat load away to ambient. Its
performance is *computed* from the fin material's measured properties:

* ``thermal_conductivity`` — sets the conductance (how fast it moves heat). M6b.
* ``melting_temperature`` — the fin fails if its own peak temperature reaches its melting point.
* ``density`` — **lower is better** (a heat sink wants to be light): the figure of merit is
  dissipation *per unit mass*.

Performance::

    conductance         = COND_K * thermal_conductivity
    temperature_rise    = heat_load / conductance
    peak_temperature    = ambient + temperature_rise
    overheated          = peak_temperature >= melting_temperature      # the fin melts
    max_heat_load       = conductance * max(0, melting_temperature - ambient)
    specific_dissipation = thermal_conductivity / density              # the headline: cooling per kg

This is where **carbon wins**: it is electrically dead (a useless coil wire — :mod:`machines.motor`),
but its M6b thermal conductivity is almost entirely *phononic* and its density is tiny, so it has
by far the best dissipation-per-mass in the element set — the diamond divergence, cashed in. A
dense metal like tungsten dissipates more heat in absolute terms but is far heavier. No gates:
a non-conducting fin (``thermal_conductivity = 0``) simply has zero conductance and overheats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .roles import Blueprint, Requirement, Role, _props

HEAT_SINK = Blueprint(
    name="heat_sink",
    roles=(
        Role(
            name="fin",
            description="conducts a heat load to ambient; light + conductive + high-melting wins",
            requirements=(
                Requirement("thermal_conductivity", ref=0.13, description="moves heat fast"),
                Requirement("melting_temperature", ref=2.5, description="survives its own peak temperature"),
                Requirement("density", ref=30.0, higher_is_better=False,
                            description="lighter is better (dissipation per mass)"),
            ),
        ),
    ),
)

# Fixed design/calibration scales (not per-material dials; cf. motor's constants).
COND_K: float = 1.0       # conductance = COND_K * thermal_conductivity


@dataclass(frozen=True)
class ThermalLoad:
    """The operating point: a heat load to shed at some ambient temperature (reduced units)."""

    heat_load: float = 0.10
    ambient_temperature: float = 0.5


@dataclass(frozen=True)
class HeatSinkPerformance:
    conductance: float
    temperature_rise: float
    peak_temperature: float
    overheated: bool                 # the fin reaches its own melting point
    max_heat_load: float             # the heat load at which it would melt
    specific_dissipation: float      # thermal_conductivity / density — the figure of merit
    suitability: float

    def summary(self) -> str:
        flag = "  OVERHEATED" if self.overheated else ""
        return (
            f"temperature_rise={self.temperature_rise:.4f}  peak={self.peak_temperature:.4f}{flag}\n"
            f"conductance={self.conductance:.4f}  max_heat_load={self.max_heat_load:.4f}\n"
            f"specific_dissipation (cooling/mass) = {self.specific_dissipation:.5f}\n"
            f"suitability: fin={self.suitability:.2f}"
        )


def build_heat_sink(fin: Any, load: ThermalLoad = ThermalLoad()) -> HeatSinkPerformance:
    """Build a heat sink from one material and compute its performance (spec §8).

    Pure function of ``(fin, load)``; consumes ``Material.properties`` only.
    """
    props = _props(fin)
    kappa = float(props.get("thermal_conductivity", 0.0))
    t_melt = float(props.get("melting_temperature", 0.0))
    density = float(props.get("density", 0.0))

    conductance = COND_K * kappa
    temperature_rise = load.heat_load / conductance if conductance > 0.0 else math.inf
    peak_temperature = load.ambient_temperature + temperature_rise
    overheated = peak_temperature >= t_melt
    max_heat_load = conductance * max(0.0, t_melt - load.ambient_temperature)
    specific_dissipation = kappa / density if density > 0.0 else 0.0

    return HeatSinkPerformance(
        conductance=conductance,
        temperature_rise=temperature_rise,
        peak_temperature=peak_temperature,
        overheated=overheated,
        max_heat_load=max_heat_load,
        specific_dissipation=specific_dissipation,
        suitability=HEAT_SINK.role("fin").suitability(props),
    )
