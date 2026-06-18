"""The rest of the machine layer (spec §8): heat sink, power cable, electromagnet, armor.

Companion to ``test_motor.py``. Same discipline — real-ish equations, a legible payoff loop, and
**emergent** requirements (never gates). Each machine rewards a different property family, so each
test pins the payoff that family creates:

* HEAT SINK — carbon (electrically dead, but stiff/light with high phononic kappa) has the best
  dissipation-per-mass: the **diamond divergence**, cashed in. A non-conducting fin overheats.
* POWER CABLE — a better conductor transmits more efficiently; distance erodes efficiency and
  ampacity; an insulator delivers nothing.
* ELECTROMAGNET — lift force is quadratic in current; a non-magnetic core or non-conducting coil
  produces zero lift (emergent, not gated).
* ARMOR — the M8 strength↔ductility dilemma is *solved by a composite*: a hard face needs a
  ductile backing, so combining opposite ends of the anti-correlation beats any single material.

Plus the shared current math (``machines._electrical``) and determinism (spec §6).
"""

import math

import pytest

from engine import elements
from engine.material import from_element
from machines import (
    ARMOR,
    CABLE,
    ELECTROMAGNET,
    HEAT_SINK,
    build_armor,
    build_cable,
    build_electromagnet,
    build_heat_sink,
)
from machines._electrical import coil_current
from machines.cable import TransmissionLoad
from machines.electromagnet import OperatingPoint
from machines.heatsink import ThermalLoad


@pytest.fixture(scope="module")
def mats():
    ids = ["iron", "copper", "silver", "tungsten", "carbon", "aluminium", "lead", "titanium"]
    return {eid: from_element(elements.ELEMENTS[eid]) for eid in ids}


# --- HEAT SINK: the diamond divergence pays off -------------------------------------------


def test_carbon_is_the_best_heat_sink_per_mass(mats):
    """Carbon has the highest dissipation-per-mass despite carrying no charge (diamond divergence)."""
    spec = {eid: build_heat_sink(mats[eid]).specific_dissipation
            for eid in ["carbon", "copper", "tungsten", "titanium", "iron"]}
    assert spec["carbon"] == max(spec.values())
    # and carbon is electrically dead — it would be a useless coil wire (the divergence)
    assert mats["carbon"].properties["conductivity_continuous"] == 0.0


def test_non_conducting_fin_overheats(mats):
    """A fin with zero thermal conductivity (lead here) has no conductance -> it overheats."""
    perf = build_heat_sink(mats["lead"])
    assert perf.conductance == 0.0
    assert math.isinf(perf.temperature_rise)
    assert perf.overheated is True


def test_heat_sink_overheats_above_its_max_load(mats):
    """Push more heat than max_heat_load and the fin reaches its melting point."""
    light = build_heat_sink(mats["tungsten"], ThermalLoad(heat_load=0.1))
    overloaded = build_heat_sink(mats["tungsten"], ThermalLoad(heat_load=light.max_heat_load * 2))
    assert light.overheated is False
    assert overloaded.overheated is True


# --- POWER CABLE: transmission loss, distance, ampacity -----------------------------------


def test_better_conductor_transmits_more_efficiently(mats):
    """Tungsten (higher sigma) loses less than copper over the same line."""
    assert (mats["tungsten"].properties["conductivity_continuous"]
            > mats["copper"].properties["conductivity_continuous"])
    assert build_cable(mats["tungsten"]).efficiency > build_cable(mats["copper"]).efficiency


def test_insulator_cable_delivers_nothing(mats):
    """Carbon (sigma = 0) -> infinite resistance -> zero transmission efficiency."""
    perf = build_cable(mats["carbon"])
    assert math.isinf(perf.resistance)
    assert perf.efficiency == 0.0
    assert perf.overheated is True


def test_distance_erodes_efficiency_and_ampacity(mats):
    """Longer line = more resistance = lower efficiency and a lower melting current."""
    short = build_cable(mats["copper"], TransmissionLoad(distance=1.0))
    long = build_cable(mats["copper"], TransmissionLoad(distance=4.0))
    assert long.efficiency < short.efficiency
    assert long.ampacity < short.ampacity


# --- ELECTROMAGNET: lift ~ I^2, magnetic + conducting both required ------------------------


def test_magnetic_core_lifts_nonmagnetic_does_not(mats):
    iron = build_electromagnet(mats["iron"], mats["copper"])
    copper_core = build_electromagnet(mats["copper"], mats["copper"])
    assert iron.lift_force > 0.0 and iron.demagnetized is False
    assert copper_core.lift_force == 0.0 and copper_core.demagnetized is True


def test_lift_is_the_square_of_the_field(mats):
    """The defining quadratic: lift force = field^2 (Maxwell stress ~ B^2)."""
    perf = build_electromagnet(mats["iron"], mats["copper"])
    assert perf.lift_force == pytest.approx(perf.field ** 2)


def test_non_conducting_coil_gives_no_field(mats):
    """A coil that can't carry current (lead) produces no field, even on a great core."""
    perf = build_electromagnet(mats["iron"], mats["lead"])
    assert perf.current == 0.0
    assert perf.field == 0.0
    assert perf.lift_force == 0.0


# --- ARMOR: the composite solves the M8 strength<->ductility dilemma ----------------------


def test_composite_needs_both_a_hard_face_and_a_ductile_backing(mats):
    """A hard face on a ductile backing beats the same face on a brittle one AND a soft face."""
    good = build_armor(mats["tungsten"], mats["aluminium"])     # strong face + ductile backing
    brittle_backing = build_armor(mats["tungsten"], mats["tungsten"])  # strong face, brittle backing
    soft_face = build_armor(mats["aluminium"], mats["aluminium"])      # no-strength face
    # the ductile backing nearly doubles protection vs a brittle one (support < 1)
    assert good.protection > brittle_backing.protection > 0.0
    assert brittle_backing.backing_support < 1.0
    # an all-ductile plate has no hard face -> it barely protects
    assert soft_face.protection == pytest.approx(0.0, abs=1e-9)
    # you cannot get good armor from one material: each best-in-class pick fails the other role
    assert good.protection > build_armor(mats["aluminium"], mats["tungsten"]).protection


def test_lighter_face_wins_protection_per_mass(mats):
    """Carbon face is far lighter than tungsten, so it wins specific protection on a fixed backing."""
    carbon = build_armor(mats["carbon"], mats["aluminium"])
    tungsten = build_armor(mats["tungsten"], mats["aluminium"])
    assert tungsten.protection > carbon.protection                 # tungsten stops more...
    assert carbon.specific_protection > tungsten.specific_protection  # ...but carbon is lighter


# --- shared current math (machines._electrical) -------------------------------------------


def test_coil_current_open_circuit_and_limits():
    insulator = coil_current({"conductivity_continuous": 0.0}, voltage=1.0,
                             ambient_temperature=0.5, r0=1.0, load_resistance=5.0, burnout_k=0.15)
    assert math.isinf(insulator.resistance)
    assert insulator.current == 0.0

    conductor = coil_current(
        {"conductivity_continuous": 0.1, "thermal_conductivity": 0.2, "melting_temperature": 3.0},
        voltage=1.0, ambient_temperature=0.5, r0=1.0, load_resistance=5.0, burnout_k=0.15,
    )
    assert conductor.current == pytest.approx(min(conductor.ohmic_current, conductor.burnout_current))
    assert conductor.limiting_factor in ("ohmic", "burnout")


# --- determinism (spec §6) ----------------------------------------------------------------


def test_machines_are_deterministic(mats):
    assert build_heat_sink(mats["carbon"]) == build_heat_sink(mats["carbon"])
    assert build_cable(mats["copper"]) == build_cable(mats["copper"])
    assert (build_electromagnet(mats["iron"], mats["copper"])
            == build_electromagnet(mats["iron"], mats["copper"]))
    assert build_armor(mats["tungsten"], mats["aluminium"]) == build_armor(mats["tungsten"], mats["aluminium"])


# --- framework: every blueprint exposes its roles consistently ----------------------------


def test_blueprints_have_expected_roles():
    assert HEAT_SINK.role_names == ("fin",)
    assert CABLE.role_names == ("conductor",)
    assert ELECTROMAGNET.role_names == ("core", "coil")
    assert ARMOR.role_names == ("hard_face", "ductile_backing")
