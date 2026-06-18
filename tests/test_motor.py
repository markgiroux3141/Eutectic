"""The machine layer (spec §8): the electric-motor worked example + the role framework.

This is an *engineering* milestone, not an emergent-physics one, so there is no textbook
value to recover. The discipline that still applies: the equations must be real-ish, the
payoff loop must be legible (a better material visibly builds a better motor), and the
"requirements" must be **emergent consequences**, never hardcoded gates. So the tests assert:

1. FRAMEWORK — :mod:`machines.roles` terms/suitability behave (geometric mean → 0 if any
   requirement is unmet; accepts a Material or a bare props dict; missing role is an error).
2. EMERGENT REQUIREMENTS — a non-conducting wire, a non-magnetic core, and an over-Curie
   operating point each *kill* the motor through the equations, with no gate in the code path.
3. PAYOFF LOOP — a better coil wire gives strictly more torque AND higher efficiency; a strong
   shaft lifts the torque a weak shaft clips.
4. OPERATING CURVE — torque is non-decreasing in supply voltage and flattens at the I²R
   burnout ceiling (the parameterized-operating-point design choice, doing its job).
5. THERMAL HEADROOM — a better heat-shedder (higher thermal conductivity) raises the burnout
   current (the M6b property genuinely matters to the motor).
6. DETERMINISM — same materials + operating point → byte-identical performance (spec §6).

Engine isolation (spec §2, §8): ``machines`` consumes ``Material.properties`` only. This file
is allowed to import both; the engine must never import ``machines`` (asserted below).
"""

import math

import pytest

from engine import elements
from engine.material import from_element
from machines.motor import (
    MOTOR,
    OperatingPoint,
    build_motor,
)
from machines.roles import Blueprint, Requirement, Role


# --- shared real materials (built once; from_element runs the gated ensemble sweeps) -------


@pytest.fixture(scope="module")
def mats():
    ids = ["iron", "copper", "silver", "tungsten", "lead", "nickel"]
    return {eid: from_element(elements.ELEMENTS[eid]) for eid in ids}


# --- 1. the role / requirement framework (machines.roles) ---------------------------------


def test_requirement_term_clamps_and_directions():
    higher = Requirement("x", ref=2.0, higher_is_better=True)
    assert higher.term({"x": 1.0}) == pytest.approx(0.5)
    assert higher.term({"x": 5.0}) == 1.0           # clamped to 1
    assert higher.term({"x": 0.0}) == 0.0
    assert higher.term({}) == 0.0                    # missing -> 0

    lower = Requirement("r", ref=2.0, higher_is_better=False)  # less is better
    assert lower.term({"r": 2.0}) == 1.0
    assert lower.term({"r": 4.0}) == pytest.approx(0.5)
    assert lower.term({"r": 0.0}) == 1.0             # perfect (zero resistance)


def test_suitability_is_geometric_mean_and_zero_if_any_term_zero():
    role = Role("r", (Requirement("a", ref=1.0), Requirement("b", ref=1.0)))
    assert role.suitability({"a": 1.0, "b": 1.0}) == pytest.approx(1.0)
    # geometric mean of 0.25 and 1.0 = 0.5
    assert role.suitability({"a": 0.25, "b": 1.0}) == pytest.approx(0.5)
    # any unmet requirement drags the whole score to 0 (expresses "all must hold")
    assert role.suitability({"a": 0.0, "b": 1.0}) == 0.0


def test_suitability_accepts_material_or_props(mats):
    core_role = MOTOR.role("core")
    iron = mats["iron"]
    assert core_role.suitability(iron) == core_role.suitability(iron.properties)


def test_blueprint_suitabilities_and_missing_role(mats):
    assignment = {"core": mats["iron"], "coil_wire": mats["copper"], "shaft": mats["tungsten"]}
    suit = MOTOR.suitabilities(assignment)
    assert set(suit) == set(MOTOR.role_names)
    assert suit["core"] == pytest.approx(1.0, abs=1e-9)  # iron: high magnetism + high Curie
    with pytest.raises(KeyError):
        MOTOR.suitabilities({"core": mats["iron"]})  # incomplete assembly


# --- 2. requirements are EMERGENT, not gated ----------------------------------------------


def test_non_conducting_wire_kills_the_motor(mats):
    """Lead's electrical sigma is 0 -> open circuit -> no current, no torque, no efficiency."""
    perf = build_motor(mats["iron"], mats["lead"], mats["tungsten"])
    assert mats["lead"].properties["conductivity_continuous"] == 0.0  # the cause
    assert math.isinf(perf.wire_resistance)
    assert perf.current == 0.0
    assert perf.torque == 0.0
    assert perf.efficiency == 0.0
    assert perf.suitabilities["coil_wire"] == 0.0   # the readout agrees, but didn't gate


def test_non_magnetic_core_produces_no_flux(mats):
    """Copper has no Curie point -> no flux -> no torque, even with a perfect wire/shaft."""
    perf = build_motor(mats["copper"], mats["copper"], mats["tungsten"])
    assert mats["copper"].properties["curie_temperature"] == 0.0
    assert perf.flux == 0.0
    assert perf.demagnetized is True
    assert perf.torque == 0.0
    # the wire still carries current — the kill is the core, isolated to flux
    assert perf.current > 0.0


def test_core_demagnetizes_above_its_curie_point(mats):
    """Operating above the core's Curie point collapses the flux (continuously, then fully)."""
    cool = build_motor(mats["nickel"], mats["copper"], mats["tungsten"],
                       OperatingPoint(ambient_temperature=0.5))
    tc = mats["nickel"].properties["curie_temperature"]
    hot = build_motor(mats["nickel"], mats["copper"], mats["tungsten"],
                      OperatingPoint(ambient_temperature=tc + 1.0))
    assert 0.0 < cool.flux                       # magnetic when cool
    assert hot.flux == 0.0 and hot.demagnetized  # gone above Tc
    assert hot.torque == 0.0


# --- 3. the payoff loop (a better material -> a better motor) ------------------------------


def test_better_wire_gives_more_torque_and_efficiency(mats):
    """Copper (higher sigma) beats silver as a coil wire on BOTH torque and efficiency."""
    assert (mats["copper"].properties["conductivity_continuous"]
            > mats["silver"].properties["conductivity_continuous"])  # the cause
    good = build_motor(mats["iron"], mats["copper"], mats["tungsten"])
    worse = build_motor(mats["iron"], mats["silver"], mats["tungsten"])
    assert good.torque > worse.torque
    assert good.efficiency > worse.efficiency


def test_strong_shaft_lifts_the_torque_a_weak_shaft_clips(mats):
    """A weak shaft yields and clips torque to its cap; a strong shaft lets the EM torque through."""
    weak = build_motor(mats["iron"], mats["copper"], mats["copper"])      # copper shaft: low strength
    strong = build_motor(mats["iron"], mats["copper"], mats["tungsten"])  # tungsten shaft: strong
    assert weak.shaft_limited is True
    assert weak.torque == pytest.approx(weak.shaft_torque_cap)
    assert strong.shaft_limited is False
    assert strong.torque > weak.torque


# --- 4. the operating curve (parameterized operating point) -------------------------------


def test_torque_rises_with_voltage_then_flattens_at_burnout(mats):
    """Torque is non-decreasing in supply voltage and saturates at the I^2R burnout ceiling."""
    def torque_at(v):
        return build_motor(mats["iron"], mats["copper"], mats["tungsten"],
                           OperatingPoint(voltage=v)).torque

    low, mid, high, higher = torque_at(0.3), torque_at(0.6), torque_at(2.0), torque_at(3.0)
    assert low < mid             # rising while ohmic-limited
    assert mid <= high           # non-decreasing
    assert high == pytest.approx(higher)  # flat: both clamped at the burnout current


# --- 5. the thermal property genuinely matters (M6b feeds the burnout limit) --------------


def test_higher_thermal_conductivity_raises_burnout_current():
    """Two identical wires but for thermal conductivity: the better heat-shedder tolerates more I.

    Uses bare property dicts (the framework is duck-typed) to isolate the one variable.
    """
    base = {
        "conductivity_continuous": 0.05,  # high resistance -> burnout-limited, so kappa bites
        "melting_temperature": 3.0,
        "ductility": 0.4,
    }
    cool_wire = {**base, "thermal_conductivity": 0.05}
    hot_wire = {**base, "thermal_conductivity": 0.50}
    core = {"magnetism": 0.7, "curie_temperature": 2.5}
    shaft = {"strength": 1.0}
    cool = build_motor(core, cool_wire, shaft)
    hot = build_motor(core, hot_wire, shaft)
    assert hot.burnout_current > cool.burnout_current
    assert hot.torque > cool.torque


# --- 6. determinism (spec §6) -------------------------------------------------------------


def test_motor_performance_is_deterministic(mats):
    op = OperatingPoint(voltage=1.3, ambient_temperature=0.7)
    a = build_motor(mats["iron"], mats["copper"], mats["tungsten"], op)
    b = build_motor(mats["iron"], mats["copper"], mats["tungsten"], op)
    assert a == b  # frozen dataclass equality over every field, including suitabilities


# --- engine isolation: the engine must never import the machine layer (spec §2, §8) -------


def test_engine_does_not_import_machines():
    import pathlib

    engine_dir = pathlib.Path(__file__).resolve().parent.parent / "engine"
    offenders = [
        p.name for p in engine_dir.rglob("*.py")
        if "machines" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"engine modules reference 'machines': {offenders}"
