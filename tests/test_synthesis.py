"""Spec §13 keystone: a synthesis ROUTE (trajectory) reaches what no static condition can.

C5 reachability is evaluated at one fixed condition. A real synthesis is a *route* — heat to make
an intermediate, quench to capture a product whose formation wants a different temperature. The
keystone: a target reachable at **no** static T falls out of a heat-then-cool trajectory, the gates
are genuine ΔG sign-crossings (not dials), and **order is load-bearing** (cool-then-heat fails).

Anchored on NO: atomic N exists only above T*≈7.8, but N+O→NO is exergonic only below T*≈5.4 — a
disjoint pair of windows, so NO is impossible at any fixed T (the C5 locked-out finding), yet a
heat(8)→quench(1) route makes it (real radical chemistry: NO forms hot, freezes in on cooling).
"""

import pytest

from chemistry import network as net
from chemistry import reaction as rx
from chemistry import synthesis as syn
from chemistry.conditions import ChemConditions


H2, O2, N2, Cl2 = (rx.diatomic(s) for s in ("H", "O", "N", "Cl"))
NO = rx.binary("N", "O")
NH3 = rx.binary("N", "H")


def _net():
    return net.demo_network()


def _inv():
    return net.demo_inventory()


# --- the keystone: a trajectory reaches what no static condition can ------------------

def test_NO_unreachable_at_every_static_temperature():
    """The C5 locked-out target: NO forms at no single fixed T (re-confirmed via one-stage routes)."""
    n, inv = _net(), _inv()
    assert all(not syn.synthesize(n, inv, syn.isothermal(t)).made(NO)
               for t in [0.5 * k for k in range(1, 80)])  # T from 0.5 to ~40


def test_heat_then_quench_synthesises_the_otherwise_impossible_target():
    """Heat to liberate radicals, quench to capture NO — the trajectory beats every static point."""
    n, inv = _net(), _inv()
    result = syn.synthesize(n, inv, syn.heat_quench(8.0, 1.0))
    assert result.made(NO)
    # NO appears at the COLD stage (index 1): it is the quench that makes N+O→NO exergonic.
    assert result.stage_made(NO) == 1


def test_order_is_load_bearing_cool_then_heat_fails():
    """Same two temperatures, reversed: cool-then-heat ends hot (NO endergonic) → no NO. Path
    dependence is real, not an artefact of which conditions were visited."""
    n, inv = _net(), _inv()
    forward = syn.Route((ChemConditions(temperature=8.0), ChemConditions(temperature=1.0)))
    reverse = syn.Route((ChemConditions(temperature=1.0), ChemConditions(temperature=8.0)))
    assert syn.synthesize(n, inv, forward).made(NO)
    assert not syn.synthesize(n, inv, reverse).made(NO)


def test_neither_endpoint_alone_makes_it():
    """The trajectory's power is in the sequence: neither the hot nor the cold set-point alone works."""
    n, inv = _net(), _inv()
    assert not syn.synthesize(n, inv, syn.isothermal(8.0)).made(NO)
    assert not syn.synthesize(n, inv, syn.isothermal(1.0)).made(NO)
    assert syn.synthesize(n, inv, syn.heat_quench(8.0, 1.0)).made(NO)


def test_captured_product_persists_through_a_reheat():
    """Cumulative attainability: once NO is captured at the cold stage, a later reheat doesn't lose
    it (you isolated it) — heat,cool,reheat still counts NO as synthesised."""
    n, inv = _net(), _inv()
    route = syn.Route((ChemConditions(temperature=8.0), ChemConditions(temperature=1.0),
                       ChemConditions(temperature=8.0)))
    result = syn.synthesize(n, inv, route)
    assert result.made(NO) and result.stage_made(NO) == 1


# --- the gate is genuine: the trajectory window tracks the ΔG sign-crossings ----------

def test_quench_must_land_below_the_recombination_crossover():
    """N+O→NO has a genuine T*≈5.4; a quench to just below it captures NO, one to just above
    does not — the trajectory gate is the ΔG sign-crossing, not a tuned cutoff."""
    n, inv = _net(), _inv()
    radical = rx.reaction(((rx.atom("N"), 1), (rx.atom("O"), 1)), ((NO, 1),))
    t_star = radical.crossover_temperature(ChemConditions())
    assert syn.synthesize(n, inv, syn.heat_quench(8.0, t_star - 0.3)).made(NO)
    assert not syn.synthesize(n, inv, syn.heat_quench(8.0, t_star + 0.3)).made(NO)


def test_heat_must_reach_the_dissociation_threshold():
    """If the hot stage never crosses atomic N's dissociation T*≈7.8, no atomic N is liberated and
    the quench has nothing to recombine — NO stays unreachable."""
    n, inv = _net(), _inv()
    assert not syn.synthesize(n, inv, syn.heat_quench(7.0, 1.0)).made(NO)  # below N₂ T*
    assert syn.synthesize(n, inv, syn.heat_quench(8.0, 1.0)).made(NO)       # above it


# --- strict generalisation of C5 + determinism (spec §14) -----------------------------

def test_isothermal_route_reproduces_static_reachability():
    """A one-stage route is exactly a C5 reachable — the trajectory layer is a strict superset."""
    n, inv = _net(), _inv()
    for t in (1.0, 5.0, 8.0):
        route_set = syn.synthesize(n, inv, syn.isothermal(t)).reachable
        static_set = n.reachable(inv, ChemConditions(temperature=t))
        assert route_set == static_set


def test_synthesis_is_deterministic():
    n, inv = _net(), _inv()
    route = syn.heat_quench(8.0, 1.0)
    a = syn.synthesize(n, inv, route)
    b = syn.synthesize(n, inv, route)
    assert a.reachable == b.reachable
    assert a.first_stage == b.first_stage


def test_route_signature_is_stable_and_order_sensitive():
    """Signature keys a route reproducibly (determinism §14) and distinguishes orderings."""
    fwd = syn.Route((ChemConditions(temperature=8.0), ChemConditions(temperature=1.0)))
    fwd2 = syn.Route((ChemConditions(temperature=8.0), ChemConditions(temperature=1.0)))
    rev = syn.Route((ChemConditions(temperature=1.0), ChemConditions(temperature=8.0)))
    assert fwd.signature() == fwd2.signature()
    assert fwd.signature() != rev.signature()


def test_empty_route_is_rejected():
    with pytest.raises(ValueError):
        syn.Route(())
