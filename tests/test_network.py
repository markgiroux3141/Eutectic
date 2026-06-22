"""C5 keystone: an emergent tech tree gated by GENUINE thresholds, not tuned dials (spec §12, §15).

The reachable set is the transitive closure of live reactions (ΔG<0 and fast enough) over an
inventory. We pressure-test that the gates are real:

* **Condition gate = a genuine ΔG sign-crossing.** Free radicals are unreachable cold and unlock
  as T crosses each diatomic's dissociation T*, *in emergent bond-strength order* (Cl<O<H<N) — the
  onset temperature equals the C3 crossover, with no tuned numbers. Reachability strictly grows.
* **Multi-step prerequisite.** A target reachable only via an intermediate that must be made first
  (atomic O → ``O+H₂→H₂O``); below the intermediate's gate the target is unreachable.
* **Catalyst gate (flagged SOFT).** Fe flips NH₃ from unreachable to reachable inside its
  thermodynamic window, without changing any ΔG.
* **Rarity is emergent.** ``NO`` sits in the graph but is reachable at *no* temperature — a closed
  thermodynamic window, not an authored "rare" tag.
* **The disguised-dial check (no-fudge norm).** The rate-gated unlock slides smoothly as the rate
  cutoff is nudged (a dial); the ΔG-gated onset is pinned at T* regardless of the cutoff (a gate).
* **Determinism (spec §14).** reachable twice → identical set; construction order is irrelevant.
"""

import pytest

from chemistry import network as net
from chemistry import reaction as rx
from chemistry.conditions import ChemConditions
from chemistry.network import DEFAULT_RATE_CUTOFF


# --- shared species / fixtures --------------------------------------------------------

H2, O2, N2, Cl2 = (rx.diatomic(s) for s in ("H", "O", "N", "Cl"))
H2O, HCl, NH3, NO = rx.binary("H", "O"), rx.binary("H", "Cl"), rx.binary("N", "H"), rx.binary("N", "O")
aH, aO, aN, aCl = (rx.atom(s) for s in ("H", "O", "N", "Cl"))

# The genuine-threshold reference: a diatomic's dissociation crossover T* (C3).
def _dissociation_Tstar(sym):
    diss = rx.reaction(((rx.diatomic(sym), 1),), ((rx.atom(sym), 2),))
    return diss.crossover_temperature(ChemConditions())


def _net():
    return net.demo_network()


def _inv():
    return net.demo_inventory()


# --- KEYSTONE 1: condition gate is a genuine ΔG sign-crossing --------------------------

def test_radicals_unreachable_cold_reachable_hot():
    """Cold, no free atoms exist; hot, dissociation has switched on and they do."""
    n, inv = _net(), _inv()
    cold = ChemConditions(temperature=1.0)
    hot = ChemConditions(temperature=10.0)
    for a in (aCl, aO, aH, aN):
        assert not n.can_reach(a, inv, cold)
        assert n.can_reach(a, inv, hot)


def test_radical_unlock_order_is_emergent_bond_strength():
    """Weaker bond dissociates cooler: the unlock temperatures order Cl<O<H<N, parameter-free."""
    n, inv = _net(), _inv()
    # require_rate=False isolates the pure-ΔG gate (the genuine threshold), no kinetic dial.
    t_cl = n.unlock_temperature(aCl, inv, require_rate=False)
    t_o = n.unlock_temperature(aO, inv, require_rate=False)
    t_h = n.unlock_temperature(aH, inv, require_rate=False)
    t_n = n.unlock_temperature(aN, inv, require_rate=False)
    assert t_cl < t_o < t_h < t_n


def test_radical_onset_equals_the_dG_crossover_temperature():
    """The unlock temperature IS the C3 dissociation T* — the gate is the sign-crossing itself."""
    n, inv = _net(), _inv()
    for sym, atom in (("Cl", aCl), ("O", aO), ("H", aH), ("N", aN)):
        onset = n.unlock_temperature(atom, inv, require_rate=False)
        assert onset == pytest.approx(_dissociation_Tstar(sym), abs=0.01)


def test_reachability_grows_monotonically_as_temperature_rises():
    """Raising T past each genuine gate only ever ADDS radicals to the reachable set (a tech tree
    unlocking, not a relabelled dial). Checked on the radical sublattice, which is pure-ΔG-gated."""
    n, inv = _net(), _inv()
    radicals = {net.species_key(a) for a in (aCl, aO, aH, aN)}
    seen = set()
    for t in (1.0, 4.5, 5.0, 7.5, 8.0, 12.0):
        got = n.reachable(inv, ChemConditions(temperature=t), require_rate=False) & radicals
        assert seen <= got  # never loses a radical it had unlocked
        seen = got
    assert seen == radicals  # all four eventually unlocked


# --- KEYSTONE 2: an emergent multi-step prerequisite ----------------------------------

def _radical_route_only():
    """A sub-network with ONLY the radical path to H₂O (no direct 2H₂+O₂→2H₂O), to isolate the
    prerequisite: H₂O via this route is impossible without first liberating atomic O."""
    return net.reaction_network([
        rx.reaction(((O2, 1),), ((aO, 2),)),          # O₂ → 2O  (the gated prerequisite)
        rx.reaction(((aO, 1), (H2, 1)), ((H2O, 1),)),  # O + H₂ → H₂O  (needs atomic O)
    ])


def test_target_unreachable_without_first_making_the_intermediate():
    """H₂O via the radical route is unreachable below atomic O's gate, reachable in the overlap
    window — and only after atomic O is produced (a genuine two-hop prerequisite chain)."""
    sub = _radical_route_only()
    inv = [O2, H2]
    below = ChemConditions(temperature=3.0)   # below O₂ dissociation T*≈4.62
    window = ChemConditions(temperature=5.0)   # in the overlap window [4.62, 6.09]
    assert not sub.can_reach(aO, inv, below) and not sub.can_reach(H2O, inv, below)
    assert sub.can_reach(aO, inv, window) and sub.can_reach(H2O, inv, window)


def test_intermediate_is_produced_before_the_target_in_the_chain():
    """Provenance: atomic O is made by dissociation; H₂O is then made by the edge that consumes O."""
    sub = _radical_route_only()
    prod = sub.first_producers([O2, H2], ChemConditions(temperature=5.0))
    o_maker = prod[net.species_key(aO)]
    h2o_maker = prod[net.species_key(H2O)]
    assert o_maker is not None and net.species_key(aO) in o_maker.product_keys()
    # the reaction that makes H₂O consumes the atomic-O intermediate (the prerequisite edge)
    assert net.species_key(aO) in h2o_maker.reactant_keys()


# --- KEYSTONE 3: catalyst gate (flagged soft — operates through the rate dial) ---------

def test_catalyst_flips_reachability_without_touching_dG():
    """In its thermodynamic window NH₃ is unreachable cold but reachable once Fe lowers the
    barrier — the catalyst changes the *path's rate*, never the destination's ΔG."""
    n, inv = _net(), _inv()
    plain = ChemConditions(temperature=1.8)
    with_fe = ChemConditions(temperature=1.8, catalysts=frozenset({"Fe"}))
    assert not n.can_reach(NH3, inv, plain)
    assert n.can_reach(NH3, inv, with_fe)
    # ΔG of the ammonia synthesis is identical with/without the catalyst.
    haber = next(e.reaction for e in n.reactions
                 if net.species_key(NH3) in e.product_keys() and e.reactant_keys()[0] == net.species_key(N2))
    assert haber.delta_G(plain) == haber.delta_G(with_fe)


def test_ammonia_needs_the_catalyst_across_its_whole_window():
    """Without Fe, NH₃ is unreachable at every temperature (thermo wants cold, kinetics wants hot —
    the windows never meet); the catalyst is what bridges them."""
    n, inv = _net(), _inv()
    assert all(not n.can_reach(NH3, inv, ChemConditions(temperature=t))
               for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0))


# --- KEYSTONE 4: rarity is emergent (a locked target, no authored tag) ----------------

def test_NO_is_locked_out_at_every_temperature():
    """NO is in the graph but reachable at NO temperature: the direct route is endothermic with
    Δn_gas=0 (ΔG>0 always) and the radical route's window is disjoint from when atomic N exists."""
    n, inv = _net(), _inv()
    assert net.species_key(NO) in {net.species_key(s) for s in n.species()}  # it IS in the graph
    assert all(not n.can_reach(NO, inv, ChemConditions(temperature=t), require_rate=False)
               for t in [0.5 * k for k in range(1, 80)])  # T from 0.5 to ~40


# --- the no-fudge check: which gate is a threshold and which is a dial -----------------

def test_dG_gate_is_pinned_but_rate_gate_slides_with_the_cutoff():
    """The disguised-dial test (no-fudge norm). Nudge the rate cutoff over orders of magnitude:

    * the ΔG-gated radical onset (atomic Cl) does NOT move — it is a genuine sign-crossing;
    * the rate-gated unlock (HCl, spontaneous at all T so gated purely on rate) slides smoothly —
      the tell-tale that 'fast enough' is a soft dial, never dressed up as a tech-tree threshold.
    """
    n, inv = _net(), _inv()
    cutoffs = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    cl_onsets = [n.unlock_temperature(aCl, inv, rate_cutoff=rc) for rc in cutoffs]
    hcl_onsets = [n.unlock_temperature(HCl, inv, rate_cutoff=rc) for rc in cutoffs]
    # ΔG gate: pinned (all equal to within quantization) regardless of the cutoff.
    assert max(cl_onsets) - min(cl_onsets) < 1e-2
    # rate gate: strictly decreasing as the cutoff relaxes — it slides, continuously.
    assert all(b < a for a, b in zip(hcl_onsets, hcl_onsets[1:]))


def test_dissociation_onset_is_dG_limited_not_rate_limited():
    """At its T*, a dissociation's rate is enormous, so the rate gate is irrelevant there: the
    onset is identical with and without the kinetic gate (the ΔG sign-crossing alone sets it)."""
    n, inv = _net(), _inv()
    with_rate = n.unlock_temperature(aCl, inv, require_rate=True)
    dg_only = n.unlock_temperature(aCl, inv, require_rate=False)
    assert with_rate == pytest.approx(dg_only, abs=1e-3)


# --- determinism (spec §14) -----------------------------------------------------------

def test_reachable_twice_is_identical():
    n, inv = _net(), _inv()
    c = ChemConditions(temperature=6.0, pressure=2.0, catalysts=frozenset({"Fe"}))
    assert n.reachable(inv, c) == n.reachable(inv, c)


def test_reachable_independent_of_construction_order():
    """Same reactions in a different order ⇒ identical canonical ordering and identical reachset."""
    base = _net()
    shuffled = net.ReactionNetwork(reactions=tuple(reversed(base.reactions)))
    assert [e.reaction.canonical_id() for e in shuffled.reactions] == \
           [e.reaction.canonical_id() for e in base.reactions]
    c = ChemConditions(temperature=8.0)
    assert shuffled.reachable(_inv(), c) == base.reachable(_inv(), c)


def test_reachable_is_a_frozenset_of_keys():
    n, inv = _net(), _inv()
    got = n.reachable(inv, ChemConditions(temperature=5.0))
    assert isinstance(got, frozenset)
    assert net.species_key(O2) in got  # inventory is always reachable
    assert all(isinstance(k, tuple) and len(k) == 2 for k in got)


def test_default_rate_cutoff_is_a_documented_dial():
    """A guard that the soft dial stays where the demos expect it (so the catalyst gate holds)."""
    assert DEFAULT_RATE_CUTOFF == 1.0e-9
