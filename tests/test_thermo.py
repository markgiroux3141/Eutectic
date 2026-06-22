"""C3 keystone: reaction feasibility is *measured* from energetics (spec §10, §15).

The headline keystone: known exergonic reactions proceed (ΔG<0), endergonic ones don't —
**until a temperature threshold flips ΔG's sign** for an entropy-favored reaction, and that
flip is a genuine sign-crossing (not a smooth dial). Le Chatelier shifts reproduce under P/c.

Per the no-fudge norm, the keystone is anchored on the class where the sign is **robust to the
coarse inputs**: dissociation/recombination (only one bond type, so ΔH's sign can't hinge on
cross-bond calibration; Δn_gas=±1, so ΔS's sign can't hinge on the entropy scale). The
reactions whose ΔH sign the C1 bond model gets *wrong* (it is linear in bond order, overstating
double/triple bonds) are pinned separately as a recorded divergence, not hidden.
"""

import math

import pytest

from chemistry import reaction as rx
from chemistry.conditions import ChemConditions, Phase
from chemistry.reaction import GAS_CONSTANT


# --- fixtures: the robust dissociation/recombination class ----------------------------

def _dissociation(sym):
    """A₂(g) → 2A(g): break one bond, form none."""
    return rx.reaction(((rx.diatomic(sym), 1),), ((rx.atom(sym), 2),))


def _recombination(sym):
    """2A(g) → A₂(g): the reverse — form one bond."""
    return rx.reaction(((rx.atom(sym), 2),), ((rx.diatomic(sym), 1),))


# --- ΔH by Hess's law (robust class: sign independent of cross-bond calibration) ------

@pytest.mark.parametrize("sym", ["H", "O", "N", "Cl"])
def test_dissociation_is_endothermic(sym):
    """Breaking a bond and forming none costs energy: ΔH > 0, always."""
    assert _dissociation(sym).delta_H > 0.0


@pytest.mark.parametrize("sym", ["H", "O", "N", "Cl"])
def test_recombination_is_exothermic(sym):
    """Forming a bond releases energy: ΔH < 0 — and exactly negates dissociation."""
    diss, rec = _dissociation(sym), _recombination(sym)
    assert rec.delta_H < 0.0
    assert rec.delta_H == pytest.approx(-diss.delta_H)


def test_dH_is_hess_law_over_bond_energies():
    """ΔH equals Σ(broken) − Σ(formed) of the species' bond energies, nothing re-fit."""
    H2, O2, H2O = rx.diatomic("H"), rx.diatomic("O"), rx.binary("H", "O")
    r = rx.reaction(((H2, 2), (O2, 1)), ((H2O, 2),))
    broken = 2 * H2.bond_energy + 1 * O2.bond_energy
    formed = 2 * H2O.bond_energy
    assert r.delta_H == pytest.approx(broken - formed)


# --- ΔS from gas-mole change (the coarse, directional estimate) -----------------------

@pytest.mark.parametrize("sym", ["H", "O", "N", "Cl"])
def test_dissociation_raises_entropy(sym):
    """1 gas mole → 2 gas moles: ΔS > 0, Δn_gas = +1 (robust to the entropy scale)."""
    d = _dissociation(sym)
    assert d.delta_n_gas == 1
    assert d.delta_S > 0.0


def test_phase_sets_entropy_contribution():
    """A species' entropy is set by phase: condensing a product (gas→liquid) lowers ΔS."""
    H2, O2 = rx.diatomic("H"), rx.diatomic("O")
    gas = rx.reaction(((H2, 2), (O2, 1)), ((rx.binary("H", "O", Phase.GAS), 2),))
    liq = rx.reaction(((H2, 2), (O2, 1)), ((rx.binary("H", "O", Phase.LIQUID), 2),))
    assert liq.delta_S < gas.delta_S


# --- THE KEYSTONE: a genuine ΔG sign-crossing at a finite positive T ------------------

@pytest.mark.parametrize("sym", ["H", "O", "Cl"])
def test_dissociation_has_a_real_crossover(sym):
    """Entropy-favored endothermic reaction: ΔG flips sign at T* = ΔH/ΔS > 0."""
    d = _dissociation(sym)
    t_star = d.crossover_temperature()
    assert t_star is not None and t_star > 0.0
    assert t_star == pytest.approx(d.delta_H / d.delta_S)


@pytest.mark.parametrize("sym", ["H", "O", "Cl"])
def test_crossing_is_a_sign_flip_not_a_dial(sym):
    """ΔG is monotone in T and crosses zero exactly once; the boolean flips *at* T*."""
    d = _dissociation(sym)
    t_star = d.crossover_temperature()
    # below T*: ΔG>0 and not spontaneous; above T*: ΔG<0 and spontaneous (a hard threshold).
    below = ChemConditions(temperature=t_star * 0.5)
    above = ChemConditions(temperature=t_star * 1.5)
    assert d.delta_G(below) > 0.0 and not d.is_spontaneous(below)
    assert d.delta_G(above) < 0.0 and d.is_spontaneous(above)
    # monotone single crossing: ΔG strictly decreasing in T (ΔS>0), so exactly one root.
    temps = [t_star * f for f in (0.2, 0.6, 1.0, 1.4, 1.8)]
    gs = [d.delta_G(ChemConditions(temperature=t)) for t in temps]
    assert all(b < a for a, b in zip(gs, gs[1:]))  # strictly decreasing


def test_crossover_ordering_tracks_bond_strength():
    """Weaker bond ⇒ lower dissociation threshold: Cl₂ < H₂ < O₂ (emergent, parameter-free)."""
    t_cl = _dissociation("Cl").crossover_temperature()
    t_h = _dissociation("H").crossover_temperature()
    t_o = _dissociation("O").crossover_temperature()
    assert t_cl < t_h < t_o


# --- exergonic proceeds, endergonic doesn't (at standard conditions) ------------------

@pytest.mark.parametrize("sym", ["H", "O", "Cl"])
def test_recombination_proceeds_at_standard_conditions(sym):
    """The exergonic direction is spontaneous at standard T (ΔG<0, K>1)."""
    rec = _recombination(sym)
    assert rec.is_spontaneous()
    assert rec.equilibrium_constant() > 1.0


@pytest.mark.parametrize("sym", ["H", "O", "Cl"])
def test_dissociation_does_not_proceed_at_standard_conditions(sym):
    """The endergonic direction does not proceed cold (ΔG>0, K<1) — until heated past T*."""
    diss = _dissociation(sym)
    assert not diss.is_spontaneous()
    assert diss.equilibrium_constant() < 1.0


def test_K_equals_one_at_the_feasibility_boundary():
    """K = exp(−ΔG/RT) = 1 exactly when ΔG = 0 (at T*) — the clean equilibrium anchor."""
    d = _dissociation("O")
    at_star = ChemConditions(temperature=d.crossover_temperature())
    assert d.delta_G(at_star) == pytest.approx(0.0, abs=1e-3)
    assert d.equilibrium_constant(at_star) == pytest.approx(1.0, abs=1e-3)


# --- Le Chatelier: P and concentration shift the equilibrium qualitatively ------------

def test_pressure_suppresses_a_gas_producing_reaction():
    """O₂→2O makes more gas (Δn_gas=+1): raising P raises T* (harder to dissociate)."""
    d = _dissociation("O")
    t_lo = d.crossover_temperature(ChemConditions(pressure=0.1))
    t_mid = d.crossover_temperature(ChemConditions(pressure=1.0))
    t_hi = d.crossover_temperature(ChemConditions(pressure=10.0))
    assert t_lo < t_mid < t_hi


def test_pressure_at_fixed_T_pushes_dissociation_back():
    """At a T just above T*, compressing the gas turns a spontaneous dissociation off (Q rises)."""
    d = _dissociation("O")
    t = d.crossover_temperature() * 1.1
    assert d.is_spontaneous(ChemConditions(temperature=t, pressure=1.0))
    assert not d.is_spontaneous(ChemConditions(temperature=t, pressure=100.0))


def test_pressure_has_no_effect_when_gas_moles_unchanged():
    """A reaction with Δn_gas=0 is pressure-insensitive (no Le Chatelier lever)."""
    # 2A → A₂ has Δn_gas=−1; build a mole-conserving swap instead: A₂ + B₂ → 2 AB is Δn=0.
    HCl = rx.binary("H", "Cl")
    r = rx.reaction(((rx.diatomic("H"), 1), (rx.diatomic("Cl"), 1)), ((HCl, 2),))
    assert r.delta_n_gas == 0
    g1 = r.delta_G(ChemConditions(pressure=1.0))
    g2 = r.delta_G(ChemConditions(pressure=50.0))
    assert g1 == pytest.approx(g2)


def test_concentration_shifts_a_mole_changing_reaction():
    """Δn_total≠0 reactions respond to concentration (mass action): higher c raises T*."""
    d = _dissociation("Cl")
    t_lo = d.crossover_temperature(ChemConditions(concentration=0.1))
    t_hi = d.crossover_temperature(ChemConditions(concentration=10.0))
    assert t_lo < t_hi


# --- the REPORTED divergence: signs the C1 bond model gets wrong (pinned, not hidden) -

@pytest.mark.parametrize("name,react,prod", [
    # these are exothermic in reality, but the C1 linear-in-order energy overstates the
    # O=O / N≡N / (the bond broken), so Hess's law reads them ENDOthermic. Recorded.
    ("2H2+O2->2H2O", (("H", "H", 2), ("O", "O", 1)), (("H", "O", 2),)),
    ("N2+3H2->2NH3", (("N", "N", 1), ("H", "H", 3)), (("H", "N", 2),)),
    ("H2+Cl2->2HCl", (("H", "H", 1), ("Cl", "Cl", 1)), (("H", "Cl", 2),)),
])
def test_known_exothermic_reactions_read_endothermic_in_model(name, react, prod):
    """DOCUMENTED LIMITATION (spec §20): bond-breaking of multiple bonds flips the ΔH sign."""
    r = rx.reaction(
        tuple((rx.binary(a, b), c) for a, b, c in react),
        tuple((rx.binary(a, b), c) for a, b, c in prod),
    )
    assert r.delta_H > 0.0  # model says endothermic — the recorded divergence from reality


@pytest.mark.parametrize("name,react,prod", [
    # no multiple bond is *broken* here, so the overcount can't flip the sign: model agrees.
    ("C+O2->CO2", ((rx.atom, "C", 1), (rx.diatomic, "O", 1)), ((rx.binary, ("C", "O"), 1),)),
])
def test_reactions_that_only_form_multiple_bonds_are_correctly_exothermic(name, react, prod):
    """Where the sign is robust, the model gets it right (C+O₂→CO₂ exothermic)."""
    def build(side):
        out = []
        for fn, arg, c in side:
            sp = fn(*arg) if isinstance(arg, tuple) else fn(arg)
            out.append((sp, c))
        return tuple(out)

    r = rx.reaction(build(react), build(prod))
    assert r.delta_H < 0.0


# --- determinism + identity (spec §14) ------------------------------------------------

def test_reaction_is_deterministic_and_order_independent():
    H2, O2, H2O = rx.diatomic("H"), rx.diatomic("O"), rx.binary("H", "O")
    r1 = rx.reaction(((H2, 2), (O2, 1)), ((H2O, 2),))
    r2 = rx.reaction(((O2, 1), (H2, 2)), ((H2O, 2),))  # reactants listed in swapped order
    assert r1.canonical_id() == r2.canonical_id()
    assert r1.delta_H == r2.delta_H and r1.delta_S == r2.delta_S


def test_chem_conditions_extends_and_validates():
    c = ChemConditions(temperature=2.0, pressure=3.0, concentration=0.5, catalysts=frozenset({"Pt"}))
    assert c.temperature == 2.0 and c.pressure == 3.0
    assert c.has_catalyst("Pt") and not c.has_catalyst("Fe")
    with pytest.raises(ValueError):
        ChemConditions(concentration=0.0)
    with pytest.raises(ValueError):
        ChemConditions(temperature=0.0)


def test_chem_conditions_seed_key_is_deterministic_and_distinguishes_dials():
    a = ChemConditions(temperature=2.0, concentration=1.0)
    b = ChemConditions(temperature=2.0, concentration=1.0)
    c = ChemConditions(temperature=2.0, concentration=2.0)
    assert a.seed_key() == b.seed_key()
    assert a.seed_key() != c.seed_key()
