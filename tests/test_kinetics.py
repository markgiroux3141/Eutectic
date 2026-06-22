"""C4 keystone: rate gates what actually happens — favourable-but-trapped (spec §11, §15).

A reaction can be exergonic (ΔG<0, C3) yet not proceed because its activation barrier is high:
it is *kinetically trapped* until ignited (raise T) or catalysed (lower Ea). The catalyst moves
the temperature needed for a given rate **without changing ΔG**. We also pressure-test the honest
limits: the rate is a smooth exponential (no true sharp transition), and a barrierless reaction
(radical recombination) is not trapped at all.

Anchored on C+O₂→CO₂ and C+2H₂→CH₄ — reactions our model agrees are exergonic. The textbook
favourable-but-trapped example (H₂+O₂) cannot serve here: its ΔG sign is wrong in our model (the
C3 linear-bond-order finding), so it is not 'favourable' to begin with.
"""

import pytest

from chemistry import kinetics as kin
from chemistry import reaction as rx
from chemistry.conditions import ChemConditions
from chemistry.kinetics import ARRHENIUS_A, CATALYST_BARRIER_FACTOR


def _combustion():
    """C(g) + O₂ → CO₂: exergonic in our model, with an O=O barrier to break (trapped)."""
    return rx.reaction(((rx.atom("C"), 1), (rx.diatomic("O"), 1)), ((rx.binary("C", "O"), 1),))


def _recombination():
    """2H → H₂: exergonic and barrierless (no reactant bonds to break)."""
    return rx.reaction(((rx.atom("H"), 2),), ((rx.diatomic("H"), 1),))


# --- activation energy from the transition-state estimate -----------------------------

def test_barrierless_when_no_reactant_bonds_break():
    """Radical recombination has Ea≈0 — physically right, and so it is never trapped."""
    k = kin.kinetics(_recombination())
    assert k.base_activation_energy() == pytest.approx(0.0)


def test_activation_energy_scales_with_reactant_bonds():
    """A reaction that must break a strong bond has a higher barrier."""
    k = kin.kinetics(_combustion())
    assert k.base_activation_energy() > 0.0
    # Ea is exactly the fixed fraction of the reactant bond energy (nothing re-fit).
    from chemistry.kinetics import BARRIER_FRACTION
    assert k.base_activation_energy() == pytest.approx(BARRIER_FRACTION * k.reactant_bond_energy)


# --- THE KEYSTONE: favourable but kinetically trapped, freed by heat or catalyst ------

def test_favourable_but_trapped_then_ignited_by_heat():
    """Exergonic cold AND hot (within its favourable window), yet negligibly slow cold.

    NB this reaction reduces gas moles (Δn_gas=−1), so above the C3 crossover (~5.4) it turns
    endergonic — correct entropy behaviour. 'Ignition' here means heating into the still-
    favourable regime where the rate has climbed by orders of magnitude.
    """
    r = _combustion()
    k = kin.kinetics(r)
    cold = ChemConditions(temperature=1.0)
    hot = ChemConditions(temperature=4.0)
    assert r.is_spontaneous(cold) and r.is_spontaneous(hot)   # favourable at both
    assert k.rate(cold) < 1e-4                                 # trapped: effectively does not go
    assert k.rate(hot) > 1.0                                   # ignited: now it runs
    assert k.rate(hot) / k.rate(cold) > 1e6                    # orders-of-magnitude jump


def test_catalyst_speeds_the_rate_without_touching_dG():
    """A catalyst lowers Ea → raises rate at fixed T; ΔG is identical with/without it."""
    r = _combustion()
    k = kin.kinetics(r, catalysts={"Pt"})
    uncat = ChemConditions(temperature=2.0)
    cat = ChemConditions(temperature=2.0, catalysts=frozenset({"Pt"}))
    assert not k.is_catalyzed(uncat) and k.is_catalyzed(cat)
    assert k.activation_energy(cat) == pytest.approx(CATALYST_BARRIER_FACTOR * k.activation_energy(uncat))
    assert k.rate(cat) > k.rate(uncat)
    # the thermodynamics is untouched — the catalyst changes the path, not the destination.
    assert r.delta_G(cat) == r.delta_G(uncat)


def test_catalyst_lowers_the_temperature_needed_for_a_given_rate():
    """The catalyst moves the 'ignition' temperature down (∝ the barrier it removes)."""
    r = _combustion()
    k = kin.kinetics(r, catalysts={"Pt"})
    target = 1.0
    t_uncat = k.temperature_for_rate(target, ChemConditions(temperature=1.0))
    t_cat = k.temperature_for_rate(target, ChemConditions(temperature=1.0, catalysts=frozenset({"Pt"})))
    assert t_cat < t_uncat
    # T ∝ Ea, so halving the barrier halves the temperature-for-rate.
    assert t_cat == pytest.approx(CATALYST_BARRIER_FACTOR * t_uncat)


def test_a_non_catalyst_species_has_no_effect():
    """A species that doesn't catalyse this reaction leaves Ea (and rate) unchanged."""
    k = kin.kinetics(_combustion(), catalysts={"Pt"})
    plain = ChemConditions(temperature=2.0)
    wrong = ChemConditions(temperature=2.0, catalysts=frozenset({"Fe"}))
    assert k.activation_energy(wrong) == k.activation_energy(plain)


# --- honesty: the rate is a smooth exponential, NOT a sharp transition ----------------

def test_rate_is_monotone_and_smooth_in_temperature():
    """rate(T) climbs steeply but continuously — no sign-crossing, no singularity."""
    k = kin.kinetics(_combustion())
    temps = [1.0, 2.0, 4.0, 8.0, 16.0]
    rates = [k.rate(ChemConditions(temperature=t)) for t in temps]
    assert all(b > a for a, b in zip(rates, rates[1:]))  # strictly increasing, no jump-to-spontaneous


def test_ignition_threshold_is_a_soft_cutoff_not_a_transition():
    """The temperature-for-rate slides smoothly as the reference rate moves — the tell that it
    is a soft cutoff, not a genuine threshold (the disguised-dial check, no-fudge norm)."""
    k = kin.kinetics(_combustion())
    cond = ChemConditions(temperature=1.0)
    temps = [k.temperature_for_rate(c, cond) for c in (0.1, 1.0, 10.0, 100.0)]
    # higher reference rate needs higher T, and it moves continuously (no plateau/jump).
    assert all(b > a for a, b in zip(temps, temps[1:]))


def test_rate_unreachable_above_the_pre_exponential_ceiling():
    """A·exp(−Ea/RT) < A always, so a target rate ≥ A is unreachable (returns None)."""
    k = kin.kinetics(_combustion())
    assert k.temperature_for_rate(ARRHENIUS_A * 2, ChemConditions(temperature=1.0)) is None


# --- determinism (spec §14) -----------------------------------------------------------

def test_rate_is_deterministic():
    k = kin.kinetics(_combustion(), catalysts={"Pt"})
    c = ChemConditions(temperature=3.0, pressure=2.0, catalysts=frozenset({"Pt"}))
    assert k.rate(c) == k.rate(c)
    assert k.activation_energy(c) == k.activation_energy(c)
