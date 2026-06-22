"""C1c keystone: stoichiometry falls out of valence/charge at the energy minimum (spec §8, §15).

The headline keystone of C1: ratios are never assigned. They emerge — ionic from charge
neutrality, covalent from the energy-minimizing ratio — parameter-free. We also pressure-test
the *energy-minimum* claim itself (the satisfied ratio must beat its neighbours), and that
noble gases and non-opposing pairs refuse to form.
"""

import pytest

from chemistry import molecule
from chemistry.bonding import BondCharacter
from chemistry.molecule import Molecule, NoCompound


def _formed(a, b) -> Molecule:
    m = molecule.form_binary(a, b)
    assert isinstance(m, Molecule), f"{a}+{b} did not form: {getattr(m, 'reason', m)}"
    return m


# --- KEYSTONE: the textbook ratios --------------------------------------------------

@pytest.mark.parametrize("a,b,formula,counts", [
    ("Na", "Cl", "NaCl",  {"Na": 1, "Cl": 1}),
    ("Mg", "Cl", "MgCl2", {"Mg": 1, "Cl": 2}),
    ("H",  "O",  "H2O",   {"O": 1, "H": 2}),
    ("C",  "O",  "CO2",   {"C": 1, "O": 2}),
    ("Mg", "O",  "MgO",   {"Mg": 1, "O": 1}),
    ("Al", "O",  "Al2O3", {"Al": 2, "O": 3}),
])
def test_stoichiometry_keystone(a, b, formula, counts):
    m = _formed(a, b)
    assert m.counts == counts, (a, b, m.counts)
    assert m.formula == formula


def test_water_is_two_to_one():
    m = _formed("H", "O")
    assert m.counts["H"] == 2 and m.counts["O"] == 1


def test_co2_has_double_bonds():
    m = _formed("C", "O")
    assert all(bond.order == 2 for bond in m.bonds)
    assert m.character is BondCharacter.POLAR_COVALENT


def test_ionic_compounds_are_charge_neutral():
    for a, b in [("Na", "Cl"), ("Mg", "Cl"), ("Al", "O"), ("Mg", "O")]:
        m = _formed(a, b)
        assert m.character is BondCharacter.IONIC
        assert sum(m.formal_charges) == 0, (a, b, m.formal_charges)


# --- KEYSTONE: noble gases / non-opposing pairs refuse --------------------------------

@pytest.mark.parametrize("a,b", [("He", "Cl"), ("Ne", "O"), ("Ar", "Na")])
def test_noble_gases_refuse(a, b):
    assert isinstance(molecule.form_binary(a, b), NoCompound)


# --- the energy-minimum claim is real (not an asserted rule) --------------------------

@pytest.mark.parametrize("central,term,satisfied_n", [("O", "H", 2), ("C", "H", 4), ("C", "O", 2)])
def test_satisfied_ratio_is_the_energy_minimum(central, term, satisfied_n):
    from chemistry.atoms import get
    c, t = get(central), get(term)
    energies = {n: molecule._covalent_energy(c, t, n) for n in range(1, c.bonding_capacity + 1)}
    best_n = min(energies, key=energies.get)
    assert best_n == satisfied_n, energies


# --- homonuclear diatomics fall out of the same rule ----------------------------------

@pytest.mark.parametrize("sym,order", [("H", 1), ("O", 2), ("N", 3), ("Cl", 1)])
def test_homonuclear_diatomic_order(sym, order):
    m = _formed(sym, sym)
    assert m.counts[sym] == 2
    assert m.bonds[0].order == order


# --- geometry is attached to formed covalent molecules --------------------------------

def test_formed_water_carries_bent_geometry():
    m = _formed("H", "O")
    assert m.geometry is not None
    assert m.geometry.shape == "bent"
    assert m.geometry.bond_angle == pytest.approx(104.5, abs=0.05)


def test_formed_co2_is_linear():
    m = _formed("C", "O")
    assert m.geometry.shape == "linear"
    assert m.geometry.bond_angle == pytest.approx(180.0, abs=0.05)


# --- determinism + canonical identity (spec §14) --------------------------------------

def test_formation_is_deterministic_and_canonical():
    m1 = _formed("H", "O")
    m2 = _formed("O", "H")     # construction order swapped
    assert m1.canonical_id() == m2.canonical_id()
    assert m1.formula == m2.formula == "H2O"
