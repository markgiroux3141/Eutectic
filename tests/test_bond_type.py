"""C1b keystone: bond character + order + energy (chemistry-engine-spec §7, §15).

Parameter-free / one-or-two-calibrated-constants: character from ΔEN thresholds + the
metallic ceiling; order from shared pairs; energy ordering from the distilled model. The
no-fudge discipline: assert the textbook *ordering*, never a fudged absolute number, and only
*within* a character (ionic vs covalent magnitudes are not cross-calibrated until C2).
"""

import pytest

from chemistry import atoms, bonding
from chemistry.bonding import BondCharacter


# --- KEYSTONE: character from ΔEN ------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("Na", "Cl", BondCharacter.IONIC),
    ("Mg", "O", BondCharacter.IONIC),
    ("Cl", "Cl", BondCharacter.COVALENT),
    ("H", "H", BondCharacter.COVALENT),
    ("O", "O", BondCharacter.COVALENT),
    ("Cu", "Cu", BondCharacter.METALLIC),
    ("Fe", "Fe", BondCharacter.METALLIC),
    ("H", "O", BondCharacter.POLAR_COVALENT),
    ("C", "O", BondCharacter.POLAR_COVALENT),
])
def test_bond_character(a, b, expected):
    assert bonding.bond_character(atoms.get(a), atoms.get(b)) is expected


def test_covalent_metallic_split_is_by_absolute_en():
    # Both ΔEN = 0, but chlorine is a nonmetal (covalent) and copper a metal (metallic).
    assert bonding.bond_character(atoms.get("Cl"), atoms.get("Cl")) is BondCharacter.COVALENT
    assert bonding.bond_character(atoms.get("Cu"), atoms.get("Cu")) is BondCharacter.METALLIC


def test_noble_gas_has_no_bond_character():
    assert bonding.bond_character(atoms.get("Ne"), atoms.get("Cl")) is None
    assert bonding.delta_en(atoms.get("He"), atoms.get("O")) is None


# --- KEYSTONE: bond-energy ordering (within covalent) ---------------------------------

def test_bond_order_energy_triple_gt_double_gt_single():
    # Homonuclear diatomics: order = capacity; energy must rise with order.
    cl2 = bonding.covalent_bond_energy(atoms.get("Cl"), atoms.get("Cl"), 1)
    o2 = bonding.covalent_bond_energy(atoms.get("O"), atoms.get("O"), 2)
    n2 = bonding.covalent_bond_energy(atoms.get("N"), atoms.get("N"), 3)
    assert cl2 < o2 < n2


def test_higher_order_same_pair_is_stronger():
    cc = atoms.get("C")
    e1 = bonding.covalent_bond_energy(cc, cc, 1)
    e2 = bonding.covalent_bond_energy(cc, cc, 2)
    e3 = bonding.covalent_bond_energy(cc, cc, 3)
    assert e1 < e2 < e3


def test_ionic_energy_grows_with_charge_product():
    # MgO (2·2) ion-pair Coulomb beats NaCl (1·1): higher charge product, similar radii.
    nacl = bonding.ionic_bond_energy(atoms.get("Na"), atoms.get("Cl"), 1, 1)
    mgo = bonding.ionic_bond_energy(atoms.get("Mg"), atoms.get("O"), 2, 2)
    assert mgo > nacl


# --- make_bond integration ------------------------------------------------------------

def test_make_bond_carries_character_and_positive_energy():
    bond = bonding.make_bond(atoms.get("O"), atoms.get("O"), 0, 1, order=2)
    assert bond.character is BondCharacter.COVALENT
    assert bond.order == 2
    assert bond.energy > 0


def test_make_bond_refuses_noble_gas():
    with pytest.raises(ValueError):
        bonding.make_bond(atoms.get("Ne"), atoms.get("O"), 0, 1, order=1)
