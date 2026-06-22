"""C2 keystone: compound bulk properties match its chemistry (spec §9, §15).

The integration keystone — chemistry generates a crystal, the *existing* materials extractors
measure it unchanged, and the measured bulk properties match what the bonding implies:
NaCl insulates, Cu conducts, diamond is an electrical insulator that conducts heat superbly,
an unpaired-electron crystal (Fe) is magnetic while a closed-shell metal (Cu) is not.

Also pins the honest affinity-derivation finding (why magnetism can't be derived from
unpaired-electron count) so the negative stays documented, not buried.
"""

import numpy as np
import pytest

from chemistry import atoms, crystal, molecule
from chemistry.molecule import Molecule
from engine import material
from engine.lattice import relax
from engine.properties import conductance, ising, percolation

SHAPE = (48, 48)


def _nacl():
    m = molecule.form_binary("Na", "Cl")
    assert isinstance(m, Molecule)
    return crystal.compound_crystal(m, shape=SHAPE)


# --- KEYSTONE: electrical conductivity tracks bond character --------------------------

def test_nacl_insulates():
    # Ionic: no metallic cells -> no charge backbone -> not an electrical conductor.
    assert percolation.conductivity_boolean(_nacl()) is False


def test_copper_conducts():
    assert percolation.conductivity_boolean(crystal.element_crystal("Cu", shape=SHAPE)) is True


def test_diamond_insulates_electrically():
    assert percolation.conductivity_boolean(crystal.element_crystal("C", shape=SHAPE)) is False


# --- KEYSTONE: the diamond divergence (electrical insulator, superb heat conductor) ---

def test_diamond_conducts_heat_despite_no_charge():
    dia = crystal.element_crystal("C", shape=SHAPE)
    # Electrical insulator...
    assert percolation.conductivity_boolean(dia) is False
    # ...yet its phonon (solid-network) thermal channel is wide open.
    assert conductance.phonon_conductivity(dia) > 0.0


def test_diamond_beats_a_metal_on_phonon_conduction():
    # Stiff + light carbon network carries phonons better than a heavy soft metal (M6b).
    dia = conductance.phonon_conductivity(crystal.element_crystal("C", shape=SHAPE))
    lead = conductance.phonon_conductivity(crystal.element_crystal("Pb", shape=SHAPE))
    assert dia > lead


# --- KEYSTONE: magnetism from real (unpaired) electron structure ----------------------

def test_iron_crystal_is_magnetic():
    fe = relax(crystal.element_crystal("Fe", shape=SHAPE), seed=12345)
    assert ising.magnetism(fe) > 0.5


def test_copper_crystal_is_not_magnetic():
    cu = relax(crystal.element_crystal("Cu", shape=SHAPE), seed=12345)
    assert ising.magnetism(cu) < 0.2


def test_covalent_and_ionic_crystals_are_not_magnetic():
    # The solid-state moment fix: s/p electrons are quenched by bonding, so diamond (covalent)
    # and NaCl (ionic, closed-shell ions) carry no moment even though free C/Cl have unpaired
    # p-electrons. Without the d/f-only rule these spuriously magnetised.
    dia = relax(crystal.element_crystal("C", shape=SHAPE), seed=3)
    assert ising.magnetism(dia) < 0.2
    nacl = relax(_nacl(), seed=3)
    assert ising.magnetism(nacl) < 0.2


def test_iron_outmagnetises_copper():
    fe = ising.magnetism(relax(crystal.element_crystal("Fe", shape=SHAPE), seed=7))
    cu = ising.magnetism(relax(crystal.element_crystal("Cu", shape=SHAPE), seed=7))
    assert fe > cu


# --- the integration thesis: existing extractors measure a compound UNCHANGED ---------

def test_materials_extractors_run_on_a_compound_unchanged():
    # engine.material.measure_properties (the full M2..M8 extractor suite) must accept a
    # chemistry-built crystal with no changes — that is the whole point of C2.
    props = material.measure_properties(_nacl())
    assert props["conductivity"] == 0.0          # NaCl insulates
    assert props["fill_fraction"] == 1.0          # a dense crystal
    assert props["density"] > 0.0
    assert "melting_temperature" in props and "strength" in props


def test_copper_compound_measures_as_conductor():
    props = material.measure_properties(crystal.element_crystal("Cu", shape=SHAPE))
    assert props["conductivity"] == 1.0


# --- determinism (spec §14) -----------------------------------------------------------

def test_crystal_build_is_deterministic():
    a = crystal.element_crystal("Fe", shape=SHAPE)
    b = crystal.element_crystal("Fe", shape=SHAPE)
    assert a.structural_signature() == b.structural_signature()
    assert np.array_equal(a.atom_type, b.atom_type)


def test_noble_gas_forms_no_crystal():
    with pytest.raises(ValueError):
        crystal.element_crystal("Ne", shape=SHAPE)


# --- the honest affinity-derivation finding, pinned as a test -------------------------

def test_unpaired_count_cannot_distinguish_ferromagnet_from_antiferromagnet():
    # The make-or-break negative: ferromagnetism is the Stoner criterion, NOT unpaired count.
    # Cr (antiferromagnet) has the SAME unpaired count as Fe (ferromagnet) -> a moment derived
    # from unpaired electrons alone would wrongly magnetise Cr. This is why we KEEP the authored
    # element affinities rather than deriving magnetic_tendency (M3/M4 would otherwise break).
    assert atoms.localized_unpaired_electrons(atoms.get("Cr").z) == \
        atoms.localized_unpaired_electrons(atoms.get("Fe").z)
    # ...while the authored engine affinities correctly separate them:
    from engine import elements
    assert elements.get("chromium").base_affinities["magnetic_tendency"] < \
        elements.get("iron").base_affinities["magnetic_tendency"]
