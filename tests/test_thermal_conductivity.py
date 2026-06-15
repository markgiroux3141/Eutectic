"""M6b: thermal conductivity — two heat carriers, and the diamond divergence (docs §4).

Heat flows two ways on the structure: electronically (the charge-carrying electrons also carry
heat → Wiedemann-Franz, κ_e = L·T·σ) and phononically (lattice vibrations through *all* occupied
matter, weighted by stiffness/mass). Charge is gated by **metallicity** (only metallic cells carry
it), heat is not — so a stiff non-metal (carbon/diamond) conducts heat superbly while carrying no
charge. These tests pin: the carrier split, the divergence, the phonon stiffness/mass dependence,
the WF electronic relation, and that *solid* percolation (melting's input) is independent of
metallicity. There is no phase transition here — it is a deterministic transport measurement.
"""

import numpy as np

from engine import elements
from engine.lattice import Lattice
from engine.material import from_element
from engine.properties import conductance as C
from engine.properties import percolation
from engine.properties.percolation import METALLIC_THRESHOLD


def _lat(occupied, *, metallicity=1.0, cohesion=1.0, mass=1.0):
    occ = np.asarray(occupied, dtype=np.uint8)
    shape = occ.shape
    return Lattice(
        occupied=occ,
        atom_type=np.where(occ == 1, 1, 0).astype(np.int8),
        spin=np.ones(shape, np.int8),
        mass=np.where(occ == 1, mass, 0.0).astype(np.float32),
        cohesion=np.full(shape, cohesion, np.float32),
        metallicity=np.full(shape, metallicity, np.float32),
    )


# --- the carrier split: metallicity gates charge, not heat ------------------------------


def test_metallicity_gates_charge_but_not_heat():
    """A non-metallic solid carries NO charge (σ=0) but still conducts heat (phonons)."""
    insulator = _lat(np.ones((16, 16)), metallicity=METALLIC_THRESHOLD - 0.1)
    metal = _lat(np.ones((16, 16)), metallicity=METALLIC_THRESHOLD + 0.1)
    assert C.conductivity(insulator) == 0.0          # charge gated off
    assert C.phonon_conductivity(insulator) > 0.0    # heat still flows
    assert C.conductivity(metal) > 0.0               # metal carries charge


def test_diamond_divergence_in_the_pipeline():
    """Carbon (non-metallic, stiff, light) is an electrical insulator but a top heat conductor."""
    carbon = from_element(elements.get("carbon"), shape=(48, 48)).properties
    copper = from_element(elements.get("copper"), shape=(48, 48)).properties
    # Diamond divergence: zero electrical conduction, substantial thermal conduction.
    assert carbon["conductivity"] == 0.0
    assert carbon["thermal_conductivity"] > 0.05
    # ...and carbon out-conducts copper *thermally* (its phonon channel beats copper's total).
    assert carbon["thermal_conductivity"] > copper["thermal_conductivity"]
    # Copper conducts both.
    assert copper["conductivity"] >= 0.5 and copper["thermal_conductivity"] > 0.0


def test_phonon_conductivity_tracks_stiffness_and_mass():
    """Phonon transport rises with stiffness (cohesion) and falls with atomic mass."""
    full = np.ones((20, 20))
    stiff_light = _lat(full, cohesion=1.3, mass=10.0)
    floppy_heavy = _lat(full, cohesion=0.6, mass=200.0)
    assert C.phonon_conductivity(stiff_light) > C.phonon_conductivity(floppy_heavy)


# --- Wiedemann-Franz electronic channel ------------------------------------------------


def test_wiedemann_franz_electronic_channel():
    """The electronic part of κ is exactly L·T·σ (the single-carrier WF relation)."""
    metal = _lat((np.indices((24, 24)).sum(0) % 5 != 0).astype(np.uint8))  # dense, metallic
    sigma = C.conductivity(metal)
    assert sigma > 0
    kappa = C.thermal_conductivity(metal, temperature=C.STANDARD_T)
    electronic = kappa - C.phonon_conductivity(metal)
    assert abs(electronic - C.WF_LORENZ * C.STANDARD_T * sigma) < 1e-9
    # Wiedemann-Franz: κ_e / (σ·T) recovers the Lorenz number.
    assert abs(electronic / (sigma * C.STANDARD_T) - C.WF_LORENZ) < 1e-6


# --- solid percolation (melting's input) is independent of metallicity ------------------


def test_solid_percolation_independent_of_metallicity():
    """A dense NON-metallic lattice still has a spanning *solid* cluster (so it can melt),
    even though it carries no charge — the matter/charge mask split (M6b)."""
    dense_insulator = _lat(np.ones((20, 20)), metallicity=0.1)
    # Solid network spans (matter), gating melting/phonons...
    assert percolation.largest_cluster_fraction(dense_insulator) == 1.0
    assert percolation.spanning_fraction(dense_insulator) == 1.0
    # ...while the charge backbone is empty.
    assert C.conductivity(dense_insulator) == 0.0
    assert not percolation.conducting_mask(dense_insulator).any()
    assert percolation.solid_mask(dense_insulator).all()


# --- defaults, carrying, determinism ---------------------------------------------------


def test_constructed_lattice_defaults_metallic():
    """A Lattice built without a metallicity field defaults to metallic (M2 charge = occupied)."""
    occ = np.ones((8, 8), np.uint8)
    lat = Lattice(occupied=occ, atom_type=occ.astype(np.int8), spin=occ.astype(np.int8))
    assert np.all(np.asarray(lat.metallicity) == 1.0)
    assert percolation.conducting_mask(lat).all()  # charge backbone = occupied


def test_metallicity_from_conduction_tendency():
    """generate_base writes per-cell metallicity from the element's conduction_tendency."""
    copper = elements.get("copper").lattice(shape=(16, 16))     # conduction 0.97 -> metallic
    sulfur = elements.get("sulfur").lattice(shape=(16, 16))     # conduction 0.20 -> non-metal
    occ_c = copper.occupied == 1
    occ_s = sulfur.occupied == 1
    assert np.all(np.asarray(copper.metallicity)[occ_c] >= METALLIC_THRESHOLD)
    assert np.all(np.asarray(sulfur.metallicity)[occ_s] < METALLIC_THRESHOLD)


def test_thermal_conductivity_is_deterministic():
    """No RNG in transport: same structure -> identical thermal conductivity."""
    lat = from_element(elements.get("iron"), shape=(32, 32)).lattice
    assert C.thermal_conductivity(lat) == C.thermal_conductivity(lat)
    assert C.measure(lat) == C.measure(lat)
