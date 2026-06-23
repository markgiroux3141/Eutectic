"""M7a keystones: the tight-binding band gap, measured (never assigned) from the lattice.

The no-fudge ladder (spec m7-spectral-spec.md §7):
1. Static, parameter-free: on the real chemistry NaCl crystal the gap = 2Δ with Δ from ΔEN,
   to numerical precision, INDEPENDENT of the hopping t; Cu (uniform metal) shows no gap.
2. Finite-size honesty: gap/level-spacing GROWS with N for NaCl, stays 0 for Cu (raw gap alone
   is rejected as a metal/insulator test).
3. Classification ordering: insulator (NaCl) ordered above conductor (Cu) by ΔEN, emergent.
4. Determinism + M0-M8 stay byte-identical (site_potential is excluded from the signature).
"""

import numpy as np
import pytest

from chemistry import atoms, crystal, molecule
from chemistry.molecule import Molecule
from engine import material
from engine.lattice import Lattice, generate_base, relax
from engine.properties import spectral

SHAPE = (24, 24)


def _nacl(shape=SHAPE):
    m = molecule.form_binary("Na", "Cl")
    assert isinstance(m, Molecule)
    return relax(crystal.compound_crystal(m, shape=shape), seed=12345)


def _cu(shape=SHAPE):
    return relax(crystal.element_crystal("Cu", shape=shape), seed=12345)


# --- KEYSTONE 1: the parameter-free ionic gap = 2*delta, from delta-EN -----------------

def test_nacl_gap_is_two_delta_from_electronegativity():
    nacl = _nacl()
    delta_en = abs(atoms.get("Na").electronegativity - atoms.get("Cl").electronegativity)
    # scale = 1 -> gap = 2*delta = delta_EN. Recovered to numerical precision on the real crystal.
    # (abs=1e-5: the on-site potential is stored as float32, so the exact-arithmetic 1e-14 of the
    # de-risk floors at float32 epsilon ~ 2e-8 here; the gap is still exact to the field precision.)
    assert spectral.raw_gap(nacl) == pytest.approx(delta_en, abs=1e-5)


def test_ionic_gap_is_independent_of_hopping_t():
    nacl = _nacl()
    gaps = [spectral.raw_gap(nacl, t=t) for t in (0.5, 1.0, 2.0, 4.0)]
    # The ionic (on-site stagger) gap is set purely by delta; t only sets the bandwidth.
    assert max(gaps) - min(gaps) < 1e-9


def test_uniform_metal_has_no_gap():
    cu = _cu()
    assert spectral.raw_gap(cu) == pytest.approx(0.0, abs=1e-9)
    assert spectral.band_gap(cu) == 0.0
    assert not spectral.has_onsite_stagger(cu)


def test_dos_at_fermi_separates_metal_from_insulator():
    assert spectral.dos_at_fermi(_nacl()) == pytest.approx(0.0, abs=1e-6)  # gapped: no states at E_F
    assert spectral.dos_at_fermi(_cu()) > 0.1                              # metallic: finite DOS


# --- KEYSTONE 2: the honest finite-size detector --------------------------------------

def test_normalized_gap_grows_with_N_for_insulator():
    # A true gap is N-independent while level spacing ~ bandwidth/N, so gap/spacing grows with N.
    ratios = [spectral.normalized_gap(_nacl((L, L))) for L in (8, 12, 16, 24)]
    assert all(b > a for a, b in zip(ratios, ratios[1:]))  # strictly increasing
    # raw gap, by contrast, is flat (it is the real, size-independent gap).
    raws = [spectral.raw_gap(_nacl((L, L))) for L in (8, 12, 16, 24)]
    assert max(raws) - min(raws) < 1e-9


def test_metal_normalized_gap_stays_small():
    ratios = [spectral.normalized_gap(_cu((L, L))) for L in (8, 12, 16, 24)]
    assert all(r < spectral.METAL_RATIO for r in ratios)  # never registers as a gap


# --- KEYSTONE 3: classification ordering, emergent from delta-EN -----------------------

def test_classification_nacl_insulator_cu_conductor():
    assert spectral.classify(_nacl()) == "insulator"
    assert spectral.classify(_cu()) == "conductor"


def test_gap_rises_with_electronegativity_difference():
    pairs = [("Na", "Br"), ("Na", "Cl"), ("K", "F")]  # increasing delta-EN
    rows = []
    for a, b in pairs:
        m = molecule.form_binary(a, b)
        assert isinstance(m, Molecule)
        lat = relax(crystal.compound_crystal(m, shape=SHAPE), seed=1)
        den = abs(atoms.get(a).electronegativity - atoms.get(b).electronegativity)
        rows.append((den, spectral.raw_gap(lat)))
    dens = [r[0] for r in rows]
    gaps = [r[1] for r in rows]
    assert dens == sorted(dens)                  # the pairs are ordered by delta-EN
    assert gaps == sorted(gaps)                  # and the gaps rise monotonically with it


# --- vacancy brittleness: a CROSSOVER, not a transition (no-fudge norm, spec §6) -------

def test_gap_is_brittle_to_vacancies():
    # The original M7 killer, now a controlled finding: even ~1% vacancies (dangling mid-gap
    # states) collapse the gap. This is why M7 needed the dense (fill=1.0) crystal, and why the
    # conditions coupling is reported as brittleness, not a phase transition.
    nacl = _nacl()
    assert spectral.raw_gap(nacl) > 1.0          # pristine: a real gap
    occ = nacl.occupied.copy()
    rng = np.random.default_rng(7)
    occ[rng.random(occ.shape) < 0.05] = 0        # 5% vacancies
    punched = nacl.copy()
    punched.occupied[:] = occ
    assert spectral.raw_gap(punched) == pytest.approx(0.0, abs=1e-9)


# --- determinism --------------------------------------------------------------------

def test_band_gap_is_deterministic():
    a = spectral.band_gap(_nacl())
    b = spectral.band_gap(_nacl())
    assert a == b


# --- M0-M8 stay byte-identical: site_potential is excluded from the signature ----------

def test_site_potential_excluded_from_structural_signature():
    # The whole byte-identical guarantee: a derived field must not perturb the combination seed.
    base = generate_base(seed=42, shape=SHAPE)
    sig_before = base.structural_signature()
    # inject a non-trivial site_potential; the signature must not move.
    staggered = Lattice(
        occupied=base.occupied, atom_type=base.atom_type, spin=base.spin,
        mass=base.mass, moment=base.moment, cohesion=base.cohesion,
        metallicity=base.metallicity,
        site_potential=np.random.default_rng(0).standard_normal(SHAPE).astype(np.float32),
    )
    assert staggered.structural_signature() == sig_before


def test_default_site_potential_is_zero_and_gap_free():
    base = generate_base(seed=7, shape=SHAPE)
    assert np.all(np.asarray(base.site_potential) == 0.0)
    assert not spectral.has_onsite_stagger(base)
    assert spectral.measure(base) == {"band_gap": 0.0}


def test_measure_properties_adds_band_gap_without_disturbing_others():
    # A generated lattice keeps every prior key and gets band_gap = 0 (no stagger -> conductor).
    base = relax(generate_base(seed=3, shape=SHAPE), seed=3)
    props = material.measure_properties(base)
    for key in ("density", "conductivity", "magnetism", "melting_temperature", "strength"):
        assert key in props
    assert props["band_gap"] == 0.0


def test_nacl_measures_as_a_gapped_insulator():
    props = material.measure_properties(_nacl(shape=(48, 48)))
    assert props["band_gap"] > 1.0          # a real ionic gap
    assert props["conductivity"] == 0.0     # consistent: ionic insulator on both axes
