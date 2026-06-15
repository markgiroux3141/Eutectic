"""The M6 keystone: honest superconductivity as XY phase coherence with a *real* Tc (docs §4).

Superconductivity is phase coherence of the order parameter — modelled as an XY model on the
conducting backbone (``engine.lattice.xy_sweep``), whose ordering Tc is *measured* (the helicity
modulus crossing the BKT universal line), replacing the retired static k-edge-connectivity proxy.

Validations, pressure-tested the same way as Curie (M4) and melting (M5):

1. a fully-conducting lattice phase-coheres at the textbook 2D-XY ``T_BKT = 0.893·J`` (parameter
   free), via the helicity-modulus universal-line crossing;
2. the BKT subtlety we must respect: the **heat-capacity peak sits ABOVE Tc** — so (unlike
   Curie/melting) the C-peak does NOT locate this transition; the stiffness detector does;
3. Tc *emerges from structure*: a redundant solid backbone coheres hotter than a thin diluted one;
4. an insulator (no spanning backbone) has no Tc; the ensemble is deterministic (spec §6); and the
   stored reference-condition snapshot (stiffness + flag) is gated by conductivity.
"""

import numpy as np

from engine import elements, thermal
from engine.conditions import Conditions
from engine.lattice import SC_J0, Lattice
from engine.material import from_element

T_BKT = 0.8929  # 2D XY BKT transition for J=1 (literature)


def _conducting_lattice(shape, fill=1.0, seed=1):
    if fill >= 1.0:
        occ = np.ones(shape, dtype=np.uint8)
    else:
        from engine.rng import SplitMix64
        occ = (SplitMix64(seed).numpy_generator().random(shape) < fill).astype(np.uint8)
    atom_type = np.where(occ == 1, 1, 0).astype(np.int8)
    spin = np.ones(shape, dtype=np.int8)
    return Lattice(occupied=occ, atom_type=atom_type, spin=spin)


# --- KEYSTONE 1: recover the textbook 2D-XY BKT transition, parameter-free --------------


def test_clean_lattice_tc_is_textbook_bkt():
    """A fully-conducting lattice (uniform J=1) phase-coheres at the parameter-free 0.893."""
    lat = _conducting_lattice((32, 32))
    tc = thermal.superconducting_tc(lat, coupling=1.0, n_temps=14,
                                    burn_in=300, n_samples=120)
    assert tc is not None
    # Finite-size + finite-sampling push the universal-line crossing slightly high; ~±0.12.
    assert abs(tc - T_BKT) < 0.12, f"BKT Tc at {tc:.3f}, expected ~{T_BKT:.3f}"


# --- KEYSTONE 2: the BKT subtlety — the C-peak is ABOVE Tc (do NOT use it as the detector) -


def test_heat_capacity_peak_is_above_tc():
    """For the XY/BKT transition the C-peak sits above Tc, unlike Curie/melting — proven here.

    Uses a sweep wide enough to span the C-peak (~1.04 for 2D XY); the helicity-modulus crossing
    locates Tc (~0.9) below it. The whole point of M6's discipline: do NOT read Tc off the C-peak.
    """
    lat = _conducting_lattice((32, 32))
    temps = np.linspace(0.4, 1.5, 16)
    sweep = thermal.xy_temperature_sweep(lat, temps, coupling=1.0,
                                         burn_in=300, n_samples=140)
    tc = thermal._universal_crossing(temps, [s.helicity_modulus for s in sweep])
    c_peak_T = float(temps[int(np.argmax([s.heat_capacity for s in sweep]))])
    assert tc is not None
    assert c_peak_T > tc, f"C-peak ({c_peak_T:.3f}) should be ABOVE Tc ({tc:.3f}) for BKT"


# --- KEYSTONE 3: Tc emerges from backbone structure (denser -> stiffer -> hotter Tc) -----


def test_tc_rises_with_backbone_density():
    """A solid backbone coheres at a higher Tc than a thin near-percolation one (emergent)."""
    solid = thermal.superconducting_tc(_conducting_lattice((32, 32), 1.0),
                                       coupling=1.0, n_temps=12, burn_in=250, n_samples=100)
    thin = thermal.superconducting_tc(_conducting_lattice((32, 32), 0.72, seed=3),
                                      coupling=1.0, n_temps=12, burn_in=250, n_samples=100)
    assert solid is not None and thin is not None
    assert solid > thin + 0.15, f"Tc did not rise with density: solid={solid:.2f} thin={thin:.2f}"


def test_helicity_modulus_brackets_the_transition():
    """Stiffness is above the universal line below Tc and collapses below it above Tc."""
    lat = _conducting_lattice((28, 28))
    slope = 2.0 / np.pi
    cold = thermal.sample_xy_ensemble(lat, Conditions(temperature=0.45),
                                      burn_in=250, n_samples=100)
    hot = thermal.sample_xy_ensemble(lat, Conditions(temperature=1.30),
                                     burn_in=250, n_samples=100)
    assert cold.helicity_modulus > slope * 0.45      # coherent (superconducting) well below Tc
    assert hot.helicity_modulus < slope * 1.30       # normal well above Tc


# --- insulator has no Tc; determinism; pipeline gating ---------------------------------


def test_insulator_has_no_tc():
    """A checkerboard (no face-connected backbone) never coheres -> no superconducting Tc."""
    checker = (np.indices((24, 24)).sum(0) % 2 == 0).astype(np.uint8)
    lat = Lattice(occupied=checker, atom_type=checker.astype(np.int8),
                  spin=np.ones((24, 24), np.int8))
    tc = thermal.superconducting_tc(lat, coupling=1.0, n_temps=10, burn_in=150, n_samples=60)
    assert tc is None


def test_xy_ensemble_is_deterministic():
    """Same structure + conditions -> byte-identical phase-coherence observables (spec §6)."""
    lat = _conducting_lattice((24, 24))
    a = thermal.sample_xy_ensemble(lat, Conditions(temperature=0.6), n_samples=30)
    b = thermal.sample_xy_ensemble(lat, Conditions(temperature=0.6), n_samples=30)
    assert a == b


def test_superconductivity_is_not_a_stored_material_property():
    """M6 retires the static proxy; SC is on-demand (not stored), so no `superconductor` field.

    A material still carries ``edge_connectivity`` (the structural input to Tc), but the SC
    transition itself is measured on demand (the keystone tests above / explorer `sc-sweep`) —
    deliberately not stored, because XY/BKT equilibration is too expensive to run per material.
    """
    copper = from_element(elements.get("copper"), shape=(40, 40))
    assert "superconductor" not in copper.properties      # static proxy retired, not replaced by a stored flag
    assert "sc_phase_stiffness" not in copper.properties
    assert "edge_connectivity" in copper.properties        # the structural input is still stored


def test_sc_coupling_unit_is_one():
    """SC_J0=1 is the natural unit in which the clean-lattice keystone recovers 0.893."""
    assert SC_J0 == 1.0
