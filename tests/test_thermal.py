"""The M4 keystone: the thermal-ensemble engine and its validation (docs §4, §6).

The architecture's proof is the *coincidence*: heat capacity ``C(T)``, measured from energy
fluctuations, peaks exactly where the order parameter ``⟨|M|⟩`` collapses — and, for a plain
unit-moment lattice, at the textbook 2D-Ising ``Tc = 2/ln(1+√2) ≈ 2.269`` with NO free
parameters. If these hold, "measure properties from a structure at conditions" is real, not
a fit. We do not proceed to M5 without them (per the project's no-fudge norm).
"""

import numpy as np

from engine import elements, thermal
from engine.conditions import STANDARD, Conditions
from engine.lattice import Lattice
from engine.material import from_element

ISING_TC = 2.0 / np.log(1 + np.sqrt(2))  # 2.2692..., exact for the infinite 2D lattice


def _full_lattice(shape, moment=1.0):
    """A fully-occupied, uniform-moment lattice: plain 2D Ising with J = J0·moment²."""
    occ = np.ones(shape, dtype=np.uint8)
    atom_type = np.ones(shape, dtype=np.int8)
    spin = np.ones(shape, dtype=np.int8)
    moment_field = np.full(shape, moment, dtype=np.float32)
    return Lattice(occupied=occ, atom_type=atom_type, spin=spin, moment=moment_field)


def _peak_temperature(temps, sweep):
    i = int(np.argmax([s.heat_capacity for s in sweep]))
    return float(temps[i])


# --- energy + ensemble basics ---------------------------------------------------------


def test_energy_is_lower_when_aligned():
    """Ferromagnetic ground state: a fully aligned spin field beats an antialigned one."""
    lat = _full_lattice((16, 16))
    aligned = np.ones((16, 16), dtype=np.int8)
    antialigned = np.where(np.indices((16, 16)).sum(0) % 2 == 0, 1, -1).astype(np.int8)
    assert thermal.energy(lat, aligned) < thermal.energy(lat, antialigned)


def test_sample_ensemble_is_deterministic():
    """Same structure + same conditions -> byte-identical observables (spec §6)."""
    lat = _full_lattice((24, 24))
    a = thermal.sample_ensemble(lat, Conditions(temperature=2.0), n_samples=20)
    b = thermal.sample_ensemble(lat, Conditions(temperature=2.0), n_samples=20)
    assert a == b


def test_field_biases_magnetization():
    """The conjugate field H breaks symmetry: above Tc, H>0 induces positive net M."""
    lat = _full_lattice((24, 24))
    hot = 3.4  # well above Tc -> zero-field magnetization averages to ~0
    zero = thermal.sample_ensemble(lat, Conditions(temperature=hot, field=0.0), n_samples=40)
    field = thermal.sample_ensemble(lat, Conditions(temperature=hot, field=0.6), n_samples=40)
    assert abs(zero.mean_mag) < 0.2
    assert field.mean_mag > zero.mean_mag + 0.2
    assert field.mean_mag > 0.2


# --- KEYSTONE 1: recover the textbook 2D-Ising Tc with no free parameters --------------


def test_heat_capacity_peaks_at_ising_tc():
    """A unit-moment full lattice: C(T) peaks at the parameter-free 2D-Ising point ~2.269."""
    lat = _full_lattice((32, 32))
    temps = np.linspace(1.8, 2.8, 11)
    sweep = thermal.temperature_sweep(lat, temps, burn_in=150, n_samples=60, sample_every=2)
    peak = _peak_temperature(temps, sweep)
    # Finite-size + finite-sampling broaden/shift the peak slightly; ±0.25 of the exact value.
    assert abs(peak - ISING_TC) < 0.25, f"C-peak at {peak:.3f}, expected ~{ISING_TC:.3f}"


# --- KEYSTONE 2: the coincidence — C(T) peaks where the order parameter collapses ------


def test_heat_capacity_peak_coincides_with_order_collapse():
    """On a real ferromagnet, the C(T) peak sits exactly at the ⟨|M|⟩ collapse.

    This is THE validation that the architecture works (docs §4): the transition the order
    parameter shows and the transition the energy fluctuations detect are the *same*
    temperature — not two separately-tuned numbers.
    """
    lat = _full_lattice((32, 32))
    temps = np.linspace(1.8, 2.8, 11)
    sweep = thermal.temperature_sweep(lat, temps, burn_in=150, n_samples=60, sample_every=2)
    mags = np.array([s.mean_abs_mag for s in sweep])

    peak_T = _peak_temperature(temps, sweep)
    # Order parameter: high (ordered) below the peak, low (disordered) above it.
    below = mags[temps < peak_T]
    above = mags[temps > peak_T]
    assert below.size and above.size
    assert below.mean() > 0.7, f"not ordered below the peak: {below.mean():.2f}"
    assert above.mean() < 0.3, f"not disordered above the peak: {above.mean():.2f}"

    # The ⟨|M|⟩ collapse (crossing 0.5) lands within ~one grid step of the C peak.
    cross = np.interp(0.5, mags[::-1], temps[::-1])  # temps ascending, mags descending
    step = float(temps[1] - temps[0])
    assert abs(cross - peak_T) <= 1.5 * step, f"M-collapse {cross:.3f} vs C-peak {peak_T:.3f}"


def test_real_ferromagnet_has_a_curie_point():
    """Iron's structure orders and yields a Curie point above standard conditions."""
    iron = from_element(elements.get("iron"), shape=(32, 32))
    tc = thermal.curie_temperature(iron.lattice, t_lo=1.0, t_hi=3.5, n_temps=11)
    assert tc is not None
    assert tc > STANDARD.temperature  # ferromagnetic at and above T0


# --- negative control: no order -> no Curie point -------------------------------------


def test_nonmagnet_has_no_curie_point():
    """Copper (low moment) never orders: max ⟨|M|⟩ stays low, curie_temperature is None."""
    copper = from_element(elements.get("copper"), shape=(32, 32))
    tc, sweep = thermal.curie_temperature(
        copper.lattice, t_lo=1.0, t_hi=3.5, n_temps=11, return_curve=True
    )
    assert tc is None
    assert max(s.mean_abs_mag for s in sweep) < 0.30


# --- the gated stored property (material pipeline) ------------------------------------


def test_stored_curie_temperature_gating():
    """measure_properties stores Tc>0 for ferromagnets, 0.0 for non-magnets (the gate)."""
    iron = from_element(elements.get("iron"), shape=(32, 32))
    copper = from_element(elements.get("copper"), shape=(32, 32))
    assert iron.properties["curie_temperature"] > 1.0
    assert copper.properties["curie_temperature"] == 0.0
