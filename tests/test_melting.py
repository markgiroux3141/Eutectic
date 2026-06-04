"""The M5 keystone: occupancy becomes thermal and crystalline **melting** falls out (docs §4/§5).

Same proof structure as the M4 Curie keystone, but for *positional* order. ``occupied`` is now
a non-conserved repulsive lattice gas (``engine.lattice.occupancy_sweep``); its order-disorder
transition is melting. The validations, all parameter-free where a known value exists:

1. a uniform-cohesion lattice melts at the textbook 2D point ``T_m = 2.269·J0·⟨c²⟩``;
2. the heat-capacity peak (the universal detector) coincides with the staggered-order collapse,
   AND the mean density stays pinned at ½ across it — order lost at *fixed* density, i.e.
   crystalline melting, NOT sublimation (the distinguishing signature of the user's chosen model);
3. ``T_m`` scales with the structural cohesion (a temperature-axis transition set by bonding,
   not the fill-axis percolation threshold);
4. melting tracks ``bond_energy`` — refractory elements (tungsten) melt above soft ones (zinc),
   recovering the real ordering with no per-material tuning (the M5 analogue of iron>cobalt>nickel);
5. pressure is now live: it tunes density and drives percolation (M2 re-validated under the new
   dynamics); and the whole thing is deterministic (spec §6).
"""

import numpy as np

from engine import elements, thermal
from engine.conditions import Conditions
from engine.lattice import COHESION_J0, Lattice
from engine.material import MELT_GATE_FLOOR, from_element
from engine.properties import percolation

ISING_TC = thermal.ISING_TC_2D  # 2.2692..., exact for the infinite 2D lattice


def _uniform_lattice(shape, cohesion=1.0):
    """A lattice with uniform cohesion: the plain repulsive lattice gas (clean order-disorder)."""
    occ = np.ones(shape, dtype=np.uint8)
    atom_type = np.ones(shape, dtype=np.int8)
    spin = np.ones(shape, dtype=np.int8)
    coh = np.full(shape, cohesion, dtype=np.float32)
    return Lattice(occupied=occ, atom_type=atom_type, spin=spin, cohesion=coh)


# --- KEYSTONE 1: recover the textbook 2D point with no free parameters -----------------


def test_melting_point_is_textbook_2d_ising_for_uniform_cohesion():
    """Uniform cohesion=1: the occupancy C(T) peak lands on the parameter-free 2.269."""
    lat = _uniform_lattice((40, 40), cohesion=1.0)
    tm = thermal.melting_point(lat, n_temps=13, bracket=(0.6, 1.4),
                               burn_in=120, n_samples=60)
    assert tm is not None
    assert abs(tm - ISING_TC) < 0.25, f"melting C-peak at {tm:.3f}, expected ~{ISING_TC:.3f}"


# --- KEYSTONE 2: the coincidence AND the crystalline signature (fixed density) ----------


def test_melting_peak_coincides_with_order_collapse_at_fixed_density():
    """C(T) peak == staggered-order collapse, while density stays at ½ (melting, not sublimation)."""
    lat = _uniform_lattice((40, 40), cohesion=1.0)
    temps = np.linspace(1.4, 3.1, 13)
    sweep = thermal.occupancy_temperature_sweep(lat, temps, burn_in=120, n_samples=60)
    psis = np.array([s.staggered_order for s in sweep])
    caps = np.array([s.heat_capacity for s in sweep])
    rhos = np.array([s.mean_density for s in sweep])

    peak_T = float(temps[int(np.argmax(caps))])
    below = psis[temps < peak_T]
    above = psis[temps > peak_T]
    assert below.size and above.size
    assert below.mean() > 0.7, f"not ordered below the peak: {below.mean():.2f}"
    assert above.mean() < 0.3, f"not disordered above the peak: {above.mean():.2f}"

    # THE crystalline signature: positional order is lost at *fixed* density.
    assert rhos.max() - rhos.min() < 0.05, f"density not fixed across melting: {rhos}"
    assert abs(rhos.mean() - 0.5) < 0.03, f"symmetric mu should hold half-filling: {rhos.mean():.3f}"

    # ψ collapse (crossing 0.5) lands within ~one grid step of the C peak.
    cross = np.interp(0.5, psis[::-1], temps[::-1])
    step = float(temps[1] - temps[0])
    assert abs(cross - peak_T) <= 1.5 * step, f"psi-collapse {cross:.3f} vs C-peak {peak_T:.3f}"


# --- KEYSTONE 3: T_m is set by structure (cohesion), on the temperature axis ------------


def test_melting_point_scales_with_cohesion():
    """Stiffer bonds melt hotter: T_m(strong) > T_m(weak), tracking 2.269·J0·c² (not p_c)."""
    soft = thermal.melting_point(_uniform_lattice((36, 36), 0.8),
                                 n_temps=11, bracket=(0.7, 1.3), burn_in=90, n_samples=45)
    stiff = thermal.melting_point(_uniform_lattice((36, 36), 1.3),
                                  n_temps=11, bracket=(0.7, 1.3), burn_in=90, n_samples=45)
    assert soft is not None and stiff is not None
    assert stiff > soft + 0.5, f"T_m did not rise with cohesion: soft={soft:.2f} stiff={stiff:.2f}"
    # Each lands near its own analytic prediction (parameter-free), not at a shared p_c.
    assert abs(soft - ISING_TC * COHESION_J0 * 0.8**2) < 0.35
    assert abs(stiff - ISING_TC * COHESION_J0 * 1.3**2) < 0.4


# --- KEYSTONE 4: melting tracks bond_energy -> recovers the real ordering ----------------


def test_melting_tracks_bond_energy_across_real_elements():
    """Refractory (tungsten) melts above mid (iron) above soft (zinc) — the real ordering."""
    def tm(eid):
        lat = from_element(elements.get(eid), shape=(36, 36)).lattice
        return thermal.melting_point(lat, n_temps=9, bracket=(0.75, 1.25),
                                     burn_in=60, n_samples=30)
    w, fe, zn = tm("tungsten"), tm("iron"), tm("zinc")
    assert w > fe > zn, f"melting order wrong: tungsten={w:.2f} iron={fe:.2f} zinc={zn:.2f}"


# --- the gated stored property (material pipeline) -------------------------------------


def test_stored_melting_temperature_gating():
    """measure_properties stores T_m>0 for a connected solid, 0.0 for a dispersed structure."""
    iron = from_element(elements.get("iron"), shape=(40, 40))      # dense, percolating solid
    hydrogen = from_element(elements.get("hydrogen"), shape=(40, 40))  # low-fill, dispersed
    assert iron.properties["melting_temperature"] > 1.0
    assert iron.properties["largest_cluster_fraction"] >= MELT_GATE_FLOOR
    assert hydrogen.properties["melting_temperature"] == 0.0
    assert hydrogen.properties["largest_cluster_fraction"] < MELT_GATE_FLOOR


# --- pressure is live: density(P) and pressure-tuned percolation (M2 re-validated) -------


def test_pressure_tunes_density_and_drives_percolation():
    """In the fluid phase, raising P densifies the lattice gas through the percolation point."""
    lat = from_element(elements.get("iron"), shape=(48, 48)).lattice
    hot = 3.2  # above T_m -> compressible fluid (a solid below T_m is ~incompressible)

    def density_spans(P):
        st = thermal.sample_occupancy_ensemble(
            lat, Conditions(temperature=hot, pressure=P), burn_in=80, n_samples=40
        )
        return st.mean_density

    lo_rho = density_spans(-3.0)
    mid_rho = density_spans(0.0)
    hi_rho = density_spans(+5.0)
    # Monotone densification; symmetric (P=0) sits at half-filling.
    assert lo_rho < mid_rho < hi_rho
    assert abs(mid_rho - 0.5) < 0.05
    # Pressure drives the structure across the percolation threshold (M2 under thermal occupancy).
    assert lo_rho < 0.55 and hi_rho > 0.60


def test_high_pressure_occupancy_percolates_under_thermal_dynamics():
    """M2 under the new dynamics: a high-pressure equilibrium occupancy *spans*; low-pressure doesn't.

    Equilibrates the occupancy field itself (the lattice-gas kernel) at high vs low pressure and
    measures percolation on the resulting structure — the percolation transition (M2) survives
    and is now driven by the pressure dial.
    """
    from engine.lattice import checkerboard_colors, occupancy_sweep
    from engine.rng import SplitMix64

    lat = from_element(elements.get("iron"), shape=(48, 48)).lattice
    colors = checkerboard_colors(lat.shape)
    mu0 = thermal.symmetric_mu(lat)

    def equilibrium_spans(P, frac_seed):
        gen = SplitMix64(frac_seed).numpy_generator()
        occ = (np.indices(lat.shape).sum(0) % 2 == 0).astype(np.uint8)
        mu = mu0 + thermal.PRESSURE_TO_MU * P
        for _ in range(120):
            occ = occupancy_sweep(occ, lat.cohesion, colors,
                                  coupling=COHESION_J0, temperature=3.2, mu=mu, gen=gen)
        probe = Lattice(occupied=occ, atom_type=np.where(occ == 1, 1, 0).astype(np.int8),
                        spin=np.ones(lat.shape, np.int8))
        return percolation.percolates_any_axis(probe), float(occ.mean())

    # Average spanning over a few equilibria (single snapshots near p_c are noisy).
    lo = [equilibrium_spans(-4.0, s)[0] for s in range(5)]
    hi = [equilibrium_spans(+6.0, s)[0] for s in range(5)]
    assert np.mean(hi) > np.mean(lo)          # pressure drives the percolation onset
    assert np.mean(hi) > 0.5                   # high pressure -> reliably conducts
    assert np.mean(lo) < 0.5                   # low pressure -> reliably insulates


# --- determinism (spec §6) ------------------------------------------------------------


def test_occupancy_ensemble_is_deterministic():
    """Same structure + same conditions -> byte-identical occupancy observables."""
    lat = _uniform_lattice((24, 24), cohesion=1.1)
    a = thermal.sample_occupancy_ensemble(lat, Conditions(temperature=2.0), n_samples=20)
    b = thermal.sample_occupancy_ensemble(lat, Conditions(temperature=2.0), n_samples=20)
    assert a == b


def test_positional_order_extractor():
    """ψ = 1 for a perfect checkerboard crystal, ~0 for a positionally random arrangement."""
    from engine.properties import microstructure
    from engine.rng import SplitMix64

    shape = (32, 32)
    crystal = ((np.indices(shape).sum(0) % 2) == 0).astype(np.uint8)
    lat_x = Lattice(occupied=crystal, atom_type=crystal.astype(np.int8),
                    spin=np.ones(shape, np.int8))
    assert microstructure.positional_order(lat_x) > 0.99

    gen = SplitMix64(123).numpy_generator()
    rnd = (gen.random(shape) < 0.5).astype(np.uint8)
    lat_r = Lattice(occupied=rnd, atom_type=rnd.astype(np.int8), spin=np.ones(shape, np.int8))
    assert microstructure.positional_order(lat_r) < 0.15


def test_symmetric_mu_holds_half_filling():
    """At the symmetric mu, the lattice gas sits at density ½ at any temperature."""
    lat = _uniform_lattice((32, 32), cohesion=1.0)
    assert abs(thermal.symmetric_mu(lat) - 8.0) < 1e-9  # 2D, J0=1, c=1 -> 8
    for T in (1.0, 2.0, 3.5):
        st = thermal.sample_occupancy_ensemble(lat, Conditions(temperature=T),
                                                burn_in=60, n_samples=30)
        assert abs(st.mean_density - 0.5) < 0.03
