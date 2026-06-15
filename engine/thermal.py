"""The thermal-ensemble measurement engine (docs §2, §4).

This is the M4 keystone: it turns "properties measured once on a frozen snapshot" into
"properties measured as **ensemble observables of a structure at conditions**". Given a
material's structure (its lattice geometry + ``moment`` couplings) and a
:class:`~engine.conditions.Conditions`, it runs the spin Hamiltonian's equilibrium ensemble
(Metropolis at temperature ``T`` and field ``H``) and reads observables out:

* **magnetization ``⟨|M|⟩``** — the order parameter at ``T`` (moment-weighted, as in
  :mod:`engine.properties.ising`).
* **heat capacity ``C = Var(E)/(N·T²)``** — energy fluctuations of the ensemble. This is the
  *universal transition detector* (docs §4): it **peaks at every phase transition**, so we
  do not hand-define the Curie point — the peak *is* it. The keystone validation is that
  ``C(T)`` peaks exactly where the order parameter ``⟨|M|⟩`` collapses, and (for a plain
  unit-moment lattice) at the textbook 2D-Ising ``Tc = 2/ln(1+√2) ≈ 2.269`` with no free
  parameters.

The Hamiltonian is the same structural one the lattice settles under (spec §5.5):

    E = -(J0/2)·Σ_i moment_i·spin_i·h_i  -  H·Σ_i moment_i·spin_i,    h_i = Σ_{j∈nbr} m_j s_j

(the ``1/2`` de-double-counts each bond). The spin updates reuse
:func:`engine.lattice.metropolis_sweep`, so settling (M3) and ensemble measurement (M4)
share one deterministic kernel.

Determinism (spec §6): every measurement seeds one :class:`~engine.rng.SplitMix64` from the
structure's signature + the *quantized* conditions, so ``measure(structure, conditions)`` is
a pure deterministic function. No global RNG, fixed sweep counts, fixed reduction order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditions import STANDARD, Conditions
from .lattice import (
    COHESION_J0,
    EXCHANGE_J0,
    Lattice,
    SC_J0,
    SC_PROPOSAL_WINDOW,
    _LATTICE_GAS_FACTOR,
    _neighbor_sum,
    checkerboard_colors,
    metropolis_sweep,
    occupancy_sweep,
    xy_sweep,
)
from .rng import SplitMix64, hash_array, mix

# --- ensemble sampling defaults -------------------------------------------------------
# A "measurement" = burn-in sweeps to reach equilibrium, then n_samples observations each
# separated by sample_every sweeps (decorrelation). Heat capacity is a *variance*, so it is
# noisier than the order parameter; these defaults resolve the C(T) peak cleanly on the
# prototype lattices. The explorer can raise them for publication-quality curves; the stored
# per-material Tc uses the leaner :data:`STORED_*` values below.
DEFAULT_BURN_IN: int = 60
DEFAULT_SAMPLES: int = 40
DEFAULT_SAMPLE_EVERY: int = 2

# Salt so the ensemble RNG stream never collides with merge/relax streams (spec §6).
_ENSEMBLE_SALT: int = 0x54  # 'T'
# A distinct salt for the occupancy (melting) ensemble so its stream never collides with the
# spin ensemble's.
_OCC_SALT: int = 0x4F       # 'O'
# And one for the XY phase-coherence (superconductivity) ensemble (M6).
_PHASE_SALT: int = 0x50     # 'P'

# The 2D-XY universal jump: at the BKT transition the helicity modulus satisfies
# Υ(T_BKT) = (2/π)·T_BKT. T_BKT is located as the crossing of Υ(T) with this line — NOT the
# heat-capacity peak, which for the XY model sits *above* T_BKT (the C-peak detector that
# locates the Curie/melting points does not apply to a BKT transition; see docs / M6 findings).
_BKT_UNIVERSAL_SLOPE: float = 2.0 / np.pi

# Pressure → chemical-potential gain (M5). ``Conditions.pressure`` shifts μ above its
# particle-hole-symmetric (half-filling) value, in the occupancy energy's own units:
# positive P raises μ → favours occupancy → higher equilibrium density (compression). At the
# standard P=0 the ensemble sits at the symmetric μ (density ½), where melting is cleanest.
PRESSURE_TO_MU: float = 1.0


@dataclass(frozen=True)
class EnsembleStats:
    """Observables averaged over one equilibrium ensemble at fixed conditions."""

    conditions: Conditions
    mean_energy: float        # ⟨E⟩
    energy_var: float         # Var(E) = ⟨E²⟩ - ⟨E⟩²
    mean_abs_mag: float       # ⟨|M|⟩, the order parameter at T (in [0, 1])
    mean_mag: float           # ⟨M⟩ (signed; ~0 at H=0 by symmetry, nonzero under a field)
    heat_capacity: float      # C = Var(E)/(N·T²), the transition detector (per active cell)
    n_active: int             # number of occupied (coupling) cells, the normalisation N


def energy(lattice: Lattice, spin: np.ndarray, *, coupling: float = EXCHANGE_J0,
           field: float = 0.0) -> float:
    """Total Hamiltonian energy of ``spin`` on ``lattice``'s structure (see module docstring).

    Pure reduction in fixed order (numpy sum) for determinism. ``spin`` is passed
    explicitly so the sampler can evaluate it on its evolving configuration without
    rebuilding a Lattice each step.
    """
    occ = lattice.occupied.astype(bool)
    m = np.asarray(lattice.moment, dtype=np.float64) * occ
    s = np.asarray(spin, dtype=np.float64)
    h = _neighbor_sum(m * s)
    bond = -0.5 * coupling * float((m * s * h).sum())
    field_e = -field * float((m * s).sum())
    return bond + field_e


def _measurement_seed(lattice: Lattice, conditions: Conditions, salt: int) -> int:
    """Deterministic seed for an ensemble: structure signature + quantized conditions."""
    t, p, h = conditions.seed_key()
    return mix(lattice.structural_signature(), t, p, h, salt, _ENSEMBLE_SALT)


def sample_ensemble(
    lattice: Lattice,
    conditions: Conditions = STANDARD,
    *,
    seed: int | None = None,
    burn_in: int = DEFAULT_BURN_IN,
    n_samples: int = DEFAULT_SAMPLES,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    coupling: float = EXCHANGE_J0,
) -> EnsembleStats:
    """Measure equilibrium observables of ``lattice``'s structure at ``conditions``.

    Starts from an aligned spin field (symmetry broken; matches the order-parameter
    convention in :func:`engine.properties.ising.magnetism`), burns in to equilibrium at
    ``conditions.temperature`` and field ``conditions.field``, then collects ``n_samples``
    observations of energy and magnetization, each ``sample_every`` sweeps apart.

    Deterministic: if ``seed`` is omitted it is derived from the structure + quantized
    conditions, so repeated measurement of the same material at the same conditions is
    byte-identical (spec §6).
    """
    T = conditions.temperature
    H = conditions.field
    occ = lattice.occupied.astype(bool)
    m = np.asarray(lattice.moment, dtype=np.float64) * occ
    n_active = int(occ.sum())
    if seed is None:
        seed = _measurement_seed(lattice, conditions, salt=0)
    gen = SplitMix64(seed).numpy_generator()
    colors = checkerboard_colors(lattice.shape)

    spin = np.ones(lattice.shape, dtype=np.int8)  # aligned start (symmetry broken)

    for _ in range(burn_in):
        spin = metropolis_sweep(spin, m, occ, colors,
                                coupling=coupling, temperature=T, field=H, gen=gen)

    if n_active == 0:
        return EnsembleStats(conditions, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    total_moment = float(m.sum())
    energies = np.empty(n_samples, dtype=np.float64)
    mags = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        for _ in range(sample_every):
            spin = metropolis_sweep(spin, m, occ, colors,
                                    coupling=coupling, temperature=T, field=H, gen=gen)
        energies[k] = energy(lattice, spin, coupling=coupling, field=H)
        mags[k] = float((m * spin).sum()) / total_moment

    e_var = float(energies.var())
    return EnsembleStats(
        conditions=conditions,
        mean_energy=float(energies.mean()),
        energy_var=e_var,
        mean_abs_mag=float(np.abs(mags).mean()),
        mean_mag=float(mags.mean()),
        heat_capacity=e_var / (n_active * T * T),
        n_active=n_active,
    )


# --- temperature sweeps + Curie-point extraction --------------------------------------


def temperature_sweep(
    lattice: Lattice,
    temperatures,
    *,
    field: float = 0.0,
    burn_in: int = DEFAULT_BURN_IN,
    n_samples: int = DEFAULT_SAMPLES,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    coupling: float = EXCHANGE_J0,
) -> list[EnsembleStats]:
    """Measure the ensemble at each temperature (fixed field). Returns one stat per point.

    The basis for the ``C(T)`` / ``⟨|M|⟩(T)`` curves the explorer plots and the keystone
    tests assert on. Each point is independently seeded (from the quantized conditions), so
    the whole sweep is deterministic and order-independent.
    """
    out: list[EnsembleStats] = []
    for T in temperatures:
        cond = Conditions(temperature=float(T), field=field)
        out.append(sample_ensemble(
            lattice, cond, burn_in=burn_in, n_samples=n_samples,
            sample_every=sample_every, coupling=coupling,
        ))
    return out


def curie_temperature(
    lattice: Lattice,
    *,
    t_lo: float = 1.0,
    t_hi: float = 4.0,
    n_temps: int = 13,
    order_floor: float = 0.30,
    burn_in: int = DEFAULT_BURN_IN,
    n_samples: int = DEFAULT_SAMPLES,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    coupling: float = EXCHANGE_J0,
    return_curve: bool = False,
):
    """Curie temperature ``Tc`` = the ``C(T)`` peak (the universal transition detector).

    Sweeps ``[t_lo, t_hi]`` and returns the temperature of maximum heat capacity — the
    transition the order parameter collapses through. ``Tc`` is located from the *energy
    fluctuations*, not from a magnetization threshold, so it is not a disguised dial (docs
    §6): for a plain unit-moment lattice this peak lands on the textbook 2D-Ising point.

    Returns ``None`` when the material never orders over the range — ``max ⟨|M|⟩`` below
    ``order_floor`` means there is no ferromagnetic phase to lose, so no Curie point. (That
    floor is an "is there *any* order" gate, not a transition location — the transition is
    still the ``C`` peak.) With ``return_curve=True`` returns ``(Tc, sweep)``.
    """
    temps = np.linspace(t_lo, t_hi, n_temps)
    sweep = temperature_sweep(
        lattice, temps, burn_in=burn_in, n_samples=n_samples,
        sample_every=sample_every, coupling=coupling,
    )
    max_order = max(s.mean_abs_mag for s in sweep)
    if max_order < order_floor:
        tc = None
    else:
        peak = max(range(len(sweep)), key=lambda i: sweep[i].heat_capacity)
        tc = float(temps[peak])
    return (tc, sweep) if return_curve else tc


# === M5: the occupancy ensemble — melting as an order-disorder transition =============
# The positional twin of the spin ensemble above. ``occupied`` is now a thermal degree of
# freedom (a non-conserved repulsive lattice gas, :func:`engine.lattice.occupancy_sweep`);
# its order-disorder transition *is* crystalline melting. We measure the same trio that
# proved the Curie point — an order parameter, the heat capacity, and (at the symmetric μ)
# the textbook 2D point — but for *positional* order: a **staggered density** that collapses
# at ``T_m`` while the mean density stays pinned at ½ (order lost at fixed density → melting,
# not sublimation). Determinism: seeded from structure + cohesion hash + quantized conditions.

# Exact 2D-Ising critical point — the parameter-free anchor both transitions land on.
ISING_TC_2D: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


@dataclass(frozen=True)
class OccupancyStats:
    """Observables of the occupancy (lattice-gas) ensemble at fixed conditions (M5)."""

    conditions: Conditions
    mean_energy: float        # ⟨E⟩ of the lattice-gas Hamiltonian
    energy_var: float         # Var(E)
    mean_density: float       # ⟨ρ⟩ = ⟨n⟩, the fill fraction at these conditions
    staggered_order: float    # ⟨|ψ|⟩, sublattice-occupancy order parameter (crystallinity)
    heat_capacity: float      # C = Var(E)/(N·T²), the melting detector
    n_cells: int              # lattice size, the normalisation N


def _coupling_scale(lattice: Lattice, coupling: float) -> float:
    """Representative bond repulsion scale ⟨ε⟩ = 4·coupling·⟨cohesion²⟩ (per-bond average)."""
    coh = np.asarray(lattice.cohesion, dtype=np.float64)
    return _LATTICE_GAS_FACTOR * coupling * float((coh * coh).mean())


def symmetric_mu(lattice: Lattice, coupling: float = COHESION_J0) -> float:
    """The particle-hole-symmetric chemical potential μ that holds ⟨ρ⟩ = ½ (M5).

    At this μ the repulsive lattice gas keeps mean density at one-half across the melting
    transition, so the staggered order can collapse at *fixed* density. Derived from the
    lattice-gas↔Ising mapping: μ_sym = (z/2)·⟨ε⟩ with coordination ``z = 2·dim`` and ⟨ε⟩ the
    mean bond repulsion (for a uniform-cohesion 2D lattice this is 8·coupling·cohesion²).
    """
    z = 2 * lattice.dim
    return 0.5 * z * _coupling_scale(lattice, coupling)


def chemical_potential(
    lattice: Lattice, conditions: Conditions, coupling: float = COHESION_J0
) -> float:
    """μ at these conditions: the symmetric (half-filling) value plus the pressure offset.

    Activates ``Conditions.pressure`` (M5): P>0 raises μ → denser, P<0 → more porous. P=0
    (standard) sits at :func:`symmetric_mu`, the clean melting point.
    """
    return symmetric_mu(lattice, coupling) + PRESSURE_TO_MU * conditions.pressure


def occupancy_energy(lattice: Lattice, n: np.ndarray, *, coupling: float, mu: float) -> float:
    """Lattice-gas Hamiltonian energy of occupancy ``n`` (see :func:`occupancy_sweep`).

    ``E = Σ_⟨ij⟩ ε_ij n_i n_j − μ Σ n_i`` with ``ε_ij = 4·coupling·coh_i·coh_j``; the bond
    term is halved to de-double-count each bond. Pure fixed-order reduction (determinism).
    """
    coh = np.asarray(lattice.cohesion, dtype=np.float64)
    nn = np.asarray(n, dtype=np.float64)
    h = _neighbor_sum(coh * nn)
    bond = 0.5 * _LATTICE_GAS_FACTOR * coupling * float((coh * nn * h).sum())
    return bond - mu * float(nn.sum())


def _staggered_sign(shape: tuple[int, ...]) -> np.ndarray:
    """(-1)^(sum of coordinates): +1/−1 on the two sublattices of the bipartite lattice."""
    return (np.indices(shape).sum(axis=0) % 2 * 2 - 1).astype(np.float64)


def _occupancy_seed(lattice: Lattice, conditions: Conditions, salt: int) -> int:
    """Seed for an occupancy ensemble: structure sig + cohesion hash + quantized conditions.

    The cohesion hash is folded in explicitly because cohesion is *not* in
    ``structural_signature`` (it carries no independent entropy for combination seeding), yet
    it fully determines the melting behaviour — so the *measurement* seed must reflect it.
    """
    t, p, h = conditions.seed_key()
    return mix(lattice.structural_signature(), hash_array(lattice.cohesion),
               t, p, h, salt, _OCC_SALT)


def sample_occupancy_ensemble(
    lattice: Lattice,
    conditions: Conditions = STANDARD,
    *,
    seed: int | None = None,
    coupling: float = COHESION_J0,
    burn_in: int = DEFAULT_BURN_IN,
    n_samples: int = DEFAULT_SAMPLES,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
) -> OccupancyStats:
    """Measure equilibrium occupancy observables of ``lattice``'s bond network at ``conditions``.

    Starts from the perfect checkerboard crystal (symmetry broken, the positional analogue of
    the aligned-spin start), burns in at ``conditions.temperature`` and the conditions' μ
    (:func:`chemical_potential`), then samples energy, density, and the staggered order
    parameter ``ψ = (1/N)·Σ (−1)^(Σcoords)·(2n−1)``. Deterministic (spec §6).
    """
    T = conditions.temperature
    mu = chemical_potential(lattice, conditions, coupling)
    shape = lattice.shape
    n_cells = int(np.prod(shape))
    if seed is None:
        seed = _occupancy_seed(lattice, conditions, salt=0)
    gen = SplitMix64(seed).numpy_generator()
    colors = checkerboard_colors(shape)
    sign = _staggered_sign(shape)
    cohesion = lattice.cohesion

    # perfect checkerboard crystal start (one sublattice full) — symmetry broken.
    occ = ((np.indices(shape).sum(axis=0) % 2) == 0).astype(np.uint8)

    for _ in range(burn_in):
        occ = occupancy_sweep(occ, cohesion, colors,
                              coupling=coupling, temperature=T, mu=mu, gen=gen)

    energies = np.empty(n_samples, dtype=np.float64)
    psis = np.empty(n_samples, dtype=np.float64)
    rhos = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        for _ in range(sample_every):
            occ = occupancy_sweep(occ, cohesion, colors,
                                  coupling=coupling, temperature=T, mu=mu, gen=gen)
        nn = occ.astype(np.float64)
        energies[k] = occupancy_energy(lattice, occ, coupling=coupling, mu=mu)
        psis[k] = abs(float((sign * (2.0 * nn - 1.0)).sum()) / n_cells)
        rhos[k] = float(nn.mean())

    e_var = float(energies.var())
    return OccupancyStats(
        conditions=conditions,
        mean_energy=float(energies.mean()),
        energy_var=e_var,
        mean_density=float(rhos.mean()),
        staggered_order=float(psis.mean()),
        heat_capacity=e_var / (n_cells * T * T),
        n_cells=n_cells,
    )


def occupancy_temperature_sweep(
    lattice: Lattice,
    temperatures,
    *,
    pressure: float = 0.0,
    coupling: float = COHESION_J0,
    burn_in: int = DEFAULT_BURN_IN,
    n_samples: int = DEFAULT_SAMPLES,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
) -> list[OccupancyStats]:
    """Measure the occupancy ensemble at each temperature (fixed pressure). One stat / point."""
    out: list[OccupancyStats] = []
    for T in temperatures:
        cond = Conditions(temperature=float(T), pressure=pressure)
        out.append(sample_occupancy_ensemble(
            lattice, cond, coupling=coupling,
            burn_in=burn_in, n_samples=n_samples, sample_every=sample_every,
        ))
    return out


def melting_point(
    lattice: Lattice,
    *,
    coupling: float = COHESION_J0,
    pressure: float = 0.0,
    n_temps: int = 11,
    bracket: tuple[float, float] = (0.45, 1.45),
    order_floor: float = 0.30,
    burn_in: int = DEFAULT_BURN_IN,
    n_samples: int = DEFAULT_SAMPLES,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    return_curve: bool = False,
):
    """Melting temperature ``T_m`` = the occupancy ``C(T)`` peak (the universal detector, M5).

    Brackets the sweep around the analytic order-disorder point ``T_m ≈ 2.269·coupling·⟨coh²⟩``
    (the lattice-gas↔2D-Ising value) — the prediction only sets *where to look*; the returned
    value is the *measured* heat-capacity peak, the temperature the staggered order collapses
    through. Returns ``None`` if the lattice never develops sublattice order over the range
    (``max ψ < order_floor`` — e.g. a degenerate all-empty structure), the "is there a crystal
    to melt" gate. With ``return_curve=True`` returns ``(T_m, sweep)``.
    """
    coh = np.asarray(lattice.cohesion, dtype=np.float64)
    # Analytic order-disorder point for the (representative) uniform bond: 2.269·coupling·⟨c²⟩.
    predicted = ISING_TC_2D * coupling * float((coh * coh).mean())
    lo, hi = bracket[0] * predicted, bracket[1] * predicted
    temps = np.linspace(lo, hi, n_temps)
    sweep = occupancy_temperature_sweep(
        lattice, temps, pressure=pressure, coupling=coupling,
        burn_in=burn_in, n_samples=n_samples, sample_every=sample_every,
    )
    max_order = max(s.staggered_order for s in sweep)
    if max_order < order_floor:
        tm = None
    else:
        peak = max(range(len(sweep)), key=lambda i: sweep[i].heat_capacity)
        tm = float(temps[peak])
    return (tm, sweep) if return_curve else tm


# === M6: the XY phase-coherence ensemble — honest superconductivity with a real Tc =====
# Superconductivity *is* phase coherence of the order parameter. We put an XY model (a phase
# θ per conducting cell, -J·cos(θ_i−θ_j) across the conducting backbone) on the structure and
# measure the **helicity modulus** Υ(T) — the superconducting (phase) stiffness. In 2D this
# orders via a BKT transition, so (unlike Curie/melting) there is no true long-range order and
# the heat-capacity peak does NOT mark Tc; the transition is where Υ(T) crosses the universal
# line Υ = (2/π)·T. Tc emerges from the backbone's rigidity — a redundant solid backbone
# coheres up to the textbook 0.893·J0, a thin near-percolation filament barely coheres — so the
# k-edge-connectivity work becomes the *coupling input*, not the label (it retires the static
# proxy). Determinism: seeded from structure signature + quantized conditions (the conducting
# mask is `occupied`, already in the signature).


@dataclass(frozen=True)
class XYStats:
    """Observables of the XY phase-coherence (superconductivity) ensemble at fixed conditions."""

    conditions: Conditions
    helicity_modulus: float   # Υ, the superconducting phase stiffness (the order parameter)
    mean_energy: float        # ⟨E⟩ of the XY Hamiltonian on the conducting backbone
    energy_var: float         # Var(E)
    heat_capacity: float      # C = Var(E)/(N·T²) — peaks ABOVE Tc for BKT (do not use as Tc)
    n_conducting: int         # number of conducting cells, the normalisation N


def _phase_seed(lattice: Lattice, conditions: Conditions, salt: int) -> int:
    t, p, h = conditions.seed_key()
    return mix(lattice.structural_signature(), t, p, h, salt, _PHASE_SALT)


def sample_xy_ensemble(
    lattice: Lattice,
    conditions: Conditions = STANDARD,
    *,
    seed: int | None = None,
    coupling: float = SC_J0,
    window: float = SC_PROPOSAL_WINDOW,
    burn_in: int = 200,
    n_samples: int = 80,
    sample_every: int = 2,
) -> XYStats:
    """Measure the equilibrium phase stiffness (helicity modulus) of ``lattice``'s backbone.

    Runs the XY ensemble on the conducting cells at ``conditions.temperature`` from an aligned
    (θ≡0) start, then samples the helicity modulus, averaged over the two lattice directions:

        Υ_d = (1/N)·⟨Σ_d cos(Δθ_d)·b_d⟩ − (1/(N·T))·Var(Σ_d sin(Δθ_d)·b_d)

    where ``b_d`` is 1 on a bond whose *both* endpoints conduct, ``Δθ_d`` the phase difference
    along direction ``d``, and ``N`` the number of conducting cells. The fluctuation term is the
    phase-winding stiffness; a rigid (coherent) backbone keeps Υ high, a floppy one drives it to
    0. Deterministic in ``(structure, conditions)`` (spec §6).
    """
    from .properties import percolation  # local import avoids a module cycle at import time

    T = conditions.temperature
    cond = percolation.conducting_mask(lattice)  # charge backbone (occupied AND metallic, M6b)
    cond_f = cond.astype(np.float64)
    n_cond = int(cond.sum())
    shape = lattice.shape
    if seed is None:
        seed = _phase_seed(lattice, conditions, salt=0)
    gen = SplitMix64(seed).numpy_generator()
    colors = checkerboard_colors(shape)

    if n_cond == 0:
        return XYStats(conditions, 0.0, 0.0, 0.0, 0.0, 0)

    # active-bond masks per direction (both endpoints conduct; non-wrapping handled by the
    # bond living "between i and i+1", periodic like the kernel's neighbour sum)
    bond = [cond_f * np.roll(cond_f, -1, axis=ax) for ax in range(lattice.dim)]

    theta = np.zeros(shape, dtype=np.float64)  # aligned start (coherent), symmetry broken
    for _ in range(burn_in):
        theta = xy_sweep(theta, cond, colors, coupling=coupling,
                         temperature=T, gen=gen, window=window)

    energies = np.empty(n_samples)
    cos_sums = np.empty(n_samples)             # Σ_d cos(Δθ) over active bonds, both dirs
    wind = [np.empty(n_samples) for _ in range(lattice.dim)]  # Σ sin(Δθ) per direction
    for k in range(n_samples):
        for _ in range(sample_every):
            theta = xy_sweep(theta, cond, colors, coupling=coupling,
                             temperature=T, gen=gen, window=window)
        cos_total = 0.0
        for ax in range(lattice.dim):
            d = theta - np.roll(theta, -1, axis=ax)
            cos_total += float((np.cos(d) * bond[ax]).sum())
            wind[ax][k] = float((np.sin(d) * bond[ax]).sum())
        cos_sums[k] = cos_total
        energies[k] = -coupling * cos_total

    # helicity modulus averaged over directions
    hel = 0.0
    for ax in range(lattice.dim):
        cos_term = coupling * (cos_sums.mean() / lattice.dim) / n_cond
        var_term = coupling * coupling * wind[ax].var() / (n_cond * T)
        hel += cos_term - var_term
    hel /= lattice.dim

    e_var = float(energies.var())
    return XYStats(
        conditions=conditions,
        helicity_modulus=float(hel),
        mean_energy=float(energies.mean()),
        energy_var=e_var,
        heat_capacity=e_var / (n_cond * T * T),
        n_conducting=n_cond,
    )


def xy_temperature_sweep(
    lattice: Lattice,
    temperatures,
    *,
    coupling: float = SC_J0,
    window: float = SC_PROPOSAL_WINDOW,
    burn_in: int = 200,
    n_samples: int = 80,
    sample_every: int = 2,
) -> list[XYStats]:
    """Measure the XY ensemble at each temperature. One :class:`XYStats` per point."""
    out: list[XYStats] = []
    for T in temperatures:
        cond = Conditions(temperature=float(T))
        out.append(sample_xy_ensemble(
            lattice, cond, coupling=coupling, window=window,
            burn_in=burn_in, n_samples=n_samples, sample_every=sample_every,
        ))
    return out


def _universal_crossing(temps, helicities) -> float | None:
    """Temperature where Υ(T) crosses the BKT universal line Υ = (2/π)·T (first down-crossing).

    Returns ``None`` if Υ is already below the line at the lowest temperature (no coherent
    phase even at low T → no superconductivity). Linear interpolation between bracketing points.
    """
    f = [h - _BKT_UNIVERSAL_SLOPE * T for h, T in zip(helicities, temps)]
    if f[0] < 0:
        return None  # never coherent in this range
    for i in range(len(f) - 1):
        if f[i] >= 0 >= f[i + 1]:
            t = f[i] / (f[i] - f[i + 1])
            return float(temps[i] + t * (temps[i + 1] - temps[i]))
    return None  # still coherent at the top of the range (Tc above it)


def superconducting_tc(
    lattice: Lattice,
    *,
    coupling: float = SC_J0,
    t_lo: float = 0.10,
    t_hi: float = 1.10,
    n_temps: int = 16,
    window: float = SC_PROPOSAL_WINDOW,
    burn_in: int = 200,
    n_samples: int = 80,
    sample_every: int = 2,
    return_curve: bool = False,
):
    """Superconducting transition temperature ``Tc`` = the BKT helicity-modulus crossing (M6).

    Sweeps temperature and returns where the phase stiffness Υ(T) crosses the universal line
    ``Υ = (2/π)·T`` — the BKT transition, the temperature below which the backbone is
    phase-coherent (superconducting). NOT the heat-capacity peak (which sits above Tc for BKT).
    Returns ``None`` when the backbone never coheres over the range (no superconductivity). For a
    fully-conducting lattice this lands on the textbook ``0.893·coupling`` (parameter-free); a
    diluted/tortuous backbone gives a *lower* Tc, so Tc emerges from structure. With
    ``return_curve=True`` returns ``(Tc, sweep)``.
    """
    temps = np.linspace(t_lo, t_hi, n_temps)
    sweep = xy_temperature_sweep(
        lattice, temps, coupling=coupling, window=window,
        burn_in=burn_in, n_samples=n_samples, sample_every=sample_every,
    )
    tc = _universal_crossing(temps, [s.helicity_modulus for s in sweep])
    return (tc, sweep) if return_curve else tc
