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
    EXCHANGE_J0,
    Lattice,
    _neighbor_sum,
    checkerboard_colors,
    metropolis_sweep,
)
from .rng import SplitMix64, mix

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
