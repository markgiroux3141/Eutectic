"""Spectral / band gap -> conductor / semiconductor / insulator (spec §5.6, milestone M7a).

The electronic structure property. We build a **tight-binding Hamiltonian** off the lattice —
``-t`` on every periodic nearest-neighbour bond, the per-cell ``site_potential`` on the diagonal
— take its eigenvalues (``numpy.linalg.eigvalsh``, deterministic), and read the **band gap** at
the Fermi level. The gap is *measured* from the structure, never assigned (spec §1): a charge-
staggered rock-salt lattice (``site_potential`` alternating ``±Δ`` on bipartite sublattices) opens
a gap ``= 2Δ`` and is an **insulator**; a uniform lattice (``site_potential ≡ 0``) opens none and
is a **conductor**. Chemistry sets the stagger from species electronegativity (``Δ ∝ χ − χ̄``,
see :mod:`chemistry.crystal`), so the conductor/insulator split *emerges* from the bonding.

This module is **generic**: it reads only ``Lattice`` fields and knows nothing about chemistry.

**Honest detector (spec §4).** A raw HOMO–LUMO gap is *not* an honest metal/insulator test: on a
finite lattice a metal's discrete levels leave a gap that shrinks like ``1/N`` (level spacing), so
every material shows *some* gap. The honest detector is the **normalized gap**
``raw_gap / mean_level_spacing``: a true gap is ``N``-independent while the spacing ``~ bandwidth/N``,
so the ratio *grows* with ``N`` for an insulator and stays ``O(1)`` for a metal. Classification uses
the normalized gap to reject finite-size noise; the insulator/semiconductor split then uses the raw
gap's physical magnitude.

**Scope / caveats (carried, not buried — README "M7 findings").**
* **Ionic gaps only (M7a).** The 2D substrate delivers the *ionic* (on-site stagger) gap cleanly.
  Covalent semiconductors (Si/diamond) need bond-alternation (SSH) hopping or real 3D tetrahedral
  coordination — deferred as **M7b** (spec §8).
* **Brittle to disorder.** The gap collapses to zero at the *first* vacancy (a dangling site is a
  mid-gap state) — confirmed in the de-risk. This is physically correct (amorphous Si metallizes),
  and why M7 needed the dense (fill = 1.0) chemistry crystal. Any "gap closes with T" is therefore a
  vacancy-**brittleness crossover**, NOT a phase transition (no-fudge norm; see the explorer).
* **Absolute eV scale is uncalibrated** (one fixed ``Δ``-scale constant in chemistry), like every
  prior layer. The emergent claims are the gap's *existence*, its *t-independence*, its *N-scaling*,
  and the *ΔEN-ordered classification* — qualitative correctness + parameter-free emergence.
"""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice

# Fixed hopping scale. The *ionic* gap (= 2Δ) is provably independent of t (de-risk [2]); t only
# sets the bandwidth, hence the level spacing the gap is normalized against.
HOPPING_T: float = 1.0

# Stored-value spectral window. ``eigvalsh`` is O(N^3): ~3.2 s at the 64x64 production shape but
# ~40 ms at 24x24 (de-risk [6]). The ionic gap is N-independent, so a capped window gives the same
# gap far cheaper; the stored value is therefore honestly *coarse* (like the stored Curie/melting
# points), and the explorer measures the full spectrum on demand. A periodic crystal is
# translationally uniform, so a fixed corner crop is representative.
SPECTRAL_WINDOW: int = 24

# Classification thresholds (on the normalized detector + the raw magnitude). These are dials, not
# emergent — the absolute eV scale is uncalibrated; only the *ordering* and the conductor/insulator
# *split* are claimed (README "M7 findings"). normalized_gap below METAL_RATIO == finite-size noise.
METAL_RATIO: float = 3.0          # normalized gap below this -> the "gap" is just level spacing
SEMI_INSULATOR_CUT: float = 1.0   # raw gap (in units of t) splitting semiconductor / insulator

# On-site stagger tolerance: below this spread (over occupied cells) there is no ionic gap
# mechanism -> conductor. The cost gate in engine.material uses this to skip eigvalsh entirely.
_STAGGER_TOL: float = 1e-9


def _window(lattice: Lattice) -> tuple[np.ndarray, np.ndarray]:
    """Crop occupancy + site_potential to a corner window of side <= SPECTRAL_WINDOW (cost cap)."""
    sl = tuple(slice(0, min(s, SPECTRAL_WINDOW)) for s in lattice.shape)
    occ = np.asarray(lattice.occupied)[sl].astype(bool)
    pot = np.asarray(lattice.site_potential, dtype=np.float64)[sl]
    return occ, pot


def _hamiltonian(occupied: np.ndarray, site_potential: np.ndarray, t: float) -> np.ndarray:
    """Real-symmetric tight-binding H: ``-t`` on each periodic NN bond, ``site_potential`` diagonal.

    Sites are the occupied cells in fixed row-major order (determinism, spec §6). One ``+1`` roll
    per axis enumerates every periodic nearest-neighbour bond exactly once.
    """
    occ = occupied.astype(bool)
    n = int(occ.sum())
    H = np.zeros((n, n), dtype=np.float64)
    if n == 0:
        return H
    flat = np.full(occ.shape, -1, dtype=np.int64)
    flat[occ] = np.arange(n)
    H[np.arange(n), np.arange(n)] = site_potential[occ]
    for axis in range(occ.ndim):
        src = flat
        dst = np.roll(flat, -1, axis=axis)
        mask = (src >= 0) & (dst >= 0)
        s = src[mask]
        d = dst[mask]
        H[s, d] += -t
        H[d, s] += -t
    return H


def spectrum(lattice: Lattice, *, t: float = HOPPING_T) -> np.ndarray:
    """Sorted tight-binding eigenvalues over the (windowed) occupied sites. Deterministic."""
    occ, pot = _window(lattice)
    H = _hamiltonian(occ, pot, t)
    if H.shape[0] == 0:
        return np.zeros(0)
    return np.sort(np.linalg.eigvalsh(H))


def has_onsite_stagger(lattice: Lattice) -> bool:
    """Cheap test (no diagonalization): does the lattice carry a non-trivial on-site stagger?

    Used by the engine.material cost gate to skip ``eigvalsh`` for the uniform/metallic majority
    (whose ``site_potential`` is zero) — exactly as Curie/melting gate their sweeps. No stagger
    means no ionic-gap mechanism, hence ``band_gap = 0`` (conductor), the validated behaviour.
    """
    occ = np.asarray(lattice.occupied).astype(bool)
    if not occ.any():
        return False
    pot = np.asarray(lattice.site_potential, dtype=np.float64)[occ]
    return float(pot.max() - pot.min()) > _STAGGER_TOL


def raw_gap(lattice: Lattice, *, t: float = HOPPING_T) -> float:
    """HOMO–LUMO gap at half filling (lower half of the spectrum filled). Physical, uncalibrated."""
    e = spectrum(lattice, t=t)
    n = len(e)
    if n < 2:
        return 0.0
    return float(e[n // 2] - e[n // 2 - 1])


def mean_level_spacing(lattice: Lattice, *, t: float = HOPPING_T) -> float:
    e = spectrum(lattice, t=t)
    if len(e) < 2:
        return 0.0
    return float(np.mean(np.diff(e)))


def normalized_gap(lattice: Lattice, *, t: float = HOPPING_T) -> float:
    """The honest detector: ``raw_gap / mean_level_spacing`` (grows with N for an insulator)."""
    e = spectrum(lattice, t=t)
    n = len(e)
    if n < 2:
        return 0.0
    gap = e[n // 2] - e[n // 2 - 1]
    spacing = float(np.mean(np.diff(e)))
    if spacing <= 0.0:
        return 0.0
    return float(gap / spacing)


def band_gap(lattice: Lattice, *, t: float = HOPPING_T) -> float:
    """The stored band gap: the raw HOMO–LUMO gap, but **0 unless a real (N-robust) gap exists**.

    Returns the raw gap only when the normalized detector confirms it is a genuine gap (not finite-
    size level spacing); otherwise 0.0 (a metal). So a uniform metal reads exactly 0 and an ionic
    crystal reads ``2Δ`` — the value the conductor/insulator classification rests on.
    """
    e = spectrum(lattice, t=t)
    n = len(e)
    if n < 2:
        return 0.0
    gap = float(e[n // 2] - e[n // 2 - 1])
    spacing = float(np.mean(np.diff(e)))
    if spacing <= 0.0 or gap / spacing < METAL_RATIO:
        return 0.0
    return gap


def dos_at_fermi(lattice: Lattice, *, t: float = HOPPING_T, broadening: float | None = None) -> float:
    """Broadened density of states at the Fermi level (secondary detector): ~0 in a gap, finite in a metal.

    Gaussian-broadened count of levels at ``E_F`` (the half-filling midpoint), per site. The
    broadening defaults to a few mean level spacings so a metal registers finite DOS while a gapped
    insulator registers ~0. (Distinct from the percolation axis: "gapped" is not "disconnected" —
    on a dense crystal the two are clean, but keep them conceptually separate, spec §4.)
    """
    e = spectrum(lattice, t=t)
    n = len(e)
    if n < 2:
        return 0.0
    e_f = 0.5 * (e[n // 2 - 1] + e[n // 2])
    spacing = float(np.mean(np.diff(e)))
    w = broadening if broadening is not None else max(3.0 * spacing, 1e-9)
    weights = np.exp(-0.5 * ((e - e_f) / w) ** 2) / (w * np.sqrt(2.0 * np.pi))
    return float(weights.sum() / n)


def classify(lattice: Lattice, *, t: float = HOPPING_T) -> str:
    """``"conductor"`` / ``"semiconductor"`` / ``"insulator"`` from the spectrum (spec §5.6).

    Conductor when the normalized gap is finite-size noise (``< METAL_RATIO``); otherwise the raw
    gap's magnitude splits semiconductor (small) from insulator (large). The split *emerges* from
    chemistry: ``Δ ∝ ΔEN``, so a large-ΔEN ionic crystal (NaCl) is an insulator and a metal (Δ=0) a
    conductor. Genuine covalent semiconductors await M7b (3D).
    """
    e = spectrum(lattice, t=t)
    n = len(e)
    if n < 2:
        return "conductor"
    gap = float(e[n // 2] - e[n // 2 - 1])
    spacing = float(np.mean(np.diff(e)))
    if spacing <= 0.0 or gap / spacing < METAL_RATIO:
        return "conductor"
    return "semiconductor" if gap < SEMI_INSULATOR_CUT * t else "insulator"


def measure(lattice: Lattice) -> dict[str, float]:
    """Spectral properties for the stored dict: the band gap (others live in the explorer/tests).

    Cost-gated by :func:`has_onsite_stagger` so the uniform/metallic majority pay no ``eigvalsh``:
    no stagger -> no ionic gap -> ``band_gap = 0`` (conductor). Only staggered (ionic) crystals
    diagonalize, and only over the capped :data:`SPECTRAL_WINDOW`.
    """
    if not has_onsite_stagger(lattice):
        return {"band_gap": 0.0}
    return {"band_gap": band_gap(lattice)}
