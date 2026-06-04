"""Magnetic microstructure: the readouts that reveal what a *process* did (process layer).

The Step-0 de-risk found that net magnetization ``|M|`` at zero field is a *poor* readout of
cooling history (at ``H=0`` the 2D Ising picks a random domain sign and the net cancels).
The cooling-rate signal lives in the **microstructure** instead — how the ordered domains are
arranged — which these extractors measure directly from the settled spin field:

* :func:`domain_fraction` — size of the largest single-orientation domain (slow anneal grows
  one big domain; quench freezes many small ones).
* :func:`domain_wall_density` — fraction of same-material bonds that are broken (anti-aligned)
  — the density of domain walls (quench high, anneal low).

Both are pure ``Lattice -> float`` measurements of the structure, never assigned (spec §1).
They complement :func:`engine.properties.ising.magnetism` (the *net* moment / remanence): a
material can have high domain order yet ~0 net moment if its domains oppose.
"""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice
from . import percolation


def domain_fraction(lattice: Lattice) -> float:
    """Fraction of occupied cells in the largest single-orientation (same-spin) domain.

    The microstructure order parameter: ~1 for a single-domain (well-annealed or
    field-cooled) magnet, small for a quench-frozen patchwork of competing domains. In
    ``[0, 1]``.
    """
    occ = lattice.occupied == 1
    denom = int(occ.sum())
    if denom == 0:
        return 0.0
    best = 0
    for s in (1, -1):
        labels, n = percolation.label_clusters(occ & (lattice.spin == s))
        if n:
            counts = np.bincount(labels.reshape(-1))
            counts[0] = 0  # ignore background
            best = max(best, int(counts.max()))
    return best / denom


def domain_wall_density(lattice: Lattice) -> float:
    """Fraction of occupied-occupied face bonds that are anti-aligned (domain walls).

    Counts physical (non-wrapping) nearest-neighbour bonds between two occupied cells and
    returns the share whose spins disagree. ~0 for a fully ordered single domain; rises with
    the frozen-in disorder a fast quench leaves behind. In ``[0, 1]``.
    """
    occ = lattice.occupied == 1
    spin = lattice.spin
    nd = lattice.dim
    walls = 0
    bonds = 0
    for axis in range(nd):
        lo = [slice(None)] * nd
        hi = [slice(None)] * nd
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        both = occ[tuple(lo)] & occ[tuple(hi)]
        disagree = spin[tuple(lo)] != spin[tuple(hi)]
        bonds += int(both.sum())
        walls += int((both & disagree).sum())
    return walls / bonds if bonds else 0.0


def positional_order(lattice: Lattice) -> float:
    """Staggered (sublattice) density of the occupancy field — the **melting order parameter** (M5).

    ``ψ = |(1/N)·Σ_i (−1)^(Σ coords)·(2·occupied_i − 1)|`` over the whole lattice: +1 for a
    perfect checkerboard crystal (one sublattice full, the other empty), ~0 for a positionally
    disordered (molten/random) arrangement. This is the order parameter whose collapse *is*
    melting — measured on the symmetry-broken occupancy ensemble it collapses at ``T_m`` while
    the mean density stays at ½ (see :func:`engine.thermal.sample_occupancy_ensemble`). The
    spin twin is :func:`magnetism`; this measures *where the atoms sit*, not *which way they
    point*.

    NB it is a **global** order parameter: like net ``|M|`` for spins, two opposite anti-phase
    crystalline domains cancel in ψ, so it is *not* a good readout of a crystallising process
    started from disorder (the cooling-rate → grain-size signal is weak here — non-conserved
    checkerboard order heals fast; the robust process readout is **density** under a pressure
    schedule, see the process layer). Pure ``Lattice -> float``, never assigned (spec §1). In
    ``[0, 1]``.
    """
    n = (lattice.occupied == 1).astype(np.float64)
    sign = (np.indices(lattice.shape).sum(axis=0) % 2 * 2 - 1).astype(np.float64)
    return abs(float((sign * (2.0 * n - 1.0)).mean()))
