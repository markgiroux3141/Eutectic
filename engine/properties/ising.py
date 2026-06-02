"""Ising magnetism: the canonical emergent phenomenon (spec §5.5).

Magnetism is *measured* from the settled spin field, not assigned. The coupling that
produces order lives in the lattice's structure: :func:`engine.lattice.relax` couples
neighbouring spins by ``J0 * moment_i * moment_j``, so high-moment regions (iron-family)
spontaneously align while low-moment regions (copper) stay thermally disordered. This
module reads the resulting order out — and because relaxation and measurement share the
same ``moment`` field, the two agree (spec §5.5).

The emergent signature is a clean critical transition: a material magnetises only when
its high-moment cells form a *connected* ordered network. Sparse or disconnected magnetic
domains break symmetry independently and cancel; a percolating magnetic backbone gives a
coherent net moment. Verify the transition empirically in the explorer (``magnetism-sweep``
and the population ``distribution``).

All functions are pure ``Lattice -> float`` and independently testable (spec §5).
"""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice


def _effective_moment(lattice: Lattice) -> np.ndarray:
    """Per-cell magnetic moment, zeroed on empty cells (the measurement weight)."""
    occ = lattice.occupied == 1
    return np.asarray(lattice.moment, dtype=np.float64) * occ


def magnetization(lattice: Lattice) -> float:
    """Signed moment-weighted net magnetization ``M`` of the settled spins (spec §5.5).

    ``M = sum(moment * spin) / sum(moment)`` over occupied cells. Weighting by moment
    means non-magnetic filler (low-moment copper) neither contributes nor dilutes: a
    coherent iron magnet reads ``M ~ +/-1`` even when embedded in copper, while a material
    whose magnetic domains order in opposing directions reads ``M ~ 0``. In ``[-1, 1]``.
    """
    m = _effective_moment(lattice)
    total = float(m.sum())
    if total <= 0.0:
        return 0.0
    return float((m * lattice.spin).sum() / total)


def magnetism(lattice: Lattice) -> float:
    """Magnetism = ``abs(net magnetization)`` of the settled spin field (spec §5.5).

    The emergent order parameter: ~0 in the disordered phase, rising toward 1 once a
    connected high-moment network spontaneously aligns. In ``[0, 1]``.
    """
    return abs(magnetization(lattice))


def magnetic_fraction(lattice: Lattice) -> float:
    """Fraction of the lattice's total moment that carries appreciable magnetism.

    A companion diagnostic: how much of the material is *capable* of ordering (high-moment
    mass), independent of whether it actually aligned. Useful for separating "no magnetic
    material present" from "magnetic material present but disordered" in the explorer.
    """
    m = _effective_moment(lattice)
    denom = float(lattice.occupied.sum())
    if denom <= 0.0:
        return 0.0
    return float(m.sum() / denom)
