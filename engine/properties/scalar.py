"""Scalar blends: density and mass (spec §5.1).

These are the *legible* properties — the ones a player can roughly predict by eyeballing
the recipe. That predictability is intentional: it's the calm anchor against which the
threshold properties (percolation, magnetism, superconductivity) feel surprising and
rare. Every value is still measured from the lattice, never assigned.
"""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice


def mean_atomic_mass(lattice: Lattice) -> float:
    """Mass-weighted mean atomic mass over occupied cells (spec §5.1).

    For a root this is just the element's atomic mass; for a combination it's the blend
    of the parents' masses in proportion to how much of each survived the merge.
    """
    occ = lattice.occupied == 1
    if not occ.any():
        return 0.0
    return float(lattice.mass[occ].mean())


def density(lattice: Lattice) -> float:
    """Density = fill fraction x mean atomic mass of the occupied matter (spec §5.1).

    Equivalent to ``total mass / number of cells`` — a legible, monotone function of how
    full and how heavy the lattice is. Reported in atomic-mass units per cell.
    """
    return float(lattice.fill_fraction) * mean_atomic_mass(lattice)


def total_mass(lattice: Lattice) -> float:
    """Total atomic mass summed over the whole lattice."""
    return float(np.asarray(lattice.mass).sum())
