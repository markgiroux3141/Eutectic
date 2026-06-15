"""Percolation -> conductivity (spec §5.2).

The first *threshold* property and the project's core rarity mechanism. We label
connected clusters of conducting cells and ask whether one **spans** the lattice from one
face to the opposite face. Below a critical fill density there is no spanning cluster
(insulator); right around the critical point a spanning cluster appears suddenly
(conductor). That sharp transition — not a probability dial — is where rarity comes from
(spec §1, §5.2).

All functions are pure ``Lattice -> float|bool`` and independently testable (spec §5).
Connectivity is *face* connectivity (4-neighbour in 2D, 6-neighbour in 3D) — the standard
site-percolation convention, so the threshold matches the known p_c (~0.5927 for 2D
square-lattice site percolation).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..lattice import Lattice


# M6b: a cell carries *charge* iff it is occupied AND metallic (``metallicity`` above this
# threshold). Below it the atom is an insulator for charge — it still carries heat (phonons),
# which is what splits the two carriers and produces the diamond divergence. The default
# metallicity is 1.0, so synthetic/constructed lattices keep the M2 behaviour (charge = occupied).
METALLIC_THRESHOLD: float = 0.60


def conducting_mask(lattice: Lattice) -> np.ndarray:
    """Boolean mask of cells that carry **charge** (the electrical / superconducting backbone).

    A cell conducts charge iff it is occupied *and* metallic (``metallicity`` ≥
    :data:`METALLIC_THRESHOLD`, M6b). The single chokepoint so percolation, conductance and
    superconductivity all agree on "what conducts charge". Heat (phonons) uses
    :func:`solid_mask` instead — all occupied matter — so an electrical insulator can still be a
    fine heat conductor (the diamond divergence).
    """
    return (lattice.occupied == 1) & (np.asarray(lattice.metallicity) >= METALLIC_THRESHOLD)


def solid_mask(lattice: Lattice) -> np.ndarray:
    """Boolean mask of cells that are **matter** (occupied), regardless of metallicity.

    The *structural* / solid network: heat (phonons) flows through it, the largest cluster is
    the solidity that gates melting, and it is the matter-percolation order parameter. This is
    the M2 ``occupied`` mask, kept distinct from the (metallicity-gated) charge backbone.
    """
    return lattice.occupied == 1


def _face_connectivity(ndim: int) -> np.ndarray:
    """Structuring element for face connectivity (no diagonals)."""
    return ndimage.generate_binary_structure(ndim, 1)


def label_clusters(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label face-connected clusters of True cells. Returns (labels, n_clusters)."""
    labels, n = ndimage.label(mask, structure=_face_connectivity(mask.ndim))
    return labels, int(n)


def _spans_axis(labels: np.ndarray, axis: int) -> bool:
    """Does any single cluster touch both the low and high face along ``axis``?"""
    lo = np.take(labels, 0, axis=axis)
    hi = np.take(labels, -1, axis=axis)
    lo_ids = set(np.unique(lo[lo > 0]).tolist())
    hi_ids = set(np.unique(hi[hi > 0]).tolist())
    return len(lo_ids & hi_ids) > 0


def spans(lattice: Lattice, axis: int = 0) -> bool:
    """True iff a conducting cluster spans the lattice along ``axis`` (default top-bottom)."""
    labels, n = label_clusters(conducting_mask(lattice))
    if n == 0:
        return False
    return _spans_axis(labels, axis)


def percolates_any_axis(lattice: Lattice) -> bool:
    """True iff a conducting cluster spans along *any* axis.

    A material is a conductor if current can cross it in some direction; anisotropy
    (spans one axis but not another) is itself interesting and surfaced separately.
    """
    labels, n = label_clusters(conducting_mask(lattice))
    if n == 0:
        return False
    return any(_spans_axis(labels, ax) for ax in range(lattice.dim))


def conductivity_boolean(lattice: Lattice) -> bool:
    """Boolean conductivity: does it percolate along any axis (spec §5.2)."""
    return percolates_any_axis(lattice)


def spanning_fraction(lattice: Lattice, axis: int = 0) -> float:
    """Fraction of all cells belonging to the spanning cluster along ``axis``.

    A continuous companion to the boolean: 0 below threshold, rising sharply above it as
    the spanning cluster engulfs most of the solid mass. Useful for histograms that
    show the transition's shape (spec §7). Measured on the **solid** (matter) network so it is
    the structural percolation order parameter — independent of metallicity (M6b).
    """
    labels, n = label_clusters(solid_mask(lattice))
    if n == 0:
        return 0.0
    lo = np.take(labels, 0, axis=axis)
    hi = np.take(labels, -1, axis=axis)
    lo_ids = set(np.unique(lo[lo > 0]).tolist())
    hi_ids = set(np.unique(hi[hi > 0]).tolist())
    spanning_ids = lo_ids & hi_ids
    if not spanning_ids:
        return 0.0
    in_span = np.isin(labels, list(spanning_ids))
    return float(in_span.sum()) / float(labels.size)


def largest_cluster_fraction(lattice: Lattice) -> float:
    """Fraction of cells in the single largest conducting cluster (the order parameter).

    This is the classic percolation order parameter: near-zero in the disconnected
    (sparse) phase, jumping toward the fill fraction once a giant cluster forms. Measured on
    the **solid** (matter) network (M6b), so it is the material's solidity — what gates melting
    — independent of whether that matter is metallic.
    """
    labels, n = label_clusters(solid_mask(lattice))
    if n == 0:
        return 0.0
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0  # ignore background
    return float(counts.max()) / float(labels.size)
