"""Effective resistance + backbone redundancy (spec §5.3; the conducting-graph measurements).

Two measurements taken from the settled lattice's conducting structure:

* **effective resistance** (§5.3): treat the conducting cells as a unit-conductance
  resistor network and solve the graph Laplacian for the resistance between two opposite
  faces (ideal equipotential bus bars). ``conductivity = 1/resistance``. A chunky,
  redundant conductor has low resistance; a thin tortuous near-percolation path has high
  resistance.
* **edge-connectivity**: the max-flow / min-cut between the two faces — the number of
  edge-disjoint paths. A single filament has width 1; a solid slab has width ~ the
  cross-section. ``k``-edge-connectivity appears at a higher-order percolation threshold
  ``p_k`` that rises with ``k`` and exceeds the ordinary point ``p_c`` (k=1 reproduces site
  ``p_c ~ 0.593``). This is the **backbone-redundancy** of the conductor.

**Superconductivity moved to M6** (`engine.thermal`): it is no longer a static topological
flag here. Real superconductivity is *phase coherence* — modelled as an XY (BKT) ordering of
the conducting backbone with an emergent transition temperature ``superconducting_tc``. The
``edge_connectivity``/``bottleneck_fraction`` measured here are the **structural input** to
that (a redundant backbone is phase-stiff → higher Tc), not the label. The earlier static
``is_superconductor`` proxy (k-edge-connectivity at fixed conditions, no real Tc) is
**retired** — see README "M6 findings" and `superconductivity-status`.

Determinism (spec §6): nodes are ordered by flattened cell index (a fixed order), the
Laplacian solve uses a direct sparse solver (``spsolve``, SuperLU — deterministic, unlike
iterative methods), max-flow is a deterministic function of the fixed graph, and all
outputs are quantized by the caller. No RNG anywhere here.

All functions are pure ``Lattice -> float`` and independently testable (spec §5).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve

from ..lattice import Lattice
from . import percolation

# Numerical stand-in for "no spanning cluster" (infinite resistance). Kept finite so
# measured properties quantize and compare cleanly (spec §6.4); conductivity reads 0 here.
RESISTANCE_INF: float = 1.0e9


def _cross_section(shape: tuple[int, ...], axis: int) -> int:
    """Number of cells in one layer perpendicular to ``axis`` (the max bottleneck width)."""
    n = 1
    for ax, s in enumerate(shape):
        if ax != axis:
            n *= s
    return n


def _spanning_cluster_mask(lattice: Lattice, axis: int) -> np.ndarray:
    """Boolean mask of cells in a cluster that touches both faces along ``axis``.

    Only these cells carry current between the faces; everything else is a dead end and is
    excluded so the resistor network is exactly the conducting backbone.
    """
    mask = percolation.conducting_mask(lattice)
    labels, n = percolation.label_clusters(mask)
    if n == 0:
        return np.zeros(lattice.shape, dtype=bool)
    lo = np.take(labels, 0, axis=axis)
    hi = np.take(labels, -1, axis=axis)
    lo_ids = set(np.unique(lo[lo > 0]).tolist())
    hi_ids = set(np.unique(hi[hi > 0]).tolist())
    spanning = lo_ids & hi_ids
    if not spanning:
        return np.zeros(lattice.shape, dtype=bool)
    return np.isin(labels, list(spanning))


def _backbone_graph(mask: np.ndarray, axis: int):
    """Build the node indexing + face-adjacency edges of the spanning backbone.

    Returns ``(node_id, edges, low_nodes, high_nodes)`` where ``node_id`` maps each masked
    cell (in flattened order) to a contiguous node index, ``edges`` is a list of
    ``(i, j)`` face-adjacency pairs (i < j), and ``low``/``high_nodes`` are the node
    indices touching the two faces along ``axis``. Node order follows flattened cell index
    so the graph is identical across runs (spec §6.2).
    """
    flat_idx = np.flatnonzero(mask.reshape(-1))  # sorted -> fixed node order
    node_id = -np.ones(mask.size, dtype=np.int64)
    node_id[flat_idx] = np.arange(flat_idx.size)
    node_id_grid = node_id.reshape(mask.shape)

    edges: list[tuple[int, int]] = []
    for ax in range(mask.ndim):
        # Neighbour pairs along axis ax: cell k and cell k+1 (no wrap — physical faces).
        a = node_id_grid.take(range(0, mask.shape[ax] - 1), axis=ax)
        b = node_id_grid.take(range(1, mask.shape[ax]), axis=ax)
        both = (a >= 0) & (b >= 0)
        for i, j in zip(a[both].tolist(), b[both].tolist()):
            edges.append((i, j) if i < j else (j, i))

    lo_face = node_id_grid.take(0, axis=axis)
    hi_face = node_id_grid.take(-1, axis=axis)
    low_nodes = lo_face[lo_face >= 0].tolist()
    high_nodes = hi_face[hi_face >= 0].tolist()
    return node_id_grid, edges, sorted(low_nodes), sorted(high_nodes)


def _effective_resistance_axis(lattice: Lattice, axis: int) -> float:
    """Face-to-face effective resistance along ``axis`` (``RESISTANCE_INF`` if no span).

    The two faces are ideal equipotential electrodes: every backbone cell on the low face
    is merged into super-node A, every cell on the high face into super-node B. The
    Laplacian is solved with node B grounded and unit current injected at A; the resulting
    potential at A is the effective resistance (spec §5.3).
    """
    mask = _spanning_cluster_mask(lattice, axis)
    if not mask.any():
        return RESISTANCE_INF

    node_id_grid, edges, low_nodes, high_nodes = _backbone_graph(mask, axis)
    n_cells = int(mask.sum())

    # Map cells -> Laplacian rows, merging the two electrodes. A = 0, B = 1, interior 2..
    elec_a, elec_b = 0, 1
    row = np.full(n_cells, -1, dtype=np.int64)
    low_set = set(low_nodes)
    high_set = set(high_nodes)
    # A cell on the low face and (degenerately) the high face would short the electrodes;
    # for any axis of size >= 2 the faces are disjoint so this cannot happen.
    next_row = 2
    rows = []
    for node in range(n_cells):
        if node in low_set:
            rows.append(elec_a)
        elif node in high_set:
            rows.append(elec_b)
        else:
            rows.append(next_row)
            next_row += 1
    row = np.asarray(rows, dtype=np.int64)
    n_rows = next_row

    # Assemble the weighted Laplacian (unit conductance per edge) over merged rows.
    data_i, data_j = [], []
    for i, j in edges:
        ri, rj = int(row[i]), int(row[j])
        if ri == rj:
            continue  # edge internal to an electrode -> no contribution
        data_i.append(ri)
        data_j.append(rj)
    if not data_i:
        # Electrodes are adjacent only through merged cells with no connecting edge.
        return RESISTANCE_INF
    ii = np.asarray(data_i)
    jj = np.asarray(data_j)
    # Off-diagonal -1 (symmetric), diagonal = degree.
    n = n_rows
    off = sparse.coo_matrix((-np.ones(ii.size), (ii, jj)), shape=(n, n))
    off = off + off.T
    deg = np.asarray(-off.sum(axis=1)).ravel()
    lap = (off + sparse.diags(deg)).tocsr()

    # Ground electrode B: drop its row/column, inject +1 A at electrode A, solve.
    keep = [r for r in range(n) if r != elec_b]
    lap_red = lap[keep][:, keep].tocsc()
    b = np.zeros(len(keep))
    a_pos = keep.index(elec_a)
    b[a_pos] = 1.0
    try:
        v = spsolve(lap_red, b)
    except Exception:
        return RESISTANCE_INF
    r_eff = float(v[a_pos])
    if not np.isfinite(r_eff) or r_eff <= 0.0:
        return RESISTANCE_INF
    return r_eff


def _bottleneck_width_axis(lattice: Lattice, axis: int) -> int:
    """Min-cut (max-flow, unit capacities) between the two faces along ``axis``.

    The number of edge-disjoint conducting paths: 1 for a single filament, up to the
    cross-section for a solid slab. 0 if there is no spanning cluster.
    """
    mask = _spanning_cluster_mask(lattice, axis)
    if not mask.any():
        return 0

    node_id_grid, edges, low_nodes, high_nodes = _backbone_graph(mask, axis)
    n_cells = int(mask.sum())
    # Super-source S and super-sink T as extra nodes; S->each low cell, each high cell->T
    # with large capacity so the cut falls inside the material, not at the contacts.
    S = n_cells
    T = n_cells + 1
    n = n_cells + 2
    big = n_cells + 1  # exceeds any internal min-cut, so contacts never bound the flow

    rows, cols, caps = [], [], []

    def add(u, v, c):
        rows.append(u)
        cols.append(v)
        caps.append(c)

    for i, j in edges:  # undirected -> unit capacity each direction
        add(i, j, 1)
        add(j, i, 1)
    for u in low_nodes:
        add(S, u, big)
    for w in high_nodes:
        add(w, T, big)

    graph = sparse.csr_matrix(
        (np.asarray(caps, dtype=np.int64), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n),
    )
    try:
        flow = csgraph.maximum_flow(graph, S, T)
        return int(flow.flow_value)
    except Exception:
        return 0


# --- public, per-lattice extractors ---------------------------------------------------


def _per_axis(lattice: Lattice):
    """(resistance, bottleneck_width) for each axis; cheap-exits non-spanning axes."""
    out = []
    for axis in range(lattice.dim):
        if not percolation.spans(lattice, axis):
            out.append((RESISTANCE_INF, 0))
        else:
            r = _effective_resistance_axis(lattice, axis)
            w = _bottleneck_width_axis(lattice, axis)
            out.append((r, w))
    return out


def effective_resistance(lattice: Lattice) -> float:
    """Lowest face-to-face effective resistance over all axes (best conducting direction).

    ``RESISTANCE_INF`` when the material does not span along any axis (an insulator).
    """
    return min(r for r, _ in _per_axis(lattice))


def conductivity(lattice: Lattice) -> float:
    """Continuous conductivity = ``1 / effective_resistance`` (spec §5.3). 0 for insulators."""
    r = effective_resistance(lattice)
    if r >= RESISTANCE_INF:
        return 0.0
    return 1.0 / r


def edge_connectivity(lattice: Lattice) -> int:
    """Widest min-cut over axes: the backbone's face-to-face edge-connectivity (spec §5.4).

    The number of edge-disjoint conducting paths across the lattice — 0 for an insulator,
    1 for a single tortuous filament, up to the cross-section for a solid slab. A genuine
    topological quantity; ``k`` edge-connectivity appears at the higher-order percolation
    threshold ``p_k > p_c``.
    """
    return max((w for _, w in _per_axis(lattice)), default=0)


def bottleneck_fraction(lattice: Lattice) -> float:
    """Edge-connectivity as a fraction of the cross-section (size-robust). In ``[0, 1]``.

    The continuous companion to :func:`edge_connectivity`: ~0 for filaments, ~1 for a solid
    slab. Used as the continuous "how loss-free" score.
    """
    best = 0.0
    for axis, (_, w) in enumerate(_per_axis(lattice)):
        cs = _cross_section(lattice.shape, axis)
        if cs > 0:
            best = max(best, w / cs)
    return best


def measure(lattice: Lattice) -> dict[str, float]:
    """All conducting-graph properties in a single per-axis solve.

    The individual extractors above each re-run the (relatively expensive) Laplacian +
    max-flow solve; this computes ``_per_axis`` once and derives every value from it, so
    :func:`engine.material.measure_properties` pays the cost only once per material.

    Returns conductivity, ``edge_connectivity`` and ``bottleneck_fraction`` (the backbone's
    redundancy — the structural input to the M6 phase-coherence superconductivity, no longer a
    superconductivity *flag* itself).
    """
    per_axis = _per_axis(lattice)
    best_cond = 0.0
    best_width = 0
    best_frac = 0.0
    for axis, (r, w) in enumerate(per_axis):
        cs = _cross_section(lattice.shape, axis)
        frac = (w / cs) if cs > 0 else 0.0
        best_frac = max(best_frac, frac)
        best_width = max(best_width, w)
        if r < RESISTANCE_INF:
            best_cond = max(best_cond, 1.0 / r)
    return {
        "conductivity_continuous": best_cond,
        "edge_connectivity": float(best_width),
        "bottleneck_fraction": best_frac,
    }
