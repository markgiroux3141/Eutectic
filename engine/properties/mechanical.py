"""Mechanical properties from a central-force spring network (spec §5.7; M8).

Strength and ductility are *measured* from the bonds of the settled lattice, never assigned.
The lattice's occupied cells are nodes (2 DOF each in 2D, 3 in 3D); bonds connect occupied
**nearest- and next-nearest (diagonal) neighbours** with a central-force (bond-stretching)
spring whose stiffness ``k_ij = cohesion_i · cohesion_j`` comes from the same ``cohesion``
field that sets melting (M5) — so a stiff-bonded material is both high-melting and strong, as
it should be. The stiffness matrix ``K`` (the elastic analogue of M3's scalar Laplacian) is

    E(u) = ½ Σ_⟨ij⟩ k_ij [ (u_i − u_j)·n̂_ij ]²

with ``n̂_ij`` the unit bond direction. Two measurements come off it:

* **strength** = the **shear modulus**: clamp two opposite faces, impose a unit shear, and read
  the relaxed strain energy (the vector analogue of the merged-electrode resistance solve in
  ``conductance.py``). Shear is *the* rigidity signal — a network with zero shear modulus is a
  mechanism that flows like a liquid; it rises only once the bond network is over-constrained
  enough to be rigid. (The **bulk modulus**, the compression response, is exposed alongside.)
* **ductility** = the **normalised coordination deficit** ``1 − z̄/z_max``: the structural
  density of under-coordinated sites — the incipient "slip planes" of spec §5.7. ``z̄`` is the
  mean number of occupied NN+diagonal neighbours and ``z_max = 3^dim − 1`` is the geometric
  maximum, so this is a knob-free, O(N) geometric quantity — invariant to scaling every ``k`` —
  independent of the cohesion magnitude that drives strength. The strength↔ductility
  anti-correlation then *emerges* through coordination (more constraints → higher modulus but
  fewer slip sites), rather than being wired in. (The *exact* zero-energy floppy-mode fraction —
  :func:`floppy_fraction`, a dense ``eigvalsh`` — is the rigorous mechanics behind this, but it
  costs 5–13 s/material and is near-zero for every true solid, so it is kept as an explorer /
  validation instrument; the cheap deficit tracks it (corr ≈ 0.9) *and* resolves solids better,
  exactly the "coarse stored value, accurate instrument on demand" pattern used for Curie/melting.)

Honest scope (de-risked, see README "M8 findings"): an NN-only central-force square lattice is a
shear *mechanism* even when full, so we add the diagonal bonds that brace it; the resulting
generic shear-rigidity threshold sits at coordination z≈6–7 (above the mean-field Maxwell
isostatic z=2d=4 — the square lattice's bonds are partially redundant). Our ~0.6-fill materials
therefore live in the *marginally-rigid* regime: moduli are small but cleanly discriminating
(refractory/dense → strong+brittle; porous/soft → weak+ductile). Stress σ is the conjugate
condition (strain → fracture); it rides inert for now, to be activated like pressure was in M5.

Determinism (spec §6): nodes are ordered by flattened cell index, the modulus uses a direct
sparse solver (``spsolve``), the floppy count uses ``eigvalsh`` (both deterministic), and there
is no RNG anywhere. All public functions are pure ``Lattice -> float``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from ..lattice import Lattice
from . import percolation

# Fixed display scale so the marginally-rigid moduli land at legible O(0.1–100) magnitudes
# (the materials sit just past the shear-rigidity threshold, so raw strain energies are small).
# A constant, NOT a tuned dial — it rescales every material identically and changes no ordering.
MODULUS_SCALE: float = 100.0

# Relative regularizer for the clamped-face solve: floppy INTERIOR modes (a dangling atom) make
# the free-block singular, but the strain energy under the imposed boundary is still well defined
# (floppy modes relax to zero energy). Solving (K_ff + eps·I) and reading energy off the *un*-
# regularized K recovers it; eps is tiny relative to the diagonal so the limit is the true modulus.
_REG_EPS_REL: float = 1.0e-8

# Eigenvalue below this (relative to the largest) counts as a zero / floppy mode.
_ZERO_TOL: float = 1.0e-8


def _node_index(mask: np.ndarray):
    """(flat_idx, node_grid, n_nodes) — map masked cells to contiguous node ids (fixed order)."""
    flat = np.flatnonzero(mask.reshape(-1))
    node = -np.ones(mask.size, dtype=np.int64)
    node[flat] = np.arange(flat.size)
    return flat, node.reshape(mask.shape), flat.size


def _bond_offsets(ndim: int):
    """Nearest- + next-nearest-neighbour offsets (positive-direction, each undirected bond once).

    2D: NN (right, down) + diagonals; 3D: the 13 forward neighbours of the 3×3×3 stencil. The
    diagonals are the bracing that makes the square/cubic central-force network rigid in shear.
    """
    if ndim == 2:
        return [(0, 1), (1, 0), (1, 1), (1, -1)]
    if ndim == 3:
        offs = []
        for dz in (0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and (dy < 0 or (dy == 0 and dx <= 0)):
                        continue  # keep one of each ± pair
                    if (dz, dy, dx) == (0, 0, 0):
                        continue
                    offs.append((dz, dy, dx))
        return offs
    raise ValueError(f"unsupported dimension: {ndim}")


def _bonds(lattice: Lattice):
    """Central-force bonds between occupied NN+diagonal cells.

    Returns ``(i, j, dirs, k)``: node-id endpoint arrays, unit direction vectors (n_bonds×dim),
    and per-bond stiffness ``k = cohesion_i·cohesion_j``. Built vectorised per offset so the order
    is a deterministic function of the offset list and flattened cell index.
    """
    mask = percolation.solid_mask(lattice)
    flat, node_grid, n_nodes = _node_index(mask)
    coh = np.asarray(lattice.cohesion, dtype=np.float64)
    ndim = lattice.dim

    iis, jjs, dirs, ks = [], [], [], []
    for off in _bond_offsets(ndim):
        length = float(np.sqrt(sum(o * o for o in off)))
        unit = np.array(off, dtype=np.float64) / length
        # shifted node grid: neighbour at +off (no wrap — physical faces, like conductance)
        shifted = node_grid
        shifted_coh = coh
        for axis, o in enumerate(off):
            shifted = np.roll(shifted, -o, axis=axis)
            shifted_coh = np.roll(shifted_coh, -o, axis=axis)
        # invalidate wrapped rows/cols so we don't bond across the periodic seam
        valid = np.ones(mask.shape, dtype=bool)
        for axis, o in enumerate(off):
            idx = [slice(None)] * ndim
            if o > 0:
                idx[axis] = slice(mask.shape[axis] - o, None)
            elif o < 0:
                idx[axis] = slice(0, -o)
            if o != 0:
                valid[tuple(idx)] = False
        both = (node_grid >= 0) & (shifted >= 0) & valid
        i_here = node_grid[both]
        j_there = shifted[both]
        k_here = (coh[both] * shifted_coh[both])
        iis.append(i_here)
        jjs.append(j_there)
        dirs.append(np.broadcast_to(unit, (i_here.size, ndim)))
        ks.append(k_here)

    if not iis or sum(a.size for a in iis) == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty((0, ndim)), np.empty(0), n_nodes)
    return (np.concatenate(iis), np.concatenate(jjs),
            np.concatenate(dirs), np.concatenate(ks), n_nodes)


def _stiffness_matrix(i, j, dirs, k, n_nodes, ndim):
    """Assemble the central-force stiffness matrix K (dim·N × dim·N), sparse CSR.

    Each bond contributes the block ``k·(n̂⊗n̂)`` on the (i,i),(j,j) diagonals and ``−k·(n̂⊗n̂)``
    on the (i,j),(j,i) off-diagonals — the standard central-force (bond-stretching) element.
    """
    n_dof = ndim * n_nodes
    if i.size == 0:
        return sparse.csr_matrix((n_dof, n_dof))
    # per-bond dim×dim block B = k * outer(n, n)
    blocks = k[:, None, None] * (dirs[:, :, None] * dirs[:, None, :])  # (n_bonds, dim, dim)
    rows, cols, vals = [], [], []
    for a in range(ndim):
        for b in range(ndim):
            v = blocks[:, a, b]
            ia, ib = ndim * i + a, ndim * i + b
            ja, jb = ndim * j + a, ndim * j + b
            # +B on ii, jj ; -B on ij, ji
            rows.append(ia); cols.append(ib); vals.append(v)
            rows.append(ja); cols.append(jb); vals.append(v)
            rows.append(ia); cols.append(jb); vals.append(-v)
            rows.append(ja); cols.append(ib); vals.append(-v)
    K = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dof, n_dof),
    ).tocsr()
    return K


def _face_nodes(node_grid: np.ndarray, axis: int):
    """(low_face_nodes, high_face_nodes): node ids on the two faces perpendicular to ``axis``."""
    lo = np.take(node_grid, 0, axis=axis)
    hi = np.take(node_grid, -1, axis=axis)
    return lo[lo >= 0], hi[hi >= 0]


def _modulus_axis(K, node_grid, n_nodes, ndim, axis: int, *, shear: bool) -> float:
    """Clamped-face elastic modulus along ``axis``: impose a unit strain, return relaxed energy.

    ``shear`` imposes a unit displacement transverse to ``axis`` on the high face (shear); else a
    unit displacement along ``axis`` (compression / bulk). Both faces are fully clamped. Returns
    the strain energy (∝ the modulus), 0.0 if there are no faces. Regularized so floppy interior
    modes don't abort the solve (see ``_REG_EPS_REL``).
    """
    lo_nodes, hi_nodes = _face_nodes(node_grid, axis)
    if lo_nodes.size == 0 or hi_nodes.size == 0:
        return 0.0
    n_dof = ndim * n_nodes
    u = np.zeros(n_dof)
    fixed = np.zeros(n_dof, dtype=bool)
    drive = (axis + 1) % ndim if shear else axis  # transverse comp for shear, axial for bulk
    for nd in lo_nodes:
        for c in range(ndim):
            fixed[ndim * nd + c] = True
    for nd in hi_nodes:
        for c in range(ndim):
            fixed[ndim * nd + c] = True
        u[ndim * nd + drive] = 1.0

    free = ~fixed
    if not free.any():
        return 0.0
    Kff = K[free][:, free]
    Kfb = K[free][:, fixed]
    rhs = -Kfb.dot(u[fixed])
    n_free = Kff.shape[0]
    diag_scale = max(1e-12, float(abs(Kff.diagonal()).mean()))
    Kreg = (Kff + (_REG_EPS_REL * diag_scale) * sparse.identity(n_free, format="csr")).tocsc()
    try:
        uf = spsolve(Kreg, rhs)
    except Exception:
        return 0.0
    if not np.all(np.isfinite(uf)):
        return 0.0
    u[free] = uf
    return 0.5 * float(u.dot(K.dot(u)))


def _cross_section(shape, axis: int) -> int:
    n = 1
    for ax, s in enumerate(shape):
        if ax != axis:
            n *= s
    return n


def shear_modulus(lattice: Lattice) -> float:
    """Best-axis shear modulus (the strength signal), scaled and normalised per cross-section."""
    i, j, dirs, k, n_nodes = _bonds(lattice)
    if n_nodes < lattice.dim + 1:
        return 0.0
    K = _stiffness_matrix(i, j, dirs, k, n_nodes, lattice.dim)
    mask = percolation.solid_mask(lattice)
    _, node_grid, _ = _node_index(mask)
    best = 0.0
    for axis in range(lattice.dim):
        e = _modulus_axis(K, node_grid, n_nodes, lattice.dim, axis, shear=True)
        cs = _cross_section(lattice.shape, axis)
        best = max(best, MODULUS_SCALE * e / cs if cs else 0.0)
    return best


def bulk_modulus(lattice: Lattice) -> float:
    """Best-axis compression (bulk) modulus, scaled and normalised per cross-section."""
    i, j, dirs, k, n_nodes = _bonds(lattice)
    if n_nodes < lattice.dim + 1:
        return 0.0
    K = _stiffness_matrix(i, j, dirs, k, n_nodes, lattice.dim)
    mask = percolation.solid_mask(lattice)
    _, node_grid, _ = _node_index(mask)
    best = 0.0
    for axis in range(lattice.dim):
        e = _modulus_axis(K, node_grid, n_nodes, lattice.dim, axis, shear=False)
        cs = _cross_section(lattice.shape, axis)
        best = max(best, MODULUS_SCALE * e / cs if cs else 0.0)
    return best


def ductility(lattice: Lattice) -> float:
    """Ductility = the normalised coordination deficit ``1 − z̄/z_max`` (spec §5.7).

    ``z̄`` is the mean count of occupied NN+diagonal neighbours over occupied sites and
    ``z_max = 3^dim − 1`` the geometric maximum (8 in 2D, 26 in 3D). The density of
    under-coordinated, slip-enabling sites: ~0 for a fully-coordinated (brittle) crystal, rising
    toward 1 as the structure becomes porous/floppy. Cheap (one convolution), deterministic, and
    independent of the spring stiffnesses — so it is a genuine second axis to :func:`shear_modulus`,
    not a relabelling of it. A dispersed structure with no occupied cells is wholly floppy (1.0).
    """
    mask = percolation.solid_mask(lattice)
    if not mask.any():
        return 1.0
    kernel = np.ones((3,) * lattice.dim, dtype=int)
    kernel[(1,) * lattice.dim] = 0  # exclude self -> the 8 (2D) / 26 (3D) NN+diagonal neighbours
    z = ndimage.convolve(mask.astype(int), kernel, mode="constant")[mask]
    z_max = 3 ** lattice.dim - 1
    return float(1.0 - z.mean() / z_max)


def floppy_fraction(lattice: Lattice) -> float:
    """Exact zero-energy (floppy) mode fraction — the rigorous mechanics behind :func:`ductility`.

    The nullity of ``K`` beyond the trivial rigid-body modes, normalised by the degrees of
    freedom — a geometric quantity (invariant to scaling every spring constant). It is the
    textbook definition of mechanical floppiness, but it is **expensive** (a dense ``eigvalsh``,
    5–13 s at 64²) and **near-zero for every true solid** (it cleanly separates solids from
    liquids/gases but barely resolves brittle-vs-ductile among solids). So it is an explorer /
    validation instrument, not the stored property; :func:`ductility` is the cheap stored measure
    that tracks it (corr ≈ 0.9) while resolving solids. Used in tests to validate the keystone.
    """
    i, j, dirs, k, n_nodes = _bonds(lattice)
    ndim = lattice.dim
    if n_nodes < ndim + 1:
        return 1.0  # nothing to brace -> wholly floppy
    K = _stiffness_matrix(i, j, dirs, k, n_nodes, ndim)
    n_dof = ndim * n_nodes
    ev = np.linalg.eigvalsh(K.toarray())
    scale = max(1.0, float(ev.max()))
    n_zero = int(np.sum(ev < _ZERO_TOL * scale))
    rigid_body = ndim * (ndim + 1) // 2  # 3 in 2D, 6 in 3D
    floppy = max(0, n_zero - rigid_body)
    return floppy / n_dof


def measure(lattice: Lattice) -> dict[str, float]:
    """All (stored) mechanical properties; builds the stiffness matrix once for the moduli.

    Returns ``strength`` (shear modulus), ``ductility`` (the cheap coordination deficit — the
    expensive exact :func:`floppy_fraction` stays an explorer/validation instrument), and the
    ``bulk_modulus`` alongside, so the strength↔ductility tradeoff is legible. Measured from the
    bond network, never assigned (spec §1, §5.7). No RNG; no dense eigendecomposition in this path.
    """
    duct = ductility(lattice)
    i, j, dirs, k, n_nodes = _bonds(lattice)
    ndim = lattice.dim
    if n_nodes < ndim + 1:
        return {"strength": 0.0, "ductility": duct, "bulk_modulus": 0.0}

    K = _stiffness_matrix(i, j, dirs, k, n_nodes, ndim)
    mask = percolation.solid_mask(lattice)
    _, node_grid, _ = _node_index(mask)

    best_shear = best_bulk = 0.0
    for axis in range(ndim):
        cs = _cross_section(lattice.shape, axis)
        if not cs:
            continue
        es = _modulus_axis(K, node_grid, n_nodes, ndim, axis, shear=True)
        eb = _modulus_axis(K, node_grid, n_nodes, ndim, axis, shear=False)
        best_shear = max(best_shear, MODULUS_SCALE * es / cs)
        best_bulk = max(best_bulk, MODULUS_SCALE * eb / cs)

    return {
        "strength": best_shear,
        "ductility": duct,
        "bulk_modulus": best_bulk,
    }
