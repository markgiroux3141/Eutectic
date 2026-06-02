"""The ``Lattice`` type and its seeded generation (spec §3.2, §4).

A material *is* a small lattice (spec §1). Properties are measurements taken from it, not
values assigned to it. This module owns the lattice's representation and the deterministic
processes that produce one:

* :class:`Lattice` — parallel numpy arrays (``occupied``, ``atom_type``, ``spin``), kept
  vectorized rather than an array-of-structs (spec §3.2).
* :func:`generate_base` — produce a lattice from a seed + a small set of affinities. Used
  both for root elements (spec §4.1) and as a building block.

Dimension is a config parameter (spec §2): default 2D 64x64 for prototyping, 3D
16x16x16 as the later target. All code here is dimension-agnostic.

:func:`merge` and :func:`relax` (spec §4.2/§4.3) are the two transform stages of the
combination pipeline; :func:`engine.material.combine` chains them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .rng import SplitMix64, hash_array, mix

# Default lattice shapes per dimension (spec §2). Override via Lattice config.
DEFAULT_SHAPE_2D: tuple[int, int] = (64, 64)
DEFAULT_SHAPE_3D: tuple[int, int, int] = (16, 16, 16)

# --- relaxation defaults (spec §4.3) --------------------------------------------------
# A short, FIXED number of settling sweeps -> identical settled lattice every time.
RELAX_STEPS: int = 8
# Ising temperature. 2D square-lattice T_c ~ 2.269 (J=1); sitting just below it lets
# ordered domains form without freezing the whole lattice into one block. Tuned for the
# magnetism transition in M3.
RELAX_TEMPERATURE: float = 2.0
# Default domain length-scale for merge: larger -> bigger contiguous parent domains.
MERGE_DOMAIN_SCALE: float = 2.0

# Number of distinct atom "kinds" a base lattice draws from. Drives bond rules later.
DEFAULT_ATOM_TYPES: int = 4


@dataclass(frozen=True)
class Lattice:
    """A settled (or freshly generated) grid of cells.

    Three parallel arrays share the same ``shape``:

    * ``occupied`` — uint8 {0,1}: is there matter in this cell (fill fraction feeds
      density and percolation).
    * ``atom_type`` — int8: which kind of site; 0 is reserved for "empty". Drives bond
      rules / coupling.
    * ``spin`` — int8 {-1,+1}: Ising spin for magnetism.

    Frozen so a Lattice is a value object; transforms return new instances. Arrays are
    not deep-copied on construction — callers should not mutate arrays they hand in.
    """

    occupied: np.ndarray
    atom_type: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        shapes = {self.occupied.shape, self.atom_type.shape, self.spin.shape}
        if len(shapes) != 1:
            raise ValueError(f"lattice arrays disagree on shape: {shapes}")
        if self.dim not in (2, 3):
            raise ValueError(f"unsupported lattice dimension: {self.dim}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.occupied.shape

    @property
    def dim(self) -> int:
        return self.occupied.ndim

    @property
    def size(self) -> int:
        return int(self.occupied.size)

    @property
    def fill_fraction(self) -> float:
        """Fraction of occupied cells (the percolation control parameter)."""
        return float(self.occupied.mean())

    def structural_signature(self) -> int:
        """Stable 64-bit hash of the lattice contents (spec §4.1).

        This is what gets fed into the combination seed, so it must reflect every array
        that affects measured properties.
        """
        return mix(
            hash_array(self.occupied),
            hash_array(self.atom_type),
            hash_array(self.spin),
        )

    def copy(self) -> "Lattice":
        return replace(
            self,
            occupied=self.occupied.copy(),
            atom_type=self.atom_type.copy(),
            spin=self.spin.copy(),
        )


def generate_base(
    seed: int,
    *,
    shape: Sequence[int] = DEFAULT_SHAPE_2D,
    affinities: Mapping[str, float] | None = None,
    n_atom_types: int = DEFAULT_ATOM_TYPES,
) -> Lattice:
    """Deterministically generate a base lattice from a seed + affinities (spec §4.1).

    The generation is intentionally simple for M0 — it must be *legible* and produce
    visible structure, not yet the full merge/relax emergence. Behaviour:

    * ``fill_density`` (from ``affinities['bond_energy']``, default ~0.55) sets how many
      cells are occupied — this is the knob that later sits near the percolation
      threshold.
    * Occupied cells get an ``atom_type`` in [1, n_atom_types]; empty cells are type 0.
    * ``spin`` is biased by ``affinities['magnetic_tendency']`` so magnetic-leaning
      elements start with a net alignment that Ising relaxation can amplify later.

    All randomness flows from a single :class:`SplitMix64` seeded by ``seed`` so the same
    seed always yields a byte-identical lattice.
    """
    affinities = dict(affinities or {})
    shape = tuple(int(s) for s in shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"shape must be 2D or 3D, got {shape}")

    rng = SplitMix64(seed)
    gen = rng.numpy_generator()

    # --- occupancy: fill fraction is the percolation control parameter (spec §5.2) ---
    bond_energy = float(affinities.get("bond_energy", 0.5))
    # Map bond_energy in [0,1] to a fill density in a useful band around the 2D site
    # percolation threshold (~0.5927) so elements naturally land on both sides of it.
    fill_density = 0.35 + 0.45 * _clamp01(bond_energy)
    occupied = (gen.random(shape) < fill_density).astype(np.uint8)

    # --- atom types: occupied cells draw a kind in [1, n_atom_types]; empty -> 0 ---
    conduction = float(affinities.get("conduction_tendency", 0.5))
    # Conduction-leaning elements concentrate mass into fewer "kinds" (more uniform
    # lattice -> better spanning clusters). Bias the categorical draw accordingly.
    type_weights = _type_weights(n_atom_types, skew=conduction, gen=gen)
    drawn = gen.choice(np.arange(1, n_atom_types + 1), size=shape, p=type_weights)
    atom_type = np.where(occupied == 1, drawn, 0).astype(np.int8)

    # --- spin: biased by magnetic tendency, only meaningful on occupied cells ---
    magnetic = float(affinities.get("magnetic_tendency", 0.5))
    up_prob = 0.5 + 0.45 * (_clamp01(magnetic) - 0.5) * 2.0  # magnetic=1 -> ~0.95 up
    spin_up = gen.random(shape) < up_prob
    spin = np.where(spin_up, 1, -1).astype(np.int8)
    # Empty cells carry no spin; pin them to +1 by convention so they don't add noise to
    # net magnetization measurements (occupied mask is applied at measure time too).
    spin = np.where(occupied == 1, spin, 1).astype(np.int8)

    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _type_weights(n: int, *, skew: float, gen: np.random.Generator) -> np.ndarray:
    """Categorical weights over n atom types, concentrated when skew is high."""
    # Higher skew -> lower Dirichlet concentration -> more uneven (concentrated) weights.
    concentration = 0.3 + (1.0 - _clamp01(skew)) * 3.0
    w = gen.dirichlet(np.full(n, concentration))
    return w


# --- combination pipeline stages (spec §4.2, §4.3) ------------------------------------


def merge(
    a: Lattice,
    b: Lattice,
    seed: int,
    *,
    domain_scale: float = MERGE_DOMAIN_SCALE,
) -> Lattice:
    """Produce a child lattice from two parents (spec §4.2).

    Rather than a per-cell coin flip (which produces salt-and-pepper noise with no
    spanning structure), we partition the lattice into contiguous **domains**, each
    inherited wholesale from one parent. Domains are carved by thresholding a *smoothed*
    random field at its median — a smooth field yields connected regions, and the median
    threshold gives a balanced ~50/50 split independent of parent fill (commutative).

    Interleaving domains like this is what later creates anisotropy and near-threshold
    spanning clusters (spec §4.2) once :func:`relax` settles the result.

    All randomness flows from ``seed`` via one :class:`SplitMix64`, so the child is a
    byte-identical function of (parents, seed).
    """
    if a.shape != b.shape:
        raise ValueError(f"cannot merge lattices of different shapes: {a.shape} vs {b.shape}")

    gen = SplitMix64(seed).numpy_generator()
    field = gen.random(a.shape)
    # Smooth into connected domains. mode="wrap" keeps it deterministic and seam-free.
    field = ndimage.gaussian_filter(field, sigma=domain_scale, mode="wrap")
    take_a = field <= np.median(field)

    occupied = np.where(take_a, a.occupied, b.occupied).astype(np.uint8)
    atom_type = np.where(take_a, a.atom_type, b.atom_type).astype(np.int8)
    spin = np.where(take_a, a.spin, b.spin).astype(np.int8)
    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin)


def _neighbor_spin_sum(spin_field: np.ndarray) -> np.ndarray:
    """Sum of face-neighbour values for every cell (periodic boundary, deterministic)."""
    total = np.zeros(spin_field.shape, dtype=np.int32)
    for axis in range(spin_field.ndim):
        total += np.roll(spin_field, 1, axis=axis)
        total += np.roll(spin_field, -1, axis=axis)
    return total


def relax(
    lattice: Lattice,
    seed: int,
    *,
    steps: int = RELAX_STEPS,
    temperature: float = RELAX_TEMPERATURE,
    coupling: float = 1.0,
) -> Lattice:
    """Settle the lattice's spins via deterministic Metropolis dynamics (spec §4.3).

    This is where emergence happens: below the critical temperature, spins spontaneously
    align into ordered domains; the magnetism extractor (M3) reads that order out.

    Determinism (spec §6) is guaranteed by three things working together:

    * all acceptance randomness comes from one :class:`SplitMix64`-seeded generator,
    * a fixed sweep count and a fixed two-colour (checkerboard) update order, and
    * the checkerboard split means no two simultaneously-updated cells are neighbours, so
      the vectorised update is exactly equivalent to a fixed sequential sweep.

    Only ``spin`` is relaxed; ``occupied``/``atom_type`` (and hence density and
    percolation) are left as :func:`merge` produced them. Empty cells carry no spin and
    are pinned to +1 by convention so they don't bias magnetisation measurements.
    """
    occ = lattice.occupied.astype(bool)
    spin = lattice.spin.astype(np.int8).copy()
    gen = SplitMix64(seed).numpy_generator()

    # Two-colour (checkerboard) masks: parity of the summed coordinates.
    parity = np.indices(lattice.shape).sum(axis=0) % 2
    colors = (parity == 0, parity == 1)

    for _ in range(steps):
        for color_mask in colors:
            field = (spin.astype(np.int32)) * occ  # empty neighbours contribute 0
            h = _neighbor_spin_sum(field)
            # Energy change on flipping s -> -s for E = -J * sum_<ij> s_i s_j.
            delta_e = 2.0 * coupling * spin * h
            r = gen.random(lattice.shape)
            # exponent capped at 0 so dE<=0 never overflows exp; those flip unconditionally.
            accept = (delta_e <= 0) | (r < np.exp(np.minimum(-delta_e / temperature, 0.0)))
            update = accept & color_mask & occ
            spin = np.where(update, -spin, spin).astype(np.int8)

    spin = np.where(occ, spin, 1).astype(np.int8)
    return Lattice(occupied=lattice.occupied, atom_type=lattice.atom_type, spin=spin)
