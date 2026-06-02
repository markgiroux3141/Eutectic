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

``merge`` and ``relax`` (spec §4.2/§4.3) arrive in M1; the hooks are stubbed at the end
of this module so the shape of the pipeline is visible now.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .rng import SplitMix64, hash_array, mix

# Default lattice shapes per dimension (spec §2). Override via Lattice config.
DEFAULT_SHAPE_2D: tuple[int, int] = (64, 64)
DEFAULT_SHAPE_3D: tuple[int, int, int] = (16, 16, 16)

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


# --- M1 pipeline hooks (stubbed now so the combine() shape is visible) -----------------


def merge(a: Lattice, b: Lattice, seed: int) -> Lattice:  # pragma: no cover - M1
    """Produce a child lattice from two parents (spec §4.2). Implemented in M1."""
    raise NotImplementedError("merge() lands in M1 (combination pipeline)")


def relax(lattice: Lattice, seed: int, *, steps: int) -> Lattice:  # pragma: no cover - M1
    """Settle a lattice via deterministic local energy minimization (spec §4.3). M1."""
    raise NotImplementedError("relax() lands in M1 (combination pipeline)")
