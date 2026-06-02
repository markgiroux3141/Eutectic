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
RELAX_STEPS: int = 12
# Ising temperature. Sat below the pure-lattice T_c (~2.269 at J=1), but these lattices
# are *site-diluted* (~60% fill), which suppresses the effective T_c toward the
# percolation point. T=1.0 is tuned so strongly-coupled (high-moment, iron-family) regions
# robustly order even when dilute, while weakly-coupled regions thermalise to disorder —
# giving the clean magnetism transition this milestone is after (spec §5.5).
RELAX_TEMPERATURE: float = 1.0
# Default domain length-scale for merge: larger -> bigger contiguous parent domains.
MERGE_DOMAIN_SCALE: float = 2.0

# --- magnetism / Ising coupling (spec §5.5) -------------------------------------------
# Global exchange constant. Bond coupling is J = EXCHANGE_J0 * moment_i * moment_j, so a
# bond between two unit-moment cells has coupling EXCHANGE_J0.
EXCHANGE_J0: float = 1.0
# Per-cell magnetic moment is mapped linearly from an element's ``magnetic_tendency`` in
# [0,1] into [MOMENT_LO, MOMENT_HI]. The critical moment (where J0*m^2/T crosses the 2D
# Ising point J/T ~ 0.4407, i.e. m_c = sqrt(0.4407*T/J0) ~ 0.94 at the defaults) is tuned
# to fall *between* the ferromagnets (iron/cobalt/nickel, tendency >= 0.78 -> m >= 1.06)
# and everything else (tendency <= 0.40 -> m <= 0.64) so the transition is legible.
MOMENT_LO: float = 0.20
MOMENT_HI: float = 1.30

# Number of distinct atom "kinds" a base lattice draws from. Drives bond rules later.
DEFAULT_ATOM_TYPES: int = 4


@dataclass(frozen=True)
class Lattice:
    """A settled (or freshly generated) grid of cells.

    Parallel arrays share the same ``shape`` (kept vectorized rather than an
    array-of-structs — spec §3.2):

    * ``occupied`` — uint8 {0,1}: is there matter in this cell (fill fraction feeds
      density and percolation).
    * ``atom_type`` — int8: which kind of site; 0 is reserved for "empty". Drives bond
      rules / coupling.
    * ``spin`` — int8 {-1,+1}: Ising spin for magnetism.
    * ``mass`` — float32: per-cell atomic mass (0 where empty). A root fills its occupied
      cells with the element's ``atomic_mass``; :func:`merge` carries each parent's mass
      through its domains, so a combination's mass field reflects its real blend. This is
      what lets density be *measured* from the structure for combos, not just roots
      (spec §5.1). Optional at construction: if omitted it defaults to unit mass on
      occupied cells, which is all synthetic/test lattices need.
    * ``moment`` — float32: per-cell magnetic moment (0 where empty). The structural
      source of spin coupling: :func:`relax` couples neighbours by ``J0 * moment_i *
      moment_j``, so high-moment cells (iron-family) order while low-moment ones (copper)
      stay thermally disordered (spec §5.5). A root fills its occupied cells from the
      element's ``magnetic_tendency``; :func:`merge` carries it through domains like mass.
      This is what makes magnetism a *measured* phase transition rather than an assigned
      number. Optional at construction; if omitted it defaults to unit moment on occupied
      cells (uniform-coupling Ising, all synthetic/test lattices need).

    Frozen so a Lattice is a value object; transforms return new instances. Arrays are
    not deep-copied on construction — callers should not mutate arrays they hand in.
    """

    occupied: np.ndarray
    atom_type: np.ndarray
    spin: np.ndarray
    mass: np.ndarray | None = None
    moment: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mass is None:
            # Backward-compatible default: unit mass on occupied cells.
            object.__setattr__(self, "mass", self.occupied.astype(np.float32))
        if self.moment is None:
            # Backward-compatible default: unit moment on occupied cells (uniform coupling).
            object.__setattr__(self, "moment", self.occupied.astype(np.float32))
        shapes = {
            self.occupied.shape,
            self.atom_type.shape,
            self.spin.shape,
            self.mass.shape,
            self.moment.shape,
        }
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
            hash_array(self.mass),
            hash_array(self.moment),
        )

    def copy(self) -> "Lattice":
        return replace(
            self,
            occupied=self.occupied.copy(),
            atom_type=self.atom_type.copy(),
            spin=self.spin.copy(),
            mass=self.mass.copy(),
            moment=self.moment.copy(),
        )


def generate_base(
    seed: int,
    *,
    shape: Sequence[int] = DEFAULT_SHAPE_2D,
    affinities: Mapping[str, float] | None = None,
    n_atom_types: int = DEFAULT_ATOM_TYPES,
    mass_per_atom: float = 1.0,
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

    # --- moment: each occupied cell's magnetic moment, from magnetic_tendency (spec §5.5).
    # This is the structural source of spin coupling in relax(); a high-moment element
    # (iron) orders, a low-moment one (copper) stays disordered. Empty cells carry 0.
    moment_val = MOMENT_LO + (MOMENT_HI - MOMENT_LO) * _clamp01(magnetic)
    moment = (occupied.astype(np.float32) * np.float32(moment_val))

    # --- mass: each occupied cell carries the element's atomic mass (spec §5.1) ---
    mass = (occupied.astype(np.float32) * np.float32(mass_per_atom))

    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin, mass=mass, moment=moment)


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
    mass = np.where(take_a, a.mass, b.mass).astype(np.float32)
    moment = np.where(take_a, a.moment, b.moment).astype(np.float32)
    return Lattice(
        occupied=occupied, atom_type=atom_type, spin=spin, mass=mass, moment=moment
    )


def _neighbor_sum(field: np.ndarray) -> np.ndarray:
    """Sum of face-neighbour values for every cell (periodic boundary, deterministic).

    Accumulates in float64 so it works for both the integer spin field and the
    moment-weighted (float) field used by the structural Ising coupling.
    """
    total = np.zeros(field.shape, dtype=np.float64)
    for axis in range(field.ndim):
        total += np.roll(field, 1, axis=axis)
        total += np.roll(field, -1, axis=axis)
    return total


def relax(
    lattice: Lattice,
    seed: int,
    *,
    steps: int = RELAX_STEPS,
    temperature: float = RELAX_TEMPERATURE,
    coupling: float = EXCHANGE_J0,
) -> Lattice:
    """Settle the lattice's spins via deterministic Metropolis dynamics (spec §4.3, §5.5).

    This is where emergence happens: where the structural coupling is strong enough,
    spins spontaneously align into ordered domains; the magnetism extractor (M3) reads
    that order out.

    **The coupling is derived from structure, not assigned.** Each bond's exchange is
    ``J_ij = coupling * moment_i * moment_j`` (separable, so the vectorised update stays a
    one-liner). High-moment cells (iron-family) couple strongly and order; low-moment
    cells (copper) couple weakly and stay thermally disordered at ``temperature``. Because
    relaxation and the :func:`engine.properties.ising.magnetism` measurement both read the
    same ``moment`` field, the settled order and the measured magnetisation agree (spec
    §5.5). With the default unit-moment field this reduces to a uniform-coupling Ising.

    Determinism (spec §6) is guaranteed by three things working together:

    * all acceptance randomness comes from one :class:`SplitMix64`-seeded generator,
    * a fixed sweep count and a fixed two-colour (checkerboard) update order, and
    * the checkerboard split means no two simultaneously-updated cells are neighbours, so
      the vectorised update is exactly equivalent to a fixed sequential sweep.

    Only ``spin`` is relaxed; ``occupied``/``atom_type``/``mass``/``moment`` (and hence
    density and percolation) are left as :func:`merge` produced them. Empty cells carry no
    spin and are pinned to +1 by convention so they don't bias magnetisation measurements.
    """
    occ = lattice.occupied.astype(bool)
    spin = lattice.spin.astype(np.int8).copy()
    # Effective moment: the structural coupling weight, zero on empty cells so they couple
    # to nothing and never flip.
    m = np.asarray(lattice.moment, dtype=np.float64) * occ
    gen = SplitMix64(seed).numpy_generator()

    # Two-colour (checkerboard) masks: parity of the summed coordinates.
    parity = np.indices(lattice.shape).sum(axis=0) % 2
    colors = (parity == 0, parity == 1)

    for _ in range(steps):
        for color_mask in colors:
            # Local field h_i = sum_j (m_j * s_j); with J_ij = J0*m_i*m_j the energy change
            # on flipping s_i -> -s_i is dE_i = 2 * J0 * s_i * m_i * h_i.
            h = _neighbor_sum(m * spin)
            delta_e = 2.0 * coupling * spin * m * h
            r = gen.random(lattice.shape)
            # exponent capped at 0 so dE<=0 never overflows exp; those flip unconditionally.
            accept = (delta_e <= 0) | (r < np.exp(np.minimum(-delta_e / temperature, 0.0)))
            update = accept & color_mask & occ
            spin = np.where(update, -spin, spin).astype(np.int8)

    spin = np.where(occ, spin, 1).astype(np.int8)
    return Lattice(
        occupied=lattice.occupied,
        atom_type=lattice.atom_type,
        spin=spin,
        mass=lattice.mass,
        moment=lattice.moment,
    )
