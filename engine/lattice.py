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

# --- occupancy / lattice-gas coupling (M5, melting) -----------------------------------
# Global occupancy coupling, the analogue of EXCHANGE_J0 for the *positional* (melting)
# transition. The occupancy order-disorder point is the textbook 2D-Ising T_m = 2.269·J0_occ
# for a uniform-cohesion lattice — i.e. COHESION_J0 plays exactly the role for melting that
# EXCHANGE_J0 plays for the Curie point, so J0_occ=1 melts where J0=1 orders. (Internally the
# lattice-gas energy carries the n=(s+1)/2 mapping factor of 4; see :func:`occupancy_sweep`.)
COHESION_J0: float = 1.0
# The lattice-gas ↔ Ising mapping factor: with n=(s+1)/2, ε n_i n_j expands to ε/4·s_i s_j,
# so an Ising-equivalent coupling J needs lattice-gas repulsion ε = 4J. Folding it in here
# lets COHESION_J0 be stated directly in Ising (Curie-comparable) units.
_LATTICE_GAS_FACTOR: float = 4.0

# --- superconductivity / XY phase-coherence coupling (M6) ------------------------------
# Global XY coupling for the phase-coherence (superconductivity) ensemble. A conducting cell
# carries a phase θ; bonds couple as -J·cos(θ_i−θ_j) across the conducting backbone. In 2D
# this orders via a Berezinskii–Kosterlitz–Thouless (BKT) transition at the textbook
# T_BKT = 0.893·SC_J0 for a *fully conducting* lattice (the parameter-free keystone). SC_J0=1
# is the natural unit in which the keystone recovers the universal 0.893; a material's actual
# superconducting Tc emerges *lower* and is set by how rigid (redundant) its backbone is.
SC_J0: float = 1.0
# Default proposal half-window for the continuous-angle Metropolis update (radians). Sized so
# acceptance stays healthy across the BKT temperature range on the prototype lattices.
SC_PROPOSAL_WINDOW: float = 1.6
# Per-cell magnetic moment is mapped linearly from an element's ``magnetic_tendency`` in
# [0,1] into [MOMENT_LO, MOMENT_HI]. The critical moment (where J0*m^2/T crosses the 2D
# Ising point J/T ~ 0.4407, i.e. m_c = sqrt(0.4407*T/J0) ~ 0.94 at the defaults) is tuned
# to fall *between* the ferromagnets (iron/cobalt/nickel, tendency >= 0.78 -> m >= 1.06)
# and everything else (tendency <= 0.40 -> m <= 0.64) so the transition is legible.
MOMENT_LO: float = 0.20
MOMENT_HI: float = 1.30

# --- cohesion / lattice-gas coupling (M5, docs §4-§5) ---------------------------------
# Per-cell **bond stiffness**: the energy scale that resists positional disordering, i.e.
# the structural source of a material's melting point. Unlike mass/moment (which are "what
# is here now", zero on empty cells), cohesion is "what this *site* is" — the bonding a cell
# would contribute *if* occupied — so it is defined on EVERY cell, because M5 lets occupancy
# itself become thermal (a previously-empty site can fill, and must know its own cohesion).
# Mapped linearly from an element's ``bond_energy`` affinity in [0,1] into [COH_LO, COH_HI].
# The occupancy order-disorder (melting) temperature scales as T_m ∝ J0_occ·cohesion² (the
# 2D-Ising/lattice-gas mapping), so high-bond_energy elements (tungsten, carbon) melt high
# and low-bond_energy ones (mercury, hydrogen) melt low — a *measured*, legible ordering.
COH_LO: float = 0.60
COH_HI: float = 1.25

# --- metallicity / charge-carrier gating (M6b) ----------------------------------------
# Per-cell **metallicity**: does the atom on this site carry *charge* (a metal) or only heat
# (phonons, like an insulator/diamond)? Mapped from an element's ``conduction_tendency``. This
# is what splits the two heat carriers (docs §4): charge conducts only through metallic cells
# (so electrical conductivity / superconductivity ride the *metallic* backbone), while heat
# (phonons) flows through all occupied matter — giving the diamond divergence (a stiff
# non-metal conducts heat superbly but no charge) and a Wiedemann–Franz electronic channel.
# Like ``cohesion`` it is a per-site property (defined everywhere; masked by occupancy at use).
DEFAULT_METALLICITY: float = 1.0  # constructed lattices default to metallic (charge = occupied)

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
    * ``cohesion`` — float32, per-cell **bond stiffness** (the source of the melting point,
      M5). Unlike mass/moment it is defined on *every* cell (it is a property of the *site*,
      not of "what is here now"), because M5 makes occupancy thermal — a previously-empty
      cell can fill and must carry its own cohesion. A root fills it from ``bond_energy``;
      :func:`merge` carries it through domains. Optional at construction; if omitted defaults
      to uniform 1.0 everywhere (the keystone's plain lattice-gas).
    * ``metallicity`` — float32, per-cell **charge-carrier quality** (M6b): whether the site's
      atom carries charge (metal) or only heat (insulator). A root fills it from
      ``conduction_tendency``; :func:`merge` carries it through domains. Charge conduction
      (electrical, superconductivity) rides only cells above the metallic threshold; heat
      (phonons) flows through all occupied matter. Optional; defaults to uniform 1.0 (metallic),
      so a constructed lattice's charge backbone is just its occupied set (M2 behaviour).

    Frozen so a Lattice is a value object; transforms return new instances. Arrays are
    not deep-copied on construction — callers should not mutate arrays they hand in.

    * ``site_potential`` — float32, per-cell **on-site electronic potential** (the source of the
      band gap, M7). A tight-binding Hamiltonian places this on its diagonal; a *staggered*
      potential on a 2-sublattice (ionic) crystal opens a gap ``= 2Δ`` (Δ the stagger amplitude),
      while a *uniform* (zero, or constant) potential opens none — so a charge-staggered rock-salt
      lattice is an insulator and a uniform metallic one a conductor. Chemistry sets it from
      species electronegativity (``Δ ∝ χ_site − χ̄``); a generated/metallic lattice leaves it 0
      (no ionic gap → conductor, the already-validated behaviour). Optional; defaults to **zeros**
      (no stagger), so every pre-M7 lattice is byte-identical and gap-free.

    NB: ``cohesion``, ``metallicity`` and ``site_potential`` are intentionally **excluded from**
    :meth:`structural_signature`. The signature seeds combination/measurement RNG, and all three
    are *always* a deterministic function of fields already hashed (an element's ``bond_energy`` /
    ``conduction_tendency`` / species electronegativity — which already shape ``occupied`` /
    ``atom_type`` — and the parents' fields + merge mask for a combination), so they add no
    independent entropy; excluding them keeps every M0–M8 seed and stored value byte-identical.
    The spectral measurement that *uses* ``site_potential`` is itself deterministic (``eigvalsh``,
    no RNG), so it needs no seed of its own.
    """

    occupied: np.ndarray
    atom_type: np.ndarray
    spin: np.ndarray
    mass: np.ndarray | None = None
    moment: np.ndarray | None = None
    cohesion: np.ndarray | None = None
    metallicity: np.ndarray | None = None
    site_potential: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mass is None:
            # Backward-compatible default: unit mass on occupied cells.
            object.__setattr__(self, "mass", self.occupied.astype(np.float32))
        if self.moment is None:
            # Backward-compatible default: unit moment on occupied cells (uniform coupling).
            object.__setattr__(self, "moment", self.occupied.astype(np.float32))
        if self.cohesion is None:
            # Backward-compatible default: uniform bond stiffness everywhere (plain lattice
            # gas — the keystone's substrate). Defined on empty cells too (site property).
            object.__setattr__(self, "cohesion", np.ones(self.occupied.shape, dtype=np.float32))
        if self.metallicity is None:
            # Backward-compatible default: metallic everywhere, so the charge backbone equals
            # the occupied set (M2 behaviour for constructed/synthetic lattices).
            object.__setattr__(
                self, "metallicity",
                np.full(self.occupied.shape, np.float32(DEFAULT_METALLICITY)),
            )
        if self.site_potential is None:
            # Backward-compatible default: zero on-site potential everywhere (no stagger -> no
            # band gap -> conductor). Every pre-M7 lattice takes this path and stays gap-free.
            object.__setattr__(
                self, "site_potential", np.zeros(self.occupied.shape, dtype=np.float32),
            )
        shapes = {
            self.occupied.shape,
            self.atom_type.shape,
            self.spin.shape,
            self.mass.shape,
            self.moment.shape,
            self.cohesion.shape,
            self.metallicity.shape,
            self.site_potential.shape,
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
            cohesion=self.cohesion.copy(),
            metallicity=self.metallicity.copy(),
            site_potential=self.site_potential.copy(),
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

    # --- cohesion: per-SITE bond stiffness from bond_energy -> melting point (M5). ---
    # Defined on every cell (a previously-empty site can fill when occupancy goes thermal),
    # so it is NOT masked by ``occupied``. Uniform for a root (one bond_energy); merge makes
    # it a domain blend. T_m ∝ cohesion², so this is what sets a material's melting point.
    cohesion_val = COH_LO + (COH_HI - COH_LO) * _clamp01(bond_energy)
    cohesion = np.full(shape, np.float32(cohesion_val), dtype=np.float32)

    # --- metallicity: per-site charge-carrier quality from conduction_tendency (M6b). ---
    # Uniform for a root; merge blends it per-domain. Charge conducts only through metallic
    # cells (see engine.properties.percolation), so a low-conduction element is an electrical
    # insulator yet still conducts heat (phonons) through its occupied matter.
    metallicity = np.full(shape, np.float32(_clamp01(conduction)), dtype=np.float32)

    return Lattice(
        occupied=occupied, atom_type=atom_type, spin=spin,
        mass=mass, moment=moment, cohesion=cohesion, metallicity=metallicity,
    )


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
    cohesion = np.where(take_a, a.cohesion, b.cohesion).astype(np.float32)
    metallicity = np.where(take_a, a.metallicity, b.metallicity).astype(np.float32)
    site_potential = np.where(take_a, a.site_potential, b.site_potential).astype(np.float32)
    return Lattice(
        occupied=occupied, atom_type=atom_type, spin=spin,
        mass=mass, moment=moment, cohesion=cohesion, metallicity=metallicity,
        site_potential=site_potential,
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


def metropolis_sweep(
    spin: np.ndarray,
    moment: np.ndarray,
    occ: np.ndarray,
    colors: tuple[np.ndarray, np.ndarray],
    *,
    coupling: float,
    temperature: float,
    field: float,
    gen: np.random.Generator,
) -> np.ndarray:
    """One deterministic checkerboard Metropolis sweep of the spin field.

    The single shared spin-update kernel: :func:`relax` (M3 settling) and
    :func:`engine.thermal.sample_ensemble` (M4 ensemble measurement) both call this, so the
    dynamics are defined in exactly one place. Returns the updated ``spin`` array.

    With ``J_ij = coupling·moment_i·moment_j`` and a uniform field ``H = field`` coupling as
    ``-H·Σ moment·spin``, flipping ``s_i`` changes the energy by
    ``dE_i = 2·moment_i·s_i·(coupling·h_i + H)`` where ``h_i = Σ_{j∈nbr} moment_j·s_j``.

    The bond and field terms are summed in the order ``coupling·h + field`` *expanded*
    (``2·coupling·s·m·h + 2·field·s·m``) so that the ``field == 0`` path is bit-for-bit the
    expression M3's :func:`relax` used — the determinism gate guards this. Each colour of the
    checkerboard updates simultaneously: no two updated cells are neighbours, so the
    vectorised flip equals a fixed sequential sweep (spec §6).
    """
    for color_mask in colors:
        h = _neighbor_sum(moment * spin)
        # Expanded so field==0 reproduces relax's original `2.0*coupling*spin*m*h` exactly
        # (adding 0.0 is an identity on finite floats); field!=0 adds the conjugate term.
        delta_e = 2.0 * coupling * spin * moment * h + 2.0 * field * spin * moment
        r = gen.random(spin.shape)
        # exponent capped at 0 so dE<=0 never overflows exp; those flip unconditionally.
        accept = (delta_e <= 0) | (r < np.exp(np.minimum(-delta_e / temperature, 0.0)))
        update = accept & color_mask & occ
        spin = np.where(update, -spin, spin).astype(np.int8)
    return spin


def checkerboard_colors(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The two-colour (parity-of-summed-coordinates) masks used by the spin update."""
    parity = np.indices(shape).sum(axis=0) % 2
    return (parity == 0, parity == 1)


def occupancy_sweep(
    occ: np.ndarray,
    cohesion: np.ndarray,
    colors: tuple[np.ndarray, np.ndarray],
    *,
    coupling: float,
    temperature: float,
    mu: float,
    gen: np.random.Generator,
) -> np.ndarray:
    """One deterministic checkerboard sweep of the **occupancy** field (M5 melting).

    The positional twin of :func:`metropolis_sweep`: instead of flipping spins it
    creates/annihilates atoms on a non-conserved **repulsive lattice gas**, the lattice model
    whose order-disorder transition *is* crystalline melting. With ``n_i ∈ {0,1}`` and energy

        E = Σ_⟨ij⟩ ε_ij n_i n_j  -  μ Σ_i n_i,     ε_ij = 4·coupling·cohesion_i·cohesion_j

    (the *repulsive* sign ε>0 makes neighbours want to alternate → a checkerboard crystal at
    half-filling; the 4 is the n=(s+1)/2 ↔ Ising mapping factor, :data:`_LATTICE_GAS_FACTOR`,
    so ``coupling`` is in Ising units). Flipping ``n_i`` changes the energy by

        dE_i = (1 - 2·n_i)·(4·coupling·cohesion_i·h_i  -  μ),   h_i = Σ_{j∈nbr} cohesion_j·n_j

    ``μ`` is the chemical potential — the conjugate of *amount of matter*, i.e. the pressure
    dial (M5 activates ``Conditions.pressure``). At the particle-hole-symmetric ``μ`` the mean
    density holds at ½ across the transition, so positional order is lost at *fixed* density —
    melting, not sublimation.

    Checkerboard parity guarantees no two simultaneously-flipped cells are neighbours, so the
    vectorised update equals a fixed sequential sweep; all acceptance randomness is from
    ``gen`` (spec §6). Returns the updated occupancy array (uint8 {0,1}).
    """
    coh = np.asarray(cohesion, dtype=np.float64)
    n = occ.astype(np.float64)
    for color_mask in colors:
        h = _neighbor_sum(coh * n)
        # dE for flipping n_i (delta = +1 if empty -> consider filling, -1 if occupied).
        delta = 1.0 - 2.0 * n
        delta_e = delta * (_LATTICE_GAS_FACTOR * coupling * coh * h - mu)
        r = gen.random(occ.shape)
        accept = (delta_e <= 0) | (r < np.exp(np.minimum(-delta_e / temperature, 0.0)))
        flip = accept & color_mask
        n = np.where(flip, 1.0 - n, n)
    return n.astype(np.uint8)


def xy_sweep(
    theta: np.ndarray,
    cond: np.ndarray,
    colors: tuple[np.ndarray, np.ndarray],
    *,
    coupling: float,
    temperature: float,
    gen: np.random.Generator,
    window: float = SC_PROPOSAL_WINDOW,
) -> np.ndarray:
    """One deterministic checkerboard Metropolis sweep of the continuous **XY phase** field (M6).

    The third sibling of :func:`metropolis_sweep` (Ising spins) and :func:`occupancy_sweep`
    (lattice gas): here each *conducting* cell carries a phase ``θ_i`` and bonds couple as
    ``-J·cos(θ_i − θ_j)`` across the conducting backbone. This is the substrate whose ordering
    *is* superconductivity — phase coherence of the order parameter.

    A move proposes ``θ_i' = θ_i + δ``, ``δ`` uniform in ``[−window, window]``, and accepts by
    Metropolis. Writing ``cos(θ_i−θ_j) = cosθ_i·cosθ_j + sinθ_i·sinθ_j``, the change for a flip
    is ``dE_i = −[(cosθ_i' − cosθ_i)·C_i + (sinθ_i' − sinθ_i)·S_i]·coupling`` where
    ``C_i = Σ_{j∈nbr} cosθ_j·cond_j`` and ``S_i = Σ_{j∈nbr} sinθ_j·cond_j`` — neighbour sums
    taken over *conducting* cells only, so the phase field lives on the backbone and ignores
    vacancies/insulating sites. Only conducting cells of the active colour update.

    Checkerboard parity makes the vectorised update equal a fixed sequential sweep; all
    proposal/acceptance randomness comes from ``gen`` (spec §6). Returns the updated phases.
    """
    cond_f = cond.astype(np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for color_mask in colors:
        C = _neighbor_sum(cos_t * cond_f)
        S = _neighbor_sum(sin_t * cond_f)
        prop = theta + (gen.random(theta.shape) * 2.0 - 1.0) * window
        cos_p = np.cos(prop)
        sin_p = np.sin(prop)
        delta_e = -coupling * ((cos_p - cos_t) * C + (sin_p - sin_t) * S)
        r = gen.random(theta.shape)
        accept = (delta_e <= 0) | (r < np.exp(np.minimum(-delta_e / temperature, 0.0)))
        update = accept & color_mask & (cond > 0)
        theta = np.where(update, prop, theta)
        cos_t = np.where(update, cos_p, cos_t)
        sin_t = np.where(update, sin_p, sin_t)
    return theta


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

    colors = checkerboard_colors(lattice.shape)

    # Settling is burn-in only: a fixed number of zero-field sweeps of the shared kernel.
    for _ in range(steps):
        spin = metropolis_sweep(
            spin, m, occ, colors,
            coupling=coupling, temperature=temperature, field=0.0, gen=gen,
        )

    spin = np.where(occ, spin, 1).astype(np.int8)
    return Lattice(
        occupied=lattice.occupied,
        atom_type=lattice.atom_type,
        spin=spin,
        mass=lattice.mass,
        moment=lattice.moment,
        cohesion=lattice.cohesion,
        metallicity=lattice.metallicity,
        site_potential=lattice.site_potential,
    )
