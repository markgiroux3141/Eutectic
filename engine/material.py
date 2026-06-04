"""The ``Material`` type and the ``combine()`` pipeline (spec §3.3, §4, §6).

A :class:`Material` is a settled lattice plus the properties *measured* from it and a
record of its lineage. :func:`combine` is the deterministic four-stage pipeline:

    hash -> merge -> relax -> measure          (then the registry caches it)

Determinism contract (spec §6) enforced here:

* parents are canonically ordered by id, so ``combine(A, B) == combine(B, A)`` (v1 is
  commutative, exactly two parents — spec §9.1);
* the seed is ``mix(A.signature, B.signature, UNIVERSE_SEED)`` with separate derived
  sub-seeds for merge vs. relax so the two stages never share a stream;
* measured properties are **quantized** before storage so float dust can't make two
  identical materials compare unequal;
* the material id is derived from the (sorted) parent ids + ``UNIVERSE_SEED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from . import lattice as lattice_mod
from . import process as process_mod
from . import thermal as thermal_mod
from .lattice import DEFAULT_SHAPE_2D, Lattice
from .process import Process
from .properties import conductance, ising, percolation, scalar
from .rng import UNIVERSE_SEED, hash_str, mix

if TYPE_CHECKING:  # avoid import cost / keep engine layering explicit
    from .elements import Element

# Quantize measured properties to this many decimals before storing (spec §6.4).
QUANT_DECIMALS: int = 4

# --- Curie temperature (M4): stored per-material, but gated for cost -------------------
# Tc is an ensemble measurement (a temperature sweep, docs §2/§4) — far more expensive than
# the snapshot extractors. A material only HAS a Curie point if it sustains ferromagnetic
# order at standard conditions, so we gate the sweep behind the (already-computed) reference
# magnetism: below the floor there is no ordered phase to lose -> Tc = 0 ("not ferromagnetic
# at standard conditions"), skipping the sweep entirely. The non-magnetic majority pay
# nothing; only ordered materials run the (lean) sweep. The explorer's `temperature-sweep`
# offers a higher-resolution curve on demand. Stored Tc is therefore an honest *coarse*
# value (the lean sweep's C(T)-peak grid point); 0.0 is the "no Curie point" sentinel.
CURIE_GATE_FLOOR: float = 0.30
# Lean sweep params for the stored value (the explorer uses the richer thermal defaults).
_CURIE_T_LO: float = 1.0          # gate guarantees order at standard T0 -> Tc >= T0
_CURIE_T_HI: float = 4.0          # covers dense high-moment couplings (J0*m^2 up to ~1.7)
_CURIE_N_TEMPS: int = 13          # grid step 0.25 -> coarse but legible
_CURIE_BURN_IN: int = 40
_CURIE_SAMPLES: int = 16
_CURIE_SAMPLE_EVERY: int = 1
_CURIE_SALT: int = 0x43           # 'C'

# Sub-seed salts so merge and relax draw from independent streams off the same base seed.
_MERGE_SALT = 0x4D  # 'M'
_RELAX_SALT = 0x52  # 'R'
# Roots are settled too (a material *is* a settled lattice, spec §1): magnetism must be
# measured from relaxed spins, or a down-biased non-magnet (copper) would read as ordered.
_ROOT_RELAX_SALT = 0x45  # 'E'


def quantize(x: float) -> float:
    """Round a measured property to fixed precision (spec §6.4)."""
    return round(float(x), QUANT_DECIMALS)


@dataclass(frozen=True)
class Material:
    """A settled material: its lattice, measured properties, and lineage (spec §3.3)."""

    id: str
    lattice: Lattice
    properties: dict[str, float]
    # (element_id,) for a root; (parent_a_id, parent_b_id) sorted for a combination.
    lineage: tuple[str, ...]

    @property
    def is_root(self) -> bool:
        return len(self.lineage) == 1


def net_spin(lattice: Lattice) -> float:
    """Mean spin over occupied cells (precursor to magnetism; full extractor in M3)."""
    occ = lattice.occupied == 1
    if not occ.any():
        return 0.0
    return float(lattice.spin[occ].mean())


def curie_temperature(lattice: Lattice, reference_magnetism: float) -> float:
    """Stored Curie point: the ``C(T)`` peak, gated by reference order (M4, docs §2/§4).

    Returns ``0.0`` when the material is not ferromagnetic at standard conditions
    (``reference_magnetism < CURIE_GATE_FLOOR``) — no ordered phase, no Curie point, and the
    expensive sweep is skipped. Otherwise runs the lean temperature sweep and returns the
    heat-capacity peak (the transition the order parameter collapses through). Deterministic:
    the sweep is seeded from the lattice signature + quantized conditions.

    ``reference_magnetism`` is passed in (rather than recomputed) so :func:`measure_properties`
    reuses the snapshot magnetism it already measured.
    """
    if reference_magnetism < CURIE_GATE_FLOOR:
        return 0.0
    tc = thermal_mod.curie_temperature(
        lattice,
        t_lo=_CURIE_T_LO, t_hi=_CURIE_T_HI, n_temps=_CURIE_N_TEMPS,
        order_floor=CURIE_GATE_FLOOR,
        burn_in=_CURIE_BURN_IN, n_samples=_CURIE_SAMPLES,
        sample_every=_CURIE_SAMPLE_EVERY,
    )
    return 0.0 if tc is None else tc


def measure_properties(lattice: Lattice) -> dict[str, float]:
    """Run the available extractors on a settled lattice and quantize (spec §4.4, §6.4).

    M2 surfaces the legible scalars (``density``, ``atomic_mass``, ``fill_fraction``) and
    the percolation/conductivity family. M3 adds magnetism (Ising) and the
    resistance/superconductivity family (Laplacian + min-cut). M4 adds the Curie temperature
    ``curie_temperature`` — the first *condition-dependent* property, measured as an ensemble
    observable (a temperature sweep) rather than read off the snapshot (docs §2). Every value
    is measured from the structure, never assigned. M5+ extend further.
    """
    magnetism = ising.magnetism(lattice)
    props = {
        "fill_fraction": quantize(lattice.fill_fraction),
        "atomic_mass": quantize(scalar.mean_atomic_mass(lattice)),
        "density": quantize(scalar.density(lattice)),
        "conductivity": quantize(float(percolation.conductivity_boolean(lattice))),
        "spanning_fraction": quantize(percolation.spanning_fraction(lattice)),
        "largest_cluster_fraction": quantize(percolation.largest_cluster_fraction(lattice)),
        "net_spin": quantize(net_spin(lattice)),
        "magnetism": quantize(magnetism),
        "curie_temperature": quantize(curie_temperature(lattice, magnetism)),
    }
    # Resistance / superconductivity: one per-axis solve, several derived values (§5.3-5.4).
    for key, value in conductance.measure(lattice).items():
        props[key] = quantize(value)
    return props


def derive_id(parent_ids: Sequence[str], universe_seed: int = UNIVERSE_SEED) -> str:
    """Material id from canonically-ordered parent ids + universe seed (spec §6.5)."""
    ordered = sorted(parent_ids)
    h = mix(*(hash_str(p) for p in ordered), universe_seed)
    return f"m_{h:016x}"


def _settle(
    lattice: Lattice, seed: int, *, relax_steps: int, process: Process | None
) -> Lattice:
    """Settle a lattice's spins, either via the default relax or an explicit process.

    ``process is None`` runs today's fixed-temperature :func:`engine.lattice.relax`
    (unchanged production path — byte-identical, so M0-M4 stay green). A supplied
    :class:`~engine.process.Process` instead runs that synthesis trajectory
    (:func:`engine.process.run_process`), with the process signature folded into the seed so
    different processes give distinct, deterministic outcomes from the same structure.
    """
    if process is None:
        return lattice_mod.relax(lattice, seed, steps=relax_steps)
    return process_mod.run_process(lattice, process, mix(seed, process.signature()))


def from_element(
    element: "Element",
    *,
    shape: Sequence[int] = DEFAULT_SHAPE_2D,
    universe_seed: int = UNIVERSE_SEED,
    process: Process | None = None,
) -> Material:
    """Build a root material from an element (its id *is* the element id).

    With ``process`` supplied, the root is synthesised along that trajectory (anneal/quench/
    field-cool) rather than the default settle — letting even a root's microstructure be
    process-determined (docs: process layer).
    """
    lat = element.lattice(shape=shape, universe_seed=universe_seed)
    # Settle the spins before measuring (spec §1, §4.3): a root is a settled lattice too.
    relax_seed = mix(lat.structural_signature(), universe_seed, _ROOT_RELAX_SALT)
    lat = _settle(lat, relax_seed, relax_steps=lattice_mod.RELAX_STEPS, process=process)
    return Material(
        id=element.id,
        lattice=lat,
        properties=measure_properties(lat),
        lineage=(element.id,),
    )


def combine(
    a: Material,
    b: Material,
    *,
    universe_seed: int = UNIVERSE_SEED,
    relax_steps: int = lattice_mod.RELAX_STEPS,
    process: Process | None = None,
) -> Material:
    """Deterministically combine two materials into a child (spec §4).

    Pure function of ``(a, b, universe_seed, relax_steps, process)``. With ``process`` given,
    the child is synthesised along that trajectory instead of the default settle — the same
    ingredients can yield different materials depending on how they're processed (the process
    layer). Caching is the registry's job (spec §4.5); this stays side-effect-free.
    """
    if a.lattice.shape != b.lattice.shape:
        raise ValueError(
            f"cannot combine materials with different lattice shapes: "
            f"{a.lattice.shape} vs {b.lattice.shape}"
        )

    # Canonical order -> commutativity (spec §4.1, §9.1).
    lo, hi = (a, b) if a.id <= b.id else (b, a)

    base_seed = mix(
        lo.lattice.structural_signature(),
        hi.lattice.structural_signature(),
        universe_seed,
    )
    child = lattice_mod.merge(lo.lattice, hi.lattice, mix(base_seed, _MERGE_SALT))
    child = _settle(child, mix(base_seed, _RELAX_SALT), relax_steps=relax_steps, process=process)

    lineage = (lo.id, hi.id)
    return Material(
        id=derive_id(lineage, universe_seed),
        lattice=child,
        properties=measure_properties(child),
        lineage=lineage,
    )
