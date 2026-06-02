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
from .lattice import DEFAULT_SHAPE_2D, Lattice
from .properties import percolation, scalar
from .rng import UNIVERSE_SEED, hash_str, mix

if TYPE_CHECKING:  # avoid import cost / keep engine layering explicit
    from .elements import Element

# Quantize measured properties to this many decimals before storing (spec §6.4).
QUANT_DECIMALS: int = 4

# Sub-seed salts so merge and relax draw from independent streams off the same base seed.
_MERGE_SALT = 0x4D  # 'M'
_RELAX_SALT = 0x52  # 'R'


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


def measure_properties(lattice: Lattice) -> dict[str, float]:
    """Run the available extractors on a settled lattice and quantize (spec §4.4, §6.4).

    M2 surfaces the legible scalars (``density``, ``atomic_mass``, ``fill_fraction``) and
    the percolation/conductivity family. M3+ extend this dict (magnetism, band gap,
    mechanical) — every value is measured from the lattice, never assigned.
    """
    return {
        "fill_fraction": quantize(lattice.fill_fraction),
        "atomic_mass": quantize(scalar.mean_atomic_mass(lattice)),
        "density": quantize(scalar.density(lattice)),
        "conductivity": quantize(float(percolation.conductivity_boolean(lattice))),
        "spanning_fraction": quantize(percolation.spanning_fraction(lattice)),
        "largest_cluster_fraction": quantize(percolation.largest_cluster_fraction(lattice)),
        "net_spin": quantize(net_spin(lattice)),
    }


def derive_id(parent_ids: Sequence[str], universe_seed: int = UNIVERSE_SEED) -> str:
    """Material id from canonically-ordered parent ids + universe seed (spec §6.5)."""
    ordered = sorted(parent_ids)
    h = mix(*(hash_str(p) for p in ordered), universe_seed)
    return f"m_{h:016x}"


def from_element(
    element: "Element",
    *,
    shape: Sequence[int] = DEFAULT_SHAPE_2D,
    universe_seed: int = UNIVERSE_SEED,
) -> Material:
    """Build a root material from an element (its id *is* the element id)."""
    lat = element.lattice(shape=shape, universe_seed=universe_seed)
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
) -> Material:
    """Deterministically combine two materials into a child (spec §4).

    Pure function of ``(a, b, universe_seed, relax_steps)``. Caching is the registry's
    job (spec §4.5); this stays side-effect-free so it is trivially testable.
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
    child = lattice_mod.relax(child, mix(base_seed, _RELAX_SALT), steps=relax_steps)

    lineage = (lo.id, hi.id)
    return Material(
        id=derive_id(lineage, universe_seed),
        lattice=child,
        properties=measure_properties(child),
        lineage=lineage,
    )
