"""Scalar properties: density / mass are legible and measured from the lattice (spec §5.1)."""

import numpy as np

from engine import elements
from engine.lattice import Lattice
from engine.material import from_element
from engine.properties import scalar


def _lattice(occupied: np.ndarray, mass_value: float) -> Lattice:
    occ = occupied.astype(np.uint8)
    return Lattice(
        occupied=occ,
        atom_type=occ.astype(np.int8),
        spin=np.ones(occ.shape, np.int8),
        mass=(occ.astype(np.float32) * np.float32(mass_value)),
    )


def test_density_of_full_lattice_equals_mass_per_cell():
    lat = _lattice(np.ones((8, 8)), 5.0)
    assert scalar.mean_atomic_mass(lat) == 5.0
    assert scalar.density(lat) == 5.0  # fill 1.0 * mean mass 5.0
    assert scalar.total_mass(lat) == 5.0 * 64


def test_density_scales_with_fill():
    occ = np.zeros((8, 8))
    occ[:4, :] = 1  # half filled
    lat = _lattice(occ, 5.0)
    assert lat.fill_fraction == 0.5
    assert scalar.mean_atomic_mass(lat) == 5.0  # mean over occupied only
    assert scalar.density(lat) == 2.5  # 0.5 * 5.0


def test_empty_lattice_has_zero_density():
    lat = _lattice(np.zeros((8, 8)), 5.0)
    assert scalar.mean_atomic_mass(lat) == 0.0
    assert scalar.density(lat) == 0.0


def test_root_atomic_mass_matches_element():
    for eid in ("hydrogen", "iron", "uranium"):
        el = elements.get(eid)
        mat = from_element(el, shape=(48, 48))
        assert abs(mat.properties["atomic_mass"] - el.atomic_mass) < 0.05, eid


def test_combination_mass_lies_between_parents():
    """A child's mean atomic mass should sit between its parents' (a blend)."""
    light = from_element(elements.get("lithium"), shape=(48, 48))   # ~6.9
    heavy = from_element(elements.get("uranium"), shape=(48, 48))   # ~238
    from engine.material import combine

    child = combine(light, heavy)
    lo = min(light.properties["atomic_mass"], heavy.properties["atomic_mass"])
    hi = max(light.properties["atomic_mass"], heavy.properties["atomic_mass"])
    assert lo <= child.properties["atomic_mass"] <= hi
