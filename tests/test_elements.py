"""Root element table integrity + deterministic lattice generation (spec §3.1)."""

import numpy as np
import pytest

from engine import elements


def test_table_has_enough_elements():
    # Spec §3.1 asks for ~15-30 root elements to start.
    assert 15 <= len(elements.ELEMENTS) <= 40


def test_all_ids_sorted_and_unique():
    ids = elements.all_ids()
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_signatures_are_unique():
    sigs = [el.signature for el in elements.ELEMENTS.values()]
    assert len(sigs) == len(set(sigs))


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        elements.get("unobtainium")


def test_element_lattice_is_deterministic():
    iron = elements.get("iron")
    a = iron.lattice(shape=(64, 64))
    b = iron.lattice(shape=(64, 64))
    assert a.structural_signature() == b.structural_signature()
    assert np.array_equal(a.atom_type, b.atom_type)


def test_universe_seed_changes_material_space():
    iron = elements.get("iron")
    a = iron.lattice(shape=(64, 64), universe_seed=0)
    b = iron.lattice(shape=(64, 64), universe_seed=1)
    assert a.structural_signature() != b.structural_signature()


def test_distinct_elements_distinct_lattices():
    a = elements.get("iron").lattice(shape=(64, 64))
    b = elements.get("copper").lattice(shape=(64, 64))
    assert a.structural_signature() != b.structural_signature()


def test_affinities_in_unit_range():
    for el in elements.ELEMENTS.values():
        for key in ("bond_energy", "magnetic_tendency", "conduction_tendency"):
            assert 0.0 <= el.base_affinities[key] <= 1.0, (el.id, key)


def test_ferromagnets_lean_magnetic():
    """Iron/cobalt/nickel should start with a clear net spin alignment."""
    for eid in ("iron", "cobalt", "nickel"):
        lat = elements.get(eid).lattice(shape=(64, 64))
        occ = lat.occupied == 1
        assert lat.spin[occ].mean() > 0.3, eid
