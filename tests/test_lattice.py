"""Lattice generation determinism + invariants (spec §3.2, §4.1, §6)."""

import numpy as np
import pytest

from engine.lattice import Lattice, generate_base


def test_generate_base_is_deterministic_2d():
    a = generate_base(123, shape=(64, 64), affinities={"bond_energy": 0.6})
    b = generate_base(123, shape=(64, 64), affinities={"bond_energy": 0.6})
    assert np.array_equal(a.occupied, b.occupied)
    assert np.array_equal(a.atom_type, b.atom_type)
    assert np.array_equal(a.spin, b.spin)
    assert a.structural_signature() == b.structural_signature()


def test_generate_base_is_deterministic_3d():
    a = generate_base(7, shape=(16, 16, 16))
    b = generate_base(7, shape=(16, 16, 16))
    assert a.dim == 3
    assert np.array_equal(a.atom_type, b.atom_type)
    assert a.structural_signature() == b.structural_signature()


def test_different_seed_changes_lattice():
    a = generate_base(1, shape=(32, 32))
    b = generate_base(2, shape=(32, 32))
    assert a.structural_signature() != b.structural_signature()


def test_empty_cells_have_atom_type_zero():
    lat = generate_base(5, shape=(48, 48))
    empty = lat.occupied == 0
    assert np.all(lat.atom_type[empty] == 0)
    occupied = lat.occupied == 1
    assert np.all(lat.atom_type[occupied] >= 1)


def test_higher_bond_energy_means_higher_fill():
    low = generate_base(11, shape=(64, 64), affinities={"bond_energy": 0.1})
    high = generate_base(11, shape=(64, 64), affinities={"bond_energy": 0.9})
    assert high.fill_fraction > low.fill_fraction


def test_magnetic_tendency_biases_net_spin():
    """High magnetic_tendency -> spins lean +1 on occupied cells."""
    lat = generate_base(3, shape=(64, 64), affinities={"magnetic_tendency": 0.98})
    occ = lat.occupied == 1
    assert lat.spin[occ].mean() > 0.5


def test_spin_values_are_pm_one():
    lat = generate_base(9, shape=(32, 32))
    assert set(np.unique(lat.spin).tolist()) <= {-1, 1}


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        Lattice(
            occupied=np.zeros((4, 4), np.uint8),
            atom_type=np.zeros((4, 5), np.int8),
            spin=np.ones((4, 4), np.int8),
        )


def test_unsupported_dimension_rejected():
    with pytest.raises(ValueError):
        generate_base(1, shape=(8,))
    with pytest.raises(ValueError):
        generate_base(1, shape=(4, 4, 4, 4))


def test_copy_is_independent():
    lat = generate_base(2, shape=(16, 16))
    clone = lat.copy()
    clone.occupied[0, 0] ^= 1
    assert not np.array_equal(lat.occupied, clone.occupied)
