"""Unit tests for Ising magnetism (spec §5.5).

These pin the *mechanism* on hand-built lattices: a strongly-coupled aligned lattice keeps
its order, a zero-moment lattice cannot order, and measurement is moment-weighted and
deterministic. The population-level critical transition is asserted in
``test_distributions.py``.
"""

import numpy as np

from engine.lattice import EXCHANGE_J0, Lattice, relax
from engine.properties import ising


def _lattice(occupied, spin, moment):
    occupied = np.asarray(occupied, dtype=np.uint8)
    atom_type = np.where(occupied == 1, 1, 0).astype(np.int8)
    spin = np.asarray(spin, dtype=np.int8)
    moment = np.asarray(moment, dtype=np.float32)
    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin, moment=moment)


def test_fully_aligned_high_moment_is_magnetic():
    """A solid, aligned, high-moment lattice reads magnetism ~1 (no relaxation needed)."""
    occ = np.ones((16, 16))
    lat = _lattice(occ, np.ones((16, 16)), np.full((16, 16), 1.3))
    assert ising.magnetism(lat) == 1.0


def test_antialigned_halves_cancel():
    """Two equal high-moment domains pointing opposite ways cancel to ~0 net magnetism."""
    spin = np.ones((16, 16), dtype=np.int8)
    spin[:, 8:] = -1  # half up, half down
    lat = _lattice(np.ones((16, 16)), spin, np.full((16, 16), 1.3))
    assert ising.magnetism(lat) < 1e-6


def test_strong_coupling_preserves_order_through_relax():
    """High moment -> strong coupling -> an aligned start stays ordered after relaxing."""
    occ = np.ones((24, 24))
    lat = _lattice(occ, np.ones((24, 24)), np.full((24, 24), 1.3))
    settled = relax(lat, seed=11)
    assert ising.magnetism(settled) > 0.9


def test_zero_moment_cannot_order():
    """Zero moment -> zero coupling -> spins never feel each other; relaxation can't order.

    Starting from a random (near-balanced) spin field, a decoupled lattice stays disordered.
    """
    gen = np.random.default_rng(0)
    spin = np.where(gen.random((32, 32)) < 0.5, 1, -1).astype(np.int8)
    lat = _lattice(np.ones((32, 32)), spin, np.zeros((32, 32)))
    settled = relax(lat, seed=3)
    assert ising.magnetism(settled) < 0.15


def test_magnetism_is_moment_weighted():
    """Low-moment filler neither contributes to nor dilutes the magnetization.

    A small block of aligned high-moment cells embedded in a sea of disordered low-moment
    cells should still read near-fully magnetic.
    """
    spin = np.where(np.indices((20, 20)).sum(0) % 2 == 0, 1, -1).astype(np.int8)  # checkerboard
    moment = np.full((20, 20), 0.02, dtype=np.float32)  # weak filler everywhere
    # A coherent aligned high-moment block in the corner.
    spin[:6, :6] = 1
    moment[:6, :6] = 1.3
    lat = _lattice(np.ones((20, 20)), spin, moment)
    # The block dominates the moment-weighted sum; checkerboard filler ~cancels. A plain
    # spin-mean would read only ~0.09 here — moment-weighting is what lifts it near 1.
    assert ising.magnetism(lat) > 0.8
    assert abs(float(spin.mean())) < 0.2  # contrast: unweighted magnetization is small


def test_relax_with_moment_is_deterministic():
    occ = np.ones((20, 20))
    spin = np.where(np.random.default_rng(1).random((20, 20)) < 0.6, 1, -1).astype(np.int8)
    moment = np.full((20, 20), 1.1, dtype=np.float32)
    lat = _lattice(occ, spin, moment)
    a = relax(lat, seed=99)
    b = relax(lat, seed=99)
    assert np.array_equal(a.spin, b.spin)
    assert ising.magnetism(a) == ising.magnetism(b)


def test_default_moment_recovers_uniform_coupling():
    """Omitting moment defaults to unit moment -> ordinary (uniform-coupling) Ising still orders."""
    occ = np.ones((20, 20), dtype=np.uint8)
    atom_type = occ.copy().astype(np.int8)
    spin = np.ones((20, 20), dtype=np.int8)
    lat = Lattice(occupied=occ, atom_type=atom_type, spin=spin)  # no moment given
    assert np.allclose(lat.moment, occ)  # default applied
    settled = relax(lat, seed=5)
    assert ising.magnetism(settled) > 0.9


def test_exchange_constant_exposed():
    """The global exchange constant is a real number the coupling scales with."""
    assert EXCHANGE_J0 > 0
