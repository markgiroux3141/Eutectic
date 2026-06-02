"""Percolation extractor: correctness, determinism, and threshold behavior (spec §5.2).

The threshold test is the important one — it's the cheapest falsification of the project's
core rarity claim (spec §1): a spanning cluster must appear *suddenly* around the critical
fill density, not gradually.
"""

import numpy as np

from engine.lattice import Lattice
from engine.properties import percolation
from engine.rng import SplitMix64, mix


def _lattice_at_fill(seed: int, shape, fill: float) -> Lattice:
    gen = SplitMix64(seed).numpy_generator()
    occupied = (gen.random(shape) < fill).astype(np.uint8)
    atom_type = np.where(occupied == 1, 1, 0).astype(np.int8)
    spin = np.ones(shape, dtype=np.int8)
    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin)


def _full(shape) -> Lattice:
    occ = np.ones(shape, np.uint8)
    return Lattice(occupied=occ, atom_type=occ.astype(np.int8), spin=occ.astype(np.int8))


def _empty(shape) -> Lattice:
    occ = np.zeros(shape, np.uint8)
    return Lattice(occupied=occ, atom_type=occ.astype(np.int8), spin=np.ones(shape, np.int8))


def test_full_lattice_spans_and_empty_does_not():
    full = _full((16, 16))
    assert percolation.spans(full) is True
    assert percolation.conductivity_boolean(full) is True
    assert percolation.spanning_fraction(full) == 1.0
    assert percolation.largest_cluster_fraction(full) == 1.0

    empty = _empty((16, 16))
    assert percolation.spans(empty) is False
    assert percolation.conductivity_boolean(empty) is False
    assert percolation.spanning_fraction(empty) == 0.0
    assert percolation.largest_cluster_fraction(empty) == 0.0


def test_single_vertical_wire_spans_axis0_only():
    occ = np.zeros((8, 8), np.uint8)
    occ[:, 3] = 1  # a vertical line: spans axis 0 (rows) but not axis 1 (cols)
    lat = Lattice(occupied=occ, atom_type=occ.astype(np.int8), spin=np.ones((8, 8), np.int8))
    assert percolation.spans(lat, axis=0) is True
    assert percolation.spans(lat, axis=1) is False
    assert percolation.percolates_any_axis(lat) is True


def test_diagonal_does_not_percolate_face_connectivity():
    # A diagonal stripe is connected only diagonally -> NOT face-connected, must not span.
    n = 8
    occ = np.eye(n, dtype=np.uint8)
    lat = Lattice(occupied=occ, atom_type=occ.astype(np.int8), spin=np.ones((n, n), np.int8))
    # Each diagonal cell is its own cluster under face connectivity.
    _, n_clusters = percolation.label_clusters(occ == 1)
    assert n_clusters == n
    assert percolation.spans(lat, axis=0) is False


def test_spanning_is_deterministic():
    lat_a = _lattice_at_fill(123, (64, 64), 0.6)
    lat_b = _lattice_at_fill(123, (64, 64), 0.6)
    assert percolation.spanning_fraction(lat_a) == percolation.spanning_fraction(lat_b)


def _p_span(fill: float, shape=(64, 64), trials: int = 30) -> float:
    spanned = 0
    for t in range(trials):
        seed = mix(int(fill * 1e6), t, 0)
        if percolation.percolates_any_axis(_lattice_at_fill(seed, shape, fill)):
            spanned += 1
    return spanned / trials


def test_threshold_is_sharp_around_pc():
    """The core claim: low spanning probability below p_c, high above (spec §1, §5.2)."""
    well_below = _p_span(0.45)
    well_above = _p_span(0.70)
    assert well_below < 0.2, f"expected insulating below p_c, got P(span)={well_below}"
    assert well_above > 0.8, f"expected conducting above p_c, got P(span)={well_above}"
    # And the transition is monotone-ish across the critical point.
    assert well_above - well_below > 0.6
