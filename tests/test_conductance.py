"""Unit tests for effective resistance + backbone redundancy (spec §5.3).

Hand-built structures pin the mechanism: a solid slab is a low-resistance, wide-bottleneck
conductor; a single filament spans but is a thin-bottleneck, high-resistance conductor; an
insulator has no spanning cluster. (Superconductivity is no longer a static flag here — it is
the M6 phase-coherence Tc measured in ``engine.thermal``; ``edge_connectivity`` is its
structural *input*. See ``tests/test_superconductivity.py``.)
"""

import numpy as np

from engine.lattice import Lattice
from engine.properties import conductance as C


def _lattice(occupied):
    occupied = np.asarray(occupied, dtype=np.uint8)
    atom_type = np.where(occupied == 1, 1, 0).astype(np.int8)
    spin = np.ones(occupied.shape, dtype=np.int8)
    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin)


def test_empty_lattice_has_no_conductance():
    lat = _lattice(np.zeros((16, 16)))
    assert C.effective_resistance(lat) >= C.RESISTANCE_INF
    assert C.conductivity(lat) == 0.0
    assert C.bottleneck_fraction(lat) == 0.0
    assert C.edge_connectivity(lat) == 0


def test_full_slab_is_low_resistance_wide_bottleneck():
    """A solid square slab: resistance ~1 square (ideal), bottleneck = full cross-section."""
    lat = _lattice(np.ones((16, 16)))
    r = C.effective_resistance(lat)
    assert 0.5 < r < 1.5  # ~ (L-1)/L, the sheet resistance of one square
    assert C.bottleneck_fraction(lat) == 1.0
    assert C.conductivity(lat) > 0.6


def test_single_filament_spans_but_chokes():
    """One conducting column spans top-to-bottom but has bottleneck width 1 and high R."""
    occ = np.zeros((16, 16))
    occ[:, 8] = 1
    lat = _lattice(occ)
    # Spans (finite resistance) ...
    assert C.effective_resistance(lat) < C.RESISTANCE_INF
    # ... but it is a thin filament: width-1 bottleneck and series resistance ~ length.
    assert C.bottleneck_fraction(lat) == 1.0 / 16
    assert C.effective_resistance(lat) > 10
    # A single filament has min-cut 1 (no redundancy) -- the weakest SC coupling input.
    assert C.edge_connectivity(lat) == 1


def test_two_parallel_filaments_double_the_bottleneck():
    occ = np.zeros((16, 16))
    occ[:, 4] = 1
    occ[:, 12] = 1
    lat = _lattice(occ)
    assert C.bottleneck_fraction(lat) == 2.0 / 16
    # Two parallel paths halve the resistance of one filament.
    one = np.zeros((16, 16)); one[:, 8] = 1
    assert C.effective_resistance(lat) < C.effective_resistance(_lattice(one))


def test_slab_has_full_width_edge_connectivity():
    """A solid slab is maximally redundant: min-cut = full cross-section (the SC coupling input)."""
    lat = _lattice(np.ones((24, 24)))
    m = C.measure(lat)
    assert m["edge_connectivity"] == 24          # full cross-section
    assert m["bottleneck_fraction"] == 1.0
    assert "superconductor" not in m             # the static SC flag is retired


def test_edge_connectivity_ranks_backbone_redundancy():
    """Slab (full width) > two filaments (2) > one filament (1) > checkerboard (0, no spanning)."""
    occ1 = np.zeros((16, 16)); occ1[:, 8] = 1
    occ2 = np.zeros((16, 16)); occ2[:, 4] = 1; occ2[:, 12] = 1
    checker = (np.indices((16, 16)).sum(0) % 2 == 0).astype(np.uint8)
    assert C.edge_connectivity(_lattice(np.ones((16, 16)))) == 16
    assert C.edge_connectivity(_lattice(occ2)) == 2
    assert C.edge_connectivity(_lattice(occ1)) == 1
    assert C.edge_connectivity(_lattice(checker)) == 0  # no face-connected spanning path


def test_resistance_is_deterministic():
    rng = np.random.default_rng(7)
    occ = (rng.random((32, 32)) < 0.7).astype(np.uint8)
    lat = _lattice(occ)
    assert C.effective_resistance(lat) == C.effective_resistance(lat)
    assert C.measure(lat) == C.measure(lat)


def test_measure_matches_individual_extractors():
    """The batched measure() must agree with the standalone extractors (no drift)."""
    rng = np.random.default_rng(2)
    occ = (rng.random((24, 24)) < 0.72).astype(np.uint8)
    lat = _lattice(occ)
    m = C.measure(lat)
    assert m["conductivity_continuous"] == C.conductivity(lat)
    assert m["bottleneck_fraction"] == C.bottleneck_fraction(lat)
    assert m["edge_connectivity"] == C.edge_connectivity(lat)
