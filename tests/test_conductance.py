"""Unit tests for effective resistance + superconductivity (spec §5.3, §5.4).

Hand-built structures pin the mechanism: a solid slab is a low-resistance, wide-bottleneck
conductor; a single filament spans but is a thin-bottleneck, high-resistance non-conductor;
an insulator has no spanning cluster. The double threshold and its determinism are checked
here; the population's rare *tail* is asserted in ``test_distributions.py``.
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
    assert not C.is_superconductor(lat)


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
    # A single filament is emphatically NOT a superconductor (fails the double threshold).
    assert not C.is_superconductor(lat)


def test_two_parallel_filaments_double_the_bottleneck():
    occ = np.zeros((16, 16))
    occ[:, 4] = 1
    occ[:, 12] = 1
    lat = _lattice(occ)
    assert C.bottleneck_fraction(lat) == 2.0 / 16
    # Two parallel paths halve the resistance of one filament.
    one = np.zeros((16, 16)); one[:, 8] = 1
    assert C.effective_resistance(lat) < C.effective_resistance(_lattice(one))


def test_slab_is_superconductor_k_connected():
    """A solid slab is highly edge-connected (min-cut = full width) -> superconductor."""
    lat = _lattice(np.ones((24, 24)))
    assert C.is_superconductor(lat)
    m = C.measure(lat)
    assert m["superconductor"] == 1.0
    # Edge-connectivity is the full cross-section, far above the required k.
    assert m["edge_connectivity"] == 24
    assert m["edge_connectivity"] >= C.required_connectivity(lat.shape)


def test_required_connectivity_scales_with_width_and_floors_at_two():
    """k scales with lattice width but never drops below 2 ('no single bottleneck edge')."""
    assert C.required_connectivity((64, 64)) == round(0.17 * 64)
    assert C.required_connectivity((40, 40)) == round(0.17 * 40)
    assert C.required_connectivity((4, 4)) == 2  # floor


def test_filament_is_not_superconductor_but_is_a_conductor():
    """A single filament spans (conductor) yet is min-cut 1 -> fails k-connectivity."""
    occ = np.zeros((16, 16)); occ[:, 8] = 1
    lat = _lattice(occ)
    assert C.conductivity(lat) > 0          # it conducts (spans)
    assert C.edge_connectivity(lat) == 1     # ... through a single path
    assert not C.is_superconductor(lat)      # so it is not loss-free


def test_superconductor_implies_spanning():
    """The double threshold: anything flagged superconducting must also conduct (span)."""
    from engine.properties import percolation

    # A solid slab (superconductor) and a checkerboard (no face connectivity) bracket it.
    slab = _lattice(np.ones((20, 20)))
    assert C.is_superconductor(slab) and percolation.conductivity_boolean(slab)
    checker = _lattice((np.indices((20, 20)).sum(0) % 2 == 0).astype(np.uint8))
    assert not C.is_superconductor(checker)


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
    assert bool(m["superconductor"]) == C.is_superconductor(lat)
