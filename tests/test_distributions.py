"""Population-level checks: the §7 verification, encoded as tests.

The point of the explorer (spec §7) is to confirm the property space looks the way the
design needs: conductivity bimodal around the percolation threshold, density a legible
spread. These tests pin that down so a future change can't silently flatten it.

Kept on a modest lattice / sample size so it runs in CI quickly; the threshold is still
clearly resolved at this size.
"""

import numpy as np

from engine.registry import Registry
from engine.rng import SplitMix64, mix
from tools.explorer import sample_population

SHAPE = (40, 40)
N = 120


def _population(universe_seed: int = 0):
    reg = Registry(universe_seed=universe_seed, shape=SHAPE)
    reg.seed_elements()
    rng = SplitMix64(mix(universe_seed, 0xD15C0))
    return sample_population(reg, N, rng)


def test_population_has_both_conductors_and_insulators():
    """The threshold must be *crossed* by the population, not sit entirely on one side."""
    children = _population()
    cond = np.array([c.properties["conductivity"] for c in children])
    frac = (cond >= 0.5).mean()
    assert 0.1 < frac < 0.95, f"conductivity not bimodal across population: {frac:.2f}"


def test_spanning_fraction_is_bimodal():
    """Insulators cluster near 0 spanning; conductors carry a large spanning cluster.

    The tell-tale of a percolation transition is an empty *middle*: few materials sit in
    the "barely spanning" region between the two modes.
    """
    children = _population()
    span = np.array([c.properties["spanning_fraction"] for c in children])
    low_mode = (span < 0.05).mean()      # clear insulators
    high_mode = (span > 0.3).mean()      # clear conductors
    middle = ((span >= 0.05) & (span <= 0.3)).mean()
    assert low_mode > 0.1, f"expected an insulating mode, got {low_mode:.2f}"
    assert high_mode > 0.1, f"expected a conducting mode, got {high_mode:.2f}"
    # The middle should be the *thinnest* region (the percolation gap).
    assert middle < low_mode and middle < high_mode, (
        f"no gap between modes: low={low_mode:.2f} mid={middle:.2f} high={high_mode:.2f}"
    )


def test_density_is_a_legible_spread():
    children = _population()
    dens = np.array([c.properties["density"] for c in children])
    assert dens.min() > 0
    # Heavy and light blends should differ by well over an order of magnitude.
    assert dens.max() / dens.min() > 10


def test_population_is_deterministic():
    """Same universe seed -> identical population (ids in order)."""
    a = [c.id for c in _population(0)]
    b = [c.id for c in _population(0)]
    assert a == b


def test_universe_seed_changes_population():
    a = {c.id for c in _population(0)}
    b = {c.id for c in _population(1)}
    assert a != b
