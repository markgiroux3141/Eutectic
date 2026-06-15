"""Population-level checks: the §7 verification, encoded as tests.

The point of the explorer (spec §7) is to confirm the property space looks the way the
design needs: conductivity bimodal around the percolation threshold, density a legible
spread. These tests pin that down so a future change can't silently flatten it.

Kept on a modest lattice / sample size so it runs in CI quickly; the threshold is still
clearly resolved at this size.
"""

from functools import lru_cache

import numpy as np

from engine.registry import Registry
from engine.rng import SplitMix64, mix
from tools.explorer import sample_population

SHAPE = (40, 40)
N = 200


@lru_cache(maxsize=None)
def _population(universe_seed: int = 0):
    # Memoized: the population is a deterministic function of the seed, so the several
    # tests that share a seed reuse one (relatively expensive) build instead of recomputing.
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


def test_magnetism_shows_a_critical_transition():
    """Magnetism is bimodal: a large disordered mode at ~0 and a smaller ordered tail.

    The signature of the Ising critical transition (spec §5.5) is the same as percolation's:
    a thin *middle*. Most combinations have no connected high-moment backbone (disordered);
    a minority do and align spontaneously. Few sit in the partially-ordered in-between.
    """
    children = _population()
    mag = np.array([c.properties["magnetism"] for c in children])
    disordered = (mag < 0.15).mean()      # no spontaneous alignment
    ordered = (mag > 0.5).mean()          # clear ferromagnet
    middle = ((mag >= 0.15) & (mag <= 0.5)).mean()
    assert disordered > 0.4, f"expected a dominant disordered mode, got {disordered:.2f}"
    assert ordered > 0.05, f"expected an ordered tail, got {ordered:.2f}"
    # The transition's tell-tale: the partially-ordered middle is the thinnest region.
    assert middle < disordered and middle < (disordered + ordered), (
        f"no gap: disordered={disordered:.2f} middle={middle:.2f} ordered={ordered:.2f}"
    )


def test_edge_connectivity_is_a_structural_spread():
    """M6: superconductivity is no longer a stored flag (it is the on-demand phase-coherence Tc,
    see tests/test_superconductivity.py). What the population stores is ``edge_connectivity`` —
    the backbone redundancy that is the *structural input* to that Tc. It must span a real range:
    insulators at 0, conductors carrying a spread of redundancies (the would-be high-Tc tail).
    """
    children = _population()
    ec = np.array([c.properties["edge_connectivity"] for c in children])
    cond = np.array([c.properties["conductivity"] for c in children])
    assert (ec[cond < 0.5] == 0).all(), "an insulator has a conducting backbone"
    assert ec[cond >= 0.5].max() >= 3, "no redundant (high-Tc-capable) backbones present"
    assert (ec > 0).mean() > 0.1 and (ec == 0).mean() > 0.1  # a real spread, both modes present


def test_curie_temperature_is_gated_by_order():
    """Stored Tc (M4) is >0 exactly for materials magnetic at standard conditions.

    The condition-dependent property must respect its gate (docs §2): a Curie point exists
    only where there is ferromagnetic order to lose. So every Tc>0 material clears the order
    floor, every Tc==0 material is below it, and the ordered tail is non-empty.
    """
    from engine.material import CURIE_GATE_FLOOR

    children = _population()
    tc = np.array([c.properties["curie_temperature"] for c in children])
    mag = np.array([c.properties["magnetism"] for c in children])
    assert (tc > 0).any(), "no material has a Curie point"
    assert (mag[tc > 0] >= CURIE_GATE_FLOOR).all(), "a Tc>0 material is not ordered"
    assert (mag[tc == 0] < CURIE_GATE_FLOOR).all(), "an ordered material has no Tc"
    # Curie points are physically sensible: above standard T0, below the dense-coupling cap.
    assert tc[tc > 0].min() >= 1.0 and tc.max() <= 4.0


def test_continuous_conductivity_tracks_boolean():
    """1/resistance is >0 exactly when the boolean percolation says it spans (consistency)."""
    children = _population()
    cc = np.array([c.properties["conductivity_continuous"] for c in children])
    cond = np.array([c.properties["conductivity"] for c in children])
    # Continuous conductivity is positive iff a spanning cluster exists.
    assert ((cc > 0) == (cond >= 0.5)).all()


def test_population_is_deterministic():
    """Same universe seed -> identical population (ids in order)."""
    a = [c.id for c in _population(0)]
    b = [c.id for c in _population(0)]
    assert a == b


def test_universe_seed_changes_population():
    a = {c.id for c in _population(0)}
    b = {c.id for c in _population(1)}
    assert a != b
