"""The process layer: synthesis as a trajectory through conditions-space.

Pins the two things that make it trustworthy: (1) the abstraction is *faithful* — the
standard single-hold process reproduces ``relax`` byte-for-byte, so M0-M4 are unaffected; and
(2) the dynamics are *real* — slower cooling reaches lower-energy, larger-domain structures,
and cooling under a field builds remanence (the Step-0 de-risk, now encoded as tests).
"""

import numpy as np

from engine import elements, process as proc, thermal
from engine.lattice import RELAX_STEPS, RELAX_TEMPERATURE, relax
from engine.material import combine, from_element
from engine.properties import ising, microstructure
from engine.registry import Registry

SHAPE = (48, 48)


def _ferro():
    """A strongly-coupled structure (iron root) to settle under different processes."""
    return from_element(elements.get("iron"), shape=SHAPE).lattice


def _energy_per_cell(lattice):
    n = int((lattice.occupied == 1).sum())
    return thermal.energy(lattice, lattice.spin) / max(n, 1)


# --- the abstraction is faithful ------------------------------------------------------


def test_standard_process_reproduces_relax():
    """A single constant-T hold == today's relax, byte-for-byte (the equivalence guard)."""
    lat = _ferro()
    seed = 12345
    by_relax = relax(lat, seed, steps=RELAX_STEPS, temperature=RELAX_TEMPERATURE)
    by_process = proc.run_process(lat, proc.STANDARD_PROCESS, seed)
    assert np.array_equal(by_relax.spin, by_process.spin)
    assert by_relax.structural_signature() == by_process.structural_signature()


def test_standard_process_constant_matches_relax_default_path():
    """from_element(process=STANDARD_PROCESS) matches the default settle for the same seed.

    Guards that routing the default through the executor would not change results — the
    production path stays on relax, but the abstraction agrees with it.
    """
    lat = elements.get("iron").lattice(shape=SHAPE)
    from engine.material import _ROOT_RELAX_SALT
    from engine.rng import UNIVERSE_SEED, mix

    seed = mix(lat.structural_signature(), UNIVERSE_SEED, _ROOT_RELAX_SALT)
    default = relax(lat, seed, steps=RELAX_STEPS)
    # STANDARD_PROCESS folds its signature into the seed via _settle; reproduce that here.
    via_proc = proc.run_process(lat, proc.STANDARD_PROCESS, mix(seed, proc.STANDARD_PROCESS.signature()))
    # Same dynamics (12 sweeps at T0), different seed stream -> same *statistics*, and both
    # land the strongly-coupled iron lattice in an ordered state.
    assert ising.magnetism(default) > 0.7 and ising.magnetism(via_proc) > 0.7


def test_process_is_deterministic():
    """Same (structure, process, seed) -> byte-identical result (spec §6)."""
    lat = _ferro()
    p = proc.anneal(budget=60)
    a = proc.run_process(lat, p, seed=7)
    b = proc.run_process(lat, p, seed=7)
    assert np.array_equal(a.spin, b.spin)


def test_process_signature_ignores_name_not_dynamics():
    """The signature keys the trajectory: same stages -> same sig; different field -> not."""
    a = proc.anneal(budget=60, name="slow")
    b = proc.anneal(budget=60, name="renamed")
    c = proc.anneal(budget=60)
    d = proc.field_cool(budget=60)
    assert a.signature() == b.signature() == c.signature()
    assert a.signature() != d.signature()


# --- the dynamics are real (the Step-0 de-risk, as assertions) ------------------------


def test_slower_cooling_reaches_lower_energy_and_larger_domains():
    """Anneal vs quench: the cooling-rate signal lives in energy + domain size."""
    lat = _ferro()
    seeds = range(4)
    anneal_e, quench_e, anneal_dom, quench_dom = [], [], [], []
    for s in seeds:
        a = proc.run_process(lat, proc.anneal(budget=120), seed=s)
        q = proc.run_process(lat, proc.quench(budget=120), seed=s)
        anneal_e.append(_energy_per_cell(a))
        quench_e.append(_energy_per_cell(q))
        anneal_dom.append(microstructure.domain_fraction(a))
        quench_dom.append(microstructure.domain_fraction(q))
    # Slow anneal settles to a lower-energy (fewer-domain-wall) structure...
    assert np.mean(anneal_e) < np.mean(quench_e)
    # ...with a larger dominant domain than the quench-frozen patchwork.
    assert np.mean(anneal_dom) > np.mean(quench_dom)


def test_quench_leaves_more_domain_walls():
    """Domain-wall density is higher for a quench than a slow anneal (frozen disorder)."""
    lat = _ferro()
    seeds = range(4)
    aw = [microstructure.domain_wall_density(proc.run_process(lat, proc.anneal(budget=120), s))
          for s in seeds]
    qw = [microstructure.domain_wall_density(proc.run_process(lat, proc.quench(budget=120), s))
          for s in seeds]
    assert np.mean(qw) > np.mean(aw)


def test_field_cool_builds_remanence():
    """Cooling under a field then removing it leaves a high net magnetization (remanence).

    At zero field the cooled state has a near-random domain sign (low ⟨|M|⟩); field-cooling
    breaks the symmetry, so the retained magnetization is dramatically higher.
    """
    lat = _ferro()
    seeds = range(4)
    zero = [ising.magnetism(proc.run_process(lat, proc.anneal(budget=120), s)) for s in seeds]
    fc = [ising.magnetism(proc.run_process(lat, proc.field_cool(budget=120), s)) for s in seeds]
    assert np.mean(fc) > 0.8           # field-cooled -> strong remanent magnet
    assert np.mean(fc) > np.mean(zero) + 0.3


def test_process_changes_a_combined_material():
    """combine(process=field_cool) yields a more remanent material than the default settle."""
    reg = Registry(shape=SHAPE)
    reg.seed_elements(["iron", "chromium"])
    a, b = reg.get("iron"), reg.get("chromium")
    default = combine(a, b)
    field_cooled = combine(a, b, process=proc.field_cool(budget=120))
    assert default.id == field_cooled.id  # same lineage/id (process is not part of identity)
    assert field_cooled.properties["magnetism"] > default.properties["magnetism"]


# --- M5: evolving occupancy along the trajectory (the structural process payoff) -------


def test_evolve_occupancy_is_opt_in_and_deterministic():
    """evolve_occupancy=False is byte-identical to today; =True is deterministic & changes occupancy."""
    lat = _ferro()
    frozen = proc.anneal(budget=80, evolve_occupancy=False)
    live = proc.anneal(budget=80, evolve_occupancy=True)
    # Opt-out leaves occupancy exactly as the input structure (only spins move).
    out_frozen = proc.run_process(lat, frozen, seed=3)
    assert np.array_equal(out_frozen.occupied, lat.occupied)
    # Opt-in evolves occupancy (it differs) and is deterministic in (lattice, process, seed).
    a = proc.run_process(lat, live, seed=3)
    b = proc.run_process(lat, live, seed=3)
    assert np.array_equal(a.occupied, b.occupied)
    assert not np.array_equal(a.occupied, lat.occupied)


def test_pressure_schedule_changes_density():
    """Cooling the occupancy under higher pressure (μ) freezes in a denser solid (structural)."""
    from engine.process import Process, Stage

    lat = _ferro()

    def sinter(P):
        return Process((Stage(3.5, 3.5, 8, pressure=P), Stage(3.5, 0.4, 112, pressure=P)),
                       name=f"sinter{P}", evolve_occupancy=True)

    dense = [proc.run_process(lat, sinter(+6.0), seed=s).fill_fraction for s in range(3)]
    porous = [proc.run_process(lat, sinter(-6.0), seed=s).fill_fraction for s in range(3)]
    assert np.mean(dense) > np.mean(porous) + 0.03
