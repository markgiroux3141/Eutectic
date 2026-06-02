"""The M1 gate: same parents + same UNIVERSE_SEED -> byte-identical materials (spec §6).

This must pass from M1 onward and run in CI. It is the single most important test in the
suite: if combinations are not perfectly reproducible, the whole "discoverable, shareable
tech tree" premise collapses.
"""

import numpy as np

from engine import material as material_mod
from engine.lattice import merge, relax
from engine.material import combine, from_element
from engine.registry import Registry
from engine import elements


SHAPE = (48, 48)


def _assert_lattices_identical(x, y):
    assert np.array_equal(x.occupied, y.occupied)
    assert np.array_equal(x.atom_type, y.atom_type)
    assert np.array_equal(x.spin, y.spin)
    assert x.structural_signature() == y.structural_signature()


def _assert_materials_identical(x, y):
    assert x.id == y.id
    assert x.lineage == y.lineage
    assert x.properties == y.properties
    _assert_lattices_identical(x.lattice, y.lattice)


# --- stage-level determinism ----------------------------------------------------------


def test_merge_is_deterministic():
    iron = from_element(elements.get("iron"), shape=SHAPE)
    copper = from_element(elements.get("copper"), shape=SHAPE)
    a = merge(iron.lattice, copper.lattice, seed=42)
    b = merge(iron.lattice, copper.lattice, seed=42)
    _assert_lattices_identical(a, b)


def test_relax_is_deterministic():
    iron = from_element(elements.get("iron"), shape=SHAPE)
    a = relax(iron.lattice, seed=7, steps=8)
    b = relax(iron.lattice, seed=7, steps=8)
    _assert_lattices_identical(a, b)


def test_relax_actually_changes_spin():
    """Relaxation must do something (orders spins) — guards against a no-op pipeline."""
    iron = from_element(elements.get("iron"), shape=SHAPE)
    settled = relax(iron.lattice, seed=7, steps=8)
    # occupied/atom_type are preserved; spin should differ after settling.
    assert np.array_equal(settled.occupied, iron.lattice.occupied)
    assert not np.array_equal(settled.spin, iron.lattice.spin)


# --- combine() determinism (the gate) -------------------------------------------------


def test_combine_twice_byte_identical_fresh_registries():
    """Two independent registries must produce byte-identical children (spec §6)."""
    r1 = Registry(shape=SHAPE)
    r2 = Registry(shape=SHAPE)
    r1.seed_elements(["iron", "copper"])
    r2.seed_elements(["iron", "copper"])
    c1 = r1.combine("iron", "copper")
    c2 = r2.combine("iron", "copper")
    _assert_materials_identical(c1, c2)


def test_combine_is_commutative():
    """combine(A, B) == combine(B, A) (spec §4.1, §9.1)."""
    iron = from_element(elements.get("iron"), shape=SHAPE)
    copper = from_element(elements.get("copper"), shape=SHAPE)
    ab = combine(iron, copper)
    ba = combine(copper, iron)
    _assert_materials_identical(ab, ba)


def test_registry_cache_returns_same_object():
    reg = Registry(shape=SHAPE)
    reg.seed_elements(["iron", "copper"])
    first = reg.combine("iron", "copper")
    second = reg.combine("copper", "iron")  # reversed order hits same cache key
    assert first is second
    assert len(reg._combo_cache) == 1


def test_multi_step_chain_is_deterministic():
    """A 3-deep combination chain reproduces byte-identically across registries."""

    def build(reg: Registry):
        reg.seed_elements(["iron", "copper", "silver", "carbon"])
        ic = reg.combine("iron", "copper")
        sc = reg.combine("silver", "carbon")
        return reg.combine(ic.id, sc.id)

    top1 = build(Registry(shape=SHAPE))
    top2 = build(Registry(shape=SHAPE))
    _assert_materials_identical(top1, top2)
    # The top material is genuinely a grandchild (lineage of two combination ids).
    assert all(pid.startswith("m_") for pid in top1.lineage)


def test_universe_seed_changes_outcome():
    """Different UNIVERSE_SEED -> different material space, still deterministic (spec §6.6)."""
    r0 = Registry(shape=SHAPE, universe_seed=0)
    r1 = Registry(shape=SHAPE, universe_seed=1)
    r0.seed_elements(["iron", "copper"])
    r1.seed_elements(["iron", "copper"])
    c0 = r0.combine("iron", "copper")
    c1 = r1.combine("iron", "copper")
    assert c0.id != c1.id
    assert c0.lattice.structural_signature() != c1.lattice.structural_signature()


def test_properties_are_quantized():
    reg = Registry(shape=SHAPE)
    reg.seed_elements(["iron", "copper"])
    child = reg.combine("iron", "copper")
    for value in child.properties.values():
        assert value == round(value, material_mod.QUANT_DECIMALS)


def test_mismatched_shapes_rejected():
    iron_small = from_element(elements.get("iron"), shape=(16, 16))
    copper_big = from_element(elements.get("copper"), shape=(32, 32))
    try:
        combine(iron_small, copper_big)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched shapes")
