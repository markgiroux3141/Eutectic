"""Determinism + correctness of the PRNG/hashing layer (spec §6)."""

import numpy as np

from engine.rng import (
    SplitMix64,
    hash_array,
    hash_ints,
    hash_str,
    mix,
)


def test_splitmix64_same_seed_same_stream():
    a = SplitMix64(12345)
    b = SplitMix64(12345)
    assert [a.next_u64() for _ in range(100)] == [b.next_u64() for _ in range(100)]


def test_splitmix64_different_seed_diverges():
    a = [SplitMix64(1).next_u64() for _ in range(20)]
    b = [SplitMix64(2).next_u64() for _ in range(20)]
    assert a != b


def test_known_splitmix64_vector():
    # Reference SplitMix64 output for seed 0: first three outputs are well-known.
    rng = SplitMix64(0)
    assert rng.next_u64() == 0xE220A8397B1DCDAF
    assert rng.next_u64() == 0x6E789E6AA1B965F4
    assert rng.next_u64() == 0x06C45D188009454F


def test_next_float_in_range():
    rng = SplitMix64(99)
    vals = [rng.next_float() for _ in range(1000)]
    assert all(0.0 <= v < 1.0 for v in vals)


def test_randint_in_range_and_covers():
    rng = SplitMix64(7)
    seen = set()
    for _ in range(2000):
        v = rng.randint(0, 5)
        assert 0 <= v < 5
        seen.add(v)
    assert seen == {0, 1, 2, 3, 4}


def test_randint_empty_range_raises():
    rng = SplitMix64(7)
    try:
        rng.randint(3, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty range")


def test_spawn_is_deterministic_and_independent():
    parent_a = SplitMix64(42)
    parent_b = SplitMix64(42)
    ca = parent_a.spawn(1)
    cb = parent_b.spawn(1)
    assert [ca.next_u64() for _ in range(10)] == [cb.next_u64() for _ in range(10)]
    # Different salt -> different stream.
    other = SplitMix64(42).spawn(2)
    assert other.next_u64() != SplitMix64(42).spawn(1).next_u64()


def test_numpy_generator_is_deterministic():
    g1 = SplitMix64(5).numpy_generator()
    g2 = SplitMix64(5).numpy_generator()
    assert np.array_equal(g1.random(50), g2.random(50))


def test_mix_is_order_sensitive_and_stable():
    assert mix(1, 2, 3) == mix(1, 2, 3)
    assert mix(1, 2, 3) != mix(3, 2, 1)


def test_hash_str_stable_and_distinct():
    assert hash_str("iron") == hash_str("iron")
    assert hash_str("iron") != hash_str("gold")


def test_hash_ints_order_sensitive():
    assert hash_ints([1, 2, 3]) == hash_ints([1, 2, 3])
    assert hash_ints([1, 2, 3]) != hash_ints([3, 2, 1])


def test_hash_array_reflects_contents_and_shape():
    a = np.arange(12).reshape(3, 4)
    assert hash_array(a) == hash_array(a.copy())
    b = a.copy()
    b[0, 0] += 1
    assert hash_array(a) != hash_array(b)
    # Same bytes, different shape -> different hash.
    assert hash_array(a) != hash_array(a.reshape(4, 3))
