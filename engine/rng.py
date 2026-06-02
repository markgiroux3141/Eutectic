"""Deterministic PRNG + hashing/mixing (spec §4.1, §6).

The determinism contract (spec §6) lives or dies here. Rules enforced by *only* using
what this module exposes:

* One seeded PRNG we control (SplitMix64). Never Python's global ``random``, ``time``,
  ``os.urandom``, or the salted built-in ``hash()`` of strings.
* All mixing is integer math on 64-bit lanes, fully reproducible across runs/platforms.

Usage::

    rng = SplitMix64(seed)
    x = rng.next_u64()          # raw 64-bit
    f = rng.next_float()        # uniform [0, 1)
    k = rng.randint(0, 10)      # uniform int in [0, 10)

For seeding from semantic data, use :func:`mix` (combine integer seeds) and
:func:`hash_str` / :func:`hash_bytes` (stable, NOT the built-in ``hash()``).
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

# 64-bit mask; all arithmetic below is performed mod 2**64.
_U64 = 0xFFFFFFFFFFFFFFFF

# A single global seed mixed into every combination (spec §6.6). Default 0 / fixed.
# Override per-save to make each playthrough's material space different while staying
# fully deterministic within a save (the Noita trick).
UNIVERSE_SEED: int = 0


def _splitmix64_step(state: int) -> tuple[int, int]:
    """One SplitMix64 step. Returns (new_state, output_u64).

    Reference algorithm by Sebastiano Vigna (public domain). Kept explicit so the bit
    pattern is auditable and portable to a TS/Rust port later (spec §2, §9.5).
    """
    state = (state + 0x9E3779B97F4A7C15) & _U64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    z = z ^ (z >> 31)
    return state, z


class SplitMix64:
    """Small, fast, fully deterministic 64-bit PRNG.

    Identical seed -> identical stream, forever, across runs and platforms. This is the
    *only* source of randomness the engine is permitted to use.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _U64

    def next_u64(self) -> int:
        """Next raw 64-bit unsigned integer."""
        self._state, z = _splitmix64_step(self._state)
        return z

    def next_float(self) -> float:
        """Uniform float in [0, 1) with 53 bits of mantissa precision."""
        # Top 53 bits of a u64 -> [0, 1), the standard construction.
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in [low, high). Rejection-sampled to avoid modulo bias."""
        if high <= low:
            raise ValueError(f"empty range [{low}, {high})")
        n = high - low
        # Largest multiple of n that fits in 64 bits; reject above it for uniformity.
        limit = (_U64 + 1) - ((_U64 + 1) % n)
        while True:
            r = self.next_u64()
            if r < limit:
                return low + (r % n)

    def spawn(self, salt: int) -> "SplitMix64":
        """Derive an independent child stream deterministically.

        Use to give a sub-stage (e.g. spin relaxation vs. cell merge) its own
        non-overlapping sequence without threading the parent's state through it.
        """
        return SplitMix64(mix(self._state, salt))

    def numpy_generator(self) -> np.random.Generator:
        """A numpy Generator seeded from this stream, for vectorized draws.

        Consumes one u64 to seed a PCG64 bit generator. Deterministic given the stream
        position. Prefer this over numpy's default_rng() with no seed (which is
        nondeterministic) anywhere the engine needs array-shaped randomness.
        """
        return np.random.Generator(np.random.PCG64(self.next_u64()))


def mix(*values: int) -> int:
    """Mix integer seeds into a single 64-bit seed (spec §4.1 ``mix(...)``).

    Order-sensitive and deterministic. Each value is folded through a SplitMix64 step so
    that nearby inputs diverge. Use to combine parent signatures with ``UNIVERSE_SEED``.
    """
    state = 0x243F6A8885A308D3  # arbitrary nonzero constant (digits of pi)
    for v in values:
        state = (state ^ (v & _U64)) & _U64
        state, _ = _splitmix64_step(state)
    _, out = _splitmix64_step(state)
    return out


def hash_bytes(data: bytes) -> int:
    """Stable 64-bit hash of bytes (NOT the salted built-in ``hash()``).

    Backed by BLAKE2b so it is reproducible across processes and platforms.
    """
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def hash_str(s: str) -> int:
    """Stable 64-bit hash of a string (UTF-8). See :func:`hash_bytes`."""
    return hash_bytes(s.encode("utf-8"))


def hash_ints(values: Iterable[int]) -> int:
    """Stable 64-bit hash of a sequence of integers, order-sensitive."""
    buf = b"".join((v & _U64).to_bytes(8, "little") for v in values)
    return hash_bytes(buf)


def hash_array(arr: np.ndarray) -> int:
    """Stable 64-bit structural hash of an ndarray's contents (spec §4.1).

    Used to derive a parent lattice's ``structural_signature``. Hashes the raw bytes in
    a canonical C-contiguous layout so equal arrays always hash equally.
    """
    a = np.ascontiguousarray(arr)
    # Include shape + dtype so different-shaped arrays with the same bytes differ.
    header = f"{a.dtype.str}|{a.shape}".encode("utf-8")
    return hash_bytes(header + a.tobytes())
