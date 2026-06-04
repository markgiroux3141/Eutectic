"""Unit tests for the Conditions value object (docs §2-§3).

Pins the dial-space type: standard point, immutability, the positive-temperature guard, and
the quantized seed key that makes ``measure(structure, conditions)`` deterministic (spec §6).
"""

import dataclasses

import pytest

from engine.conditions import STANDARD, STANDARD_TEMPERATURE, Conditions


def test_standard_is_the_reference_point():
    assert STANDARD.temperature == STANDARD_TEMPERATURE
    assert STANDARD.pressure == 0.0
    assert STANDARD.field == 0.0


def test_defaults_match_standard():
    assert Conditions() == STANDARD


def test_is_frozen():
    c = Conditions(temperature=2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.temperature = 3.0  # type: ignore[misc]


def test_nonpositive_temperature_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            Conditions(temperature=bad)


def test_with_temperature_keeps_other_dials():
    c = Conditions(temperature=1.0, pressure=2.0, field=0.5)
    d = c.with_temperature(3.0)
    assert d.temperature == 3.0
    assert d.pressure == 2.0 and d.field == 0.5


def test_seed_key_is_quantized_and_stable():
    # Two conditions that agree to the quantization precision seed identically...
    a = Conditions(temperature=2.0)
    b = Conditions(temperature=2.0 + 1e-9)
    assert a.seed_key() == b.seed_key()
    # ...and meaningfully different ones do not.
    assert a.seed_key() != Conditions(temperature=2.01).seed_key()
    assert a.seed_key() != Conditions(temperature=2.0, field=0.5).seed_key()
