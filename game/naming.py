"""Deterministic coined names for discovered materials (reproducible alchemy).

A combination's engine id is a hash (``m_9f3a...``) — fine for the registry, awful to type or
remember. The game gives each *discovered* material a short, pronounceable handle coined
**deterministically from that id**, so the same combination always earns the same name across
sessions and saves (the determinism contract extends to the UX). Root elements keep their real
name; only combinations are coined.
"""

from __future__ import annotations

# Alternating consonant/vowel tables -> pronounceable CVCVC tokens. Sizes are coprime-ish with
# 16 so successive hex nibbles spread across them.
_CONS = "bdfgklmnprstvz"
_VOWELS = "aeiou"


def coin_name(material_id: str) -> str:
    """A short pronounceable name derived deterministically from a material id.

    Folds the id's hex digits into consonant/vowel slots to make a CVCVC token (e.g. ``tovak``).
    Pure function of the id, so it is stable across sessions — two players who discover the same
    alloy in the same universe see the same name.
    """
    # Pull the hex tail of the id (after the "m_" prefix); fall back to the raw string's chars.
    tail = material_id.split("_", 1)[-1]
    digits = [int(c, 16) for c in tail if c in "0123456789abcdef"]
    if not digits:
        digits = [ord(c) % 16 for c in material_id] or [0]
    slots = [_CONS, _VOWELS, _CONS, _VOWELS, _CONS]
    name = []
    for k, table in enumerate(slots):
        d = digits[k % len(digits)]
        name.append(table[d % len(table)])
    return "".join(name)
