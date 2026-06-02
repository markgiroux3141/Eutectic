"""Deterministic, headless materials engine.

This package must never import game/UI code (spec §2). Everything in here is a pure,
reproducible function of its inputs plus the global ``UNIVERSE_SEED``.
"""
