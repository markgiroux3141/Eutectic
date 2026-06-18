"""The machine / object layer (spec §8) — assemblies that consume ``Material.properties``.

Objects are *assemblies*: each declares named **roles**, each role reads a few measured
properties of the material assigned to it, and **performance** is computed by real-ish
equations from those properties. The first worked example is the electric motor
(:mod:`machines.motor`).

Two hard rules for this layer:

* **It consumes ``Material.properties`` only** — never the lattice, the kernels, or any
  engine internals. The performance equations see a plain dict of measured numbers.
* **The engine never imports it** (spec §2, §8). The dependency arrow points one way:
  ``machines -> engine.properties (data)`` at runtime via the caller, never the reverse.
"""

from .roles import Blueprint, Requirement, Role
from .motor import MOTOR, MotorPerformance, OperatingPoint, build_motor

__all__ = [
    "Blueprint",
    "Requirement",
    "Role",
    "MOTOR",
    "MotorPerformance",
    "OperatingPoint",
    "build_motor",
]
