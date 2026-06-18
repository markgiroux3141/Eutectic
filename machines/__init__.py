"""The machine / object layer (spec §8) — assemblies that consume ``Material.properties``.

Objects are *assemblies*: each declares named **roles**, each role reads a few measured
properties of the material assigned to it, and **performance** is computed by real-ish
equations from those properties. The worked examples span the four property families so each
rewards a different rare material:

* :mod:`machines.motor`         — magnetic + electrical + mechanical (torque)
* :mod:`machines.electromagnet` — magnetic + electrical (lift force ∝ I²)
* :mod:`machines.cable`         — electrical transmission (loss + ampacity + sag over distance)
* :mod:`machines.heatsink`      — thermal (the diamond divergence pays off: carbon wins)
* :mod:`machines.armor`         — mechanical composite (solves the M8 strength↔ductility dilemma)

Two hard rules for this layer:

* **It consumes ``Material.properties`` only** — never the lattice, the kernels, or any
  engine internals. The performance equations see a plain dict of measured numbers.
* **The engine never imports it** (spec §2, §8). The dependency arrow points one way.
"""

from .roles import Blueprint, Requirement, Role
from .motor import MOTOR, MotorPerformance, OperatingPoint, build_motor
from .electromagnet import ELECTROMAGNET, ElectromagnetPerformance, build_electromagnet
from .cable import CABLE, CablePerformance, TransmissionLoad, build_cable
from .heatsink import HEAT_SINK, HeatSinkPerformance, ThermalLoad, build_heat_sink
from .armor import ARMOR, ArmorPerformance, build_armor

__all__ = [
    # framework
    "Blueprint",
    "Requirement",
    "Role",
    # motor
    "MOTOR",
    "MotorPerformance",
    "OperatingPoint",
    "build_motor",
    # electromagnet
    "ELECTROMAGNET",
    "ElectromagnetPerformance",
    "build_electromagnet",
    # cable
    "CABLE",
    "CablePerformance",
    "TransmissionLoad",
    "build_cable",
    # heat sink
    "HEAT_SINK",
    "HeatSinkPerformance",
    "ThermalLoad",
    "build_heat_sink",
    # armor
    "ARMOR",
    "ArmorPerformance",
    "build_armor",
]
