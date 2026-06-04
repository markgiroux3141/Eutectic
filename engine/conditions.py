"""Thermodynamic conditions: the layer that separates **structure** from **state**.

See ``docs/conditions-and-properties.md`` (§2-§3). A material *is* its permanent
structure (geometry + couplings on the lattice). A property is an **ensemble observable of
that structure at given conditions** — not a frozen number. :class:`Conditions` is the
"dial" half of ``measure(structure, conditions) -> value``.

The core dial set is ``(T, P, H)`` — each a thermodynamic conjugate to a response the
lattice already has (docs §3):

* **temperature ``T``** couples to entropy/energy — *the* thermal dial, already a parameter
  of :func:`engine.lattice.relax`. Drives the spin ensemble (M4) and, later, the occupancy
  ensemble (M5).
* **magnetic field ``H``** couples to magnetization via a ``-H·Σ(moment·spin)`` term in the
  Hamiltonian. Wired into the Ising engine in M4 (see :mod:`engine.thermal`).
* **pressure ``P``** couples to the amount of matter (``occupied``). **ACTIVE as of M5** —
  ``occupied`` is now a thermal degree of freedom (the lattice-gas / melting ensemble), and
  ``P`` enters as a chemical-potential offset ``μ = μ_sym + PRESSURE_TO_MU·P`` (see
  :func:`engine.thermal.chemical_potential`): P>0 raises μ → favours occupancy → denser; the
  standard P=0 sits at the particle-hole-symmetric μ (half-filling), where melting is clean.
  Until M5 it was carried-but-inert (flagged honestly); it now changes measured density and
  shifts the melting/percolation behaviour. (It remains inert for the *spin* ensemble — the
  Curie measurement reads only ``T`` and ``H``.)

Conditions are quantized when used to seed a measurement (:meth:`Conditions.seed_key`) so
that ``measure(structure, conditions)`` is a *pure deterministic function* — measuring the
same structure at the same conditions twice yields byte-identical results (spec §6).
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard conditions ``(T0, P0, H0)``. T0 matches :data:`engine.lattice.RELAX_TEMPERATURE`
# (the temperature the reference snapshot is settled at), kept as a literal here to avoid a
# module import cycle; the two are intentionally the same value.
STANDARD_TEMPERATURE: float = 1.0
STANDARD_PRESSURE: float = 0.0
STANDARD_FIELD: float = 0.0

# Decimals the dials are quantized to before seeding a measurement (spec §6.4). Coarse
# enough that float dust can't perturb a seed, fine enough to resolve a temperature sweep.
_SEED_DECIMALS: int = 4
_SEED_SCALE: int = 10 ** _SEED_DECIMALS


@dataclass(frozen=True)
class Conditions:
    """A point in thermodynamic dial-space ``(temperature, pressure, field)`` (docs §3).

    Frozen value object. ``pressure`` is carried but inert until M5 (see module docstring).
    """

    temperature: float = STANDARD_TEMPERATURE
    pressure: float = STANDARD_PRESSURE
    field: float = STANDARD_FIELD

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            # Boltzmann weights need T > 0; T=0 is a separate (deterministic) limit we don't
            # model as an ensemble. Callers sweep T over a strictly positive range.
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    def seed_key(self) -> tuple[int, int, int]:
        """Quantized ``(T, P, H)`` as integers, for deterministic measurement seeding.

        Two ``Conditions`` that quantize equal seed identically, so a measurement is a pure
        function of ``(structure, quantized conditions)`` (spec §6).
        """
        return (
            int(round(self.temperature * _SEED_SCALE)),
            int(round(self.pressure * _SEED_SCALE)),
            int(round(self.field * _SEED_SCALE)),
        )

    def with_temperature(self, temperature: float) -> "Conditions":
        """A copy at a different temperature (the common move when sweeping T)."""
        return Conditions(temperature=temperature, pressure=self.pressure, field=self.field)


# The reference point every Material's stored snapshot is measured at (docs §2).
STANDARD = Conditions()
