"""Synthesis as a **trajectory through conditions-space** (the process layer).

Real materials are process-sensitive: the *same* ingredients, annealed vs. quenched vs.
field-cooled, settle into measurably different structures (de-risked — slower cooling
reaches lower-energy, larger-domain configurations; cooling under a field builds remanence).
That sensitivity is not new machinery — it is the existing relaxation dynamics
(:func:`engine.lattice.metropolis_sweep`) carried along a *schedule* of conditions instead
of run once at a fixed temperature.

A :class:`Process` is an ordered list of :class:`Stage`s; :func:`run_process` threads the
live spin state through them. This generalises :func:`engine.lattice.relax`: a single
constant-temperature stage *is* a relax (``STANDARD_PROCESS`` reproduces it byte-for-byte —
see ``tests/test_process.py``), and everything else (anneal, quench, field-cool, hold-time)
is the same kernel on a richer schedule.

The One Principle is preserved (spec §1): the process shapes the *structure*; properties are
still **measured** from the result, never assigned. "Wrong process = bad material" means the
trajectory landed the lattice in a basin whose *measured* microstructure is poor — a legible
consequence, not a hidden gate.

Determinism (spec §6): a process runs from one :class:`~engine.rng.SplitMix64` stream, fixed
stage/sweep order, and the shared checkerboard kernel. ``Process.signature`` is a stable hash
(quantized dials) so a process can key a cache and seed a trajectory reproducibly.

Pressure ``P`` is carried on each stage but **inert until M5** (``occupied`` is not yet a
thermal degree of freedom — docs §3/§5); it is stored, not faked into having an effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import (
    COHESION_J0,
    EXCHANGE_J0,
    Lattice,
    checkerboard_colors,
    metropolis_sweep,
    occupancy_sweep,
)
from .rng import SplitMix64, hash_ints

# Salt so the occupancy stream never collides with the spin stream inside a process.
_OCC_STREAM_SALT: int = 0x4F  # 'O'

# Quantization for the process signature (matches the conditions seed precision).
_SIG_SCALE: int = 10_000


def _q(x: float) -> int:
    return int(round(float(x) * _SIG_SCALE))


@dataclass(frozen=True)
class Stage:
    """One leg of a synthesis schedule: hold/ramp temperature over ``sweeps`` sweeps.

    Temperature ramps **linearly** from ``t_start`` to ``t_end`` across the stage (a *hold*
    is ``t_start == t_end``; a *cooling ramp* is ``t_start > t_end``). ``field`` is the
    magnetic dial ``H`` applied throughout the stage; ``pressure`` is carried but inert
    until M5.
    """

    t_start: float
    t_end: float
    sweeps: int
    field: float = 0.0
    pressure: float = 0.0

    def __post_init__(self) -> None:
        if self.sweeps < 0:
            raise ValueError(f"stage sweeps must be >= 0, got {self.sweeps}")
        if self.t_start <= 0.0 or self.t_end <= 0.0:
            raise ValueError("stage temperatures must be > 0 (Boltzmann weights)")

    def temperature_at(self, i: int) -> float:
        """Temperature on sweep ``i`` of this stage (linear interpolation)."""
        if self.sweeps <= 1 or self.t_start == self.t_end:
            return self.t_start
        return self.t_start + (self.t_end - self.t_start) * (i / (self.sweeps - 1))


@dataclass(frozen=True)
class Process:
    """An ordered synthesis schedule (a trajectory through conditions-space).

    ``evolve_occupancy`` (M5) opts the trajectory into evolving the **occupancy** field too
    (the lattice-gas / crystallisation dynamics), not just spins — so a slow anneal coarsens
    atoms into positional (sublattice) order while a quench freezes positional disorder, a
    *structural* (non-magnetic) path-dependence. Default ``False`` keeps occupancy frozen, so
    ``STANDARD_PROCESS`` still reproduces :func:`engine.lattice.relax` byte-for-byte.
    """

    stages: tuple[Stage, ...]
    name: str = "custom"
    evolve_occupancy: bool = False

    @property
    def total_sweeps(self) -> int:
        return sum(s.sweeps for s in self.stages)

    def signature(self) -> int:
        """Stable 64-bit hash of the schedule (quantized dials; ``name`` excluded).

        Two processes with the same stages (and same dynamics flags) produce the same
        trajectory, so the signature depends only on the dynamics, not the label.
        """
        ints: list[int] = [int(self.evolve_occupancy)]
        for s in self.stages:
            ints += [_q(s.t_start), _q(s.t_end), int(s.sweeps), _q(s.field), _q(s.pressure)]
        return hash_ints(ints)


def run_process(
    lattice: Lattice,
    process: Process,
    seed: int,
    *,
    coupling: float = EXCHANGE_J0,
) -> Lattice:
    """Evolve ``lattice``'s spin state along ``process`` and return the settled lattice.

    Starts from the lattice's *current* spins (so a single constant stage reproduces
    :func:`engine.lattice.relax`); a process that wants to start from a melt simply opens
    with a hot hold (see :func:`anneal`/:func:`quench`/:func:`field_cool`). By default only
    ``spin`` is evolved — geometry/couplings are the permanent structure (docs §2) — and empty
    cells are pinned to +1 at the end, as in ``relax``.

    With ``process.evolve_occupancy`` (M5) the **occupancy** field is evolved too, along the
    same schedule (the lattice-gas / crystallisation kernel at each stage's ``T`` and a
    pressure-set chemical potential μ): a slow anneal coarsens atoms into sublattice order, a
    quench freezes positional disorder. Its acceptance stream is *spawned separately* so the
    spin trajectory is unchanged; with the default ``evolve_occupancy=False`` occupancy is
    frozen and the spin path is byte-for-byte today's ``relax`` (the equivalence guard).

    Deterministic in ``(lattice, process, seed)``: seeded streams, fixed stage/sweep order
    (spec §6).
    """
    occ_mask = lattice.occupied.astype(bool)
    spin = lattice.spin.astype(np.int8).copy()
    m = np.asarray(lattice.moment, dtype=np.float64) * occ_mask
    gen = SplitMix64(seed).numpy_generator()
    colors = checkerboard_colors(lattice.shape)

    evolve_occ = process.evolve_occupancy
    if evolve_occ:
        # Imported lazily so the lattice-level executor needs the higher-level thermal module
        # only when occupancy evolution is actually requested.
        from .thermal import PRESSURE_TO_MU, symmetric_mu

        occ = lattice.occupied.astype(np.uint8).copy()
        cohesion = lattice.cohesion
        occ_gen = SplitMix64(seed).spawn(_OCC_STREAM_SALT).numpy_generator()
        mu_base = symmetric_mu(lattice, COHESION_J0)  # half-filling μ; pressure shifts it

    for stage in process.stages:
        mu = (mu_base + PRESSURE_TO_MU * stage.pressure) if evolve_occ else 0.0
        for i in range(stage.sweeps):
            T = stage.temperature_at(i)
            spin = metropolis_sweep(
                spin, m, occ_mask, colors,
                coupling=coupling, temperature=T, field=stage.field, gen=gen,
            )
            if evolve_occ:
                occ = occupancy_sweep(
                    occ, cohesion, colors,
                    coupling=COHESION_J0, temperature=T, mu=mu, gen=occ_gen,
                )

    if evolve_occ:
        occ_mask = occ.astype(bool)
        spin = np.where(occ_mask, spin, 1).astype(np.int8)
        return Lattice(
            occupied=occ, atom_type=lattice.atom_type, spin=spin,
            mass=lattice.mass, moment=lattice.moment, cohesion=lattice.cohesion,
        )

    spin = np.where(occ_mask, spin, 1).astype(np.int8)
    return Lattice(
        occupied=lattice.occupied,
        atom_type=lattice.atom_type,
        spin=spin,
        mass=lattice.mass,
        moment=lattice.moment,
        cohesion=lattice.cohesion,
    )


# --- the standard process (today's relax, as a schedule) ------------------------------
# A single 12-sweep hold at the standard temperature reproduces engine.lattice.relax
# byte-for-byte (the equivalence is pinned in tests/test_process.py). T0 = RELAX_TEMPERATURE.
STANDARD_STEPS: int = 12
STANDARD_TEMPERATURE: float = 1.0
STANDARD_PROCESS = Process(
    (Stage(STANDARD_TEMPERATURE, STANDARD_TEMPERATURE, STANDARD_STEPS),), name="standard"
)


# --- process presets (self-contained: each opens by melting, so they are start-agnostic) ---
# Defaults sized to the Step-0 de-risk: ~120 total sweeps, hot melt -> cold, t_hot above the
# ferromagnets' Tc (~2.0) so the melt is genuinely disordered.
_T_HOT: float = 4.0
_T_COLD: float = 0.3
_BUDGET: int = 120
_MELT: int = 8


def anneal(
    *, t_hot: float = _T_HOT, t_cold: float = _T_COLD, budget: int = _BUDGET,
    melt: int = _MELT, ramp_sweeps: int | None = None, name: str = "anneal",
    evolve_occupancy: bool = False,
) -> Process:
    """Melt, then cool slowly to ``t_cold`` over the budget (large domains, low energy).

    ``ramp_sweeps`` shorter than the budget gives a *faster* cool followed by a cold hold —
    use it to build an "anneal-fast" for comparison. With ``evolve_occupancy=True`` (M5) the
    slow cool also crystallises the *occupancy* into sublattice order (high positional order).
    """
    ramp = (budget - melt) if ramp_sweeps is None else ramp_sweeps
    cold = budget - melt - ramp
    stages = [Stage(t_hot, t_hot, melt), Stage(t_hot, t_cold, ramp)]
    if cold > 0:
        stages.append(Stage(t_cold, t_cold, cold))
    return Process(tuple(stages), name=name, evolve_occupancy=evolve_occupancy)


def quench(
    *, t_hot: float = _T_HOT, t_cold: float = _T_COLD, budget: int = _BUDGET,
    melt: int = _MELT, name: str = "quench", evolve_occupancy: bool = False,
) -> Process:
    """Melt, then slam cold and hold (frozen-in disorder: small domains, high energy).

    With ``evolve_occupancy=True`` (M5) the fast cool freezes *positional* disorder too — the
    occupancy never reaches sublattice order (low positional order), the structural quench.
    """
    return Process(
        (Stage(t_hot, t_hot, melt), Stage(t_cold, t_cold, budget - melt)),
        name=name, evolve_occupancy=evolve_occupancy,
    )


def field_cool(
    *, t_hot: float = _T_HOT, t_cold: float = _T_COLD, budget: int = _BUDGET,
    melt: int = _MELT, field: float = 0.5, relax_off: int = 10, name: str = "field-cool",
) -> Process:
    """Melt, cool *under a field* (texture/alignment), then remove the field and hold cold.

    The closing zero-field hold makes the measured magnetization a true **remanence** (what
    the material retains after the field is turned off). NB: with isotropic Ising this is
    *kinetic* remanence (low-T freezing), not coercive remanence — that needs the anisotropy
    block (a later, separately de-risked milestone).
    """
    cool = budget - melt - relax_off
    return Process(
        (
            Stage(t_hot, t_hot, melt),
            Stage(t_hot, t_cold, cool, field=field),
            Stage(t_cold, t_cold, relax_off, field=0.0),
        ),
        name=name,
    )
