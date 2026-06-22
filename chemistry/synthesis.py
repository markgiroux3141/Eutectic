"""Synthesis as a **trajectory through conditions-space** (spec §13 — the chemistry process layer).

C5 (:mod:`chemistry.network`) answers "what is reachable at *one* fixed condition?". But a real
synthesis is a *route*: heat to drive one step, quench to capture its product, add a catalyst,
pressurise. This module is the chemistry analog of :mod:`engine.process` — where that threads a
lattice's **spin** state through a schedule of metropolis sweeps, this threads the **species
inventory** through a schedule of :class:`~chemistry.conditions.ChemConditions`, growing it by the
C5 reachability closure at each stage.

A :class:`Route` is an ordered list of condition **set-points** (holds); :func:`synthesize` folds
:meth:`ReactionNetwork.reachable` across them, **accumulating** the inventory (a species attained at
any stage stays available — you isolated/captured it there). A single-stage route is exactly a C5
``reachable`` (``isothermal`` reproduces it — pinned by a test), so this is a strict generalisation.

Why a route beats every static condition (the keystone, de-risked)
------------------------------------------------------------------
Some targets are reachable at **no fixed temperature** yet fall out of a trajectory, because the
step that *makes* an intermediate and the step that *consumes* it want **different** temperatures —
and a single set-point can only be at one. The headline case is **NO**:

* atomic N exists only above its dissociation T*≈7.8 (a genuine ΔG sign-crossing, C3);
* but ``N+O→NO`` is exergonic only below T*≈5.4 (another genuine sign-crossing);
* those windows are **disjoint**, so NO is unreachable at every static T (C5 found exactly this).

A **heat-then-cool** route makes it: heat (T≈8) to dissociate N₂/O₂ into radicals, then **quench**
(T≈1) where ``N+O→NO`` is now exergonic, capturing the NO. This is real radical chemistry — NO forms
in high-T combustion/lightning and is **frozen in** by rapid cooling. The gates are genuine ΔG
sign-crossings, not dials, and **order is load-bearing**: cool-then-heat does *not* make NO (it ends
hot, where NO won't form). Path-dependence as a measured consequence, not an authored gate.

Honest scope (no-fudge norm, flagged not buried)
------------------------------------------------
* **Cumulative attainability, not concentrations.** ``synthesize`` tracks *what species the route
  can yield* (capture-at-the-favourable-stage), the same set semantics as C5 ``reachable`` extended
  in time. It does **not** model that, left at the cold stage, NO would slowly back-react if not
  isolated, nor partial yields/kinetics-limited conversion — that is a quantitative concentration
  model, deliberately out of scope (spec §20: qualitative correctness + emergence, not yields).
* **The gates inherit C5's honesty split.** Reachability per stage gates on ΔG (genuine threshold)
  AND a soft rate cutoff; the trajectory keystones above are anchored on the ΔG sign-crossings. The
  rate cutoff is the same flagged dial as in :mod:`chemistry.network`.
* **Discrete holds, not continuous ramps.** A stage is a set-point; a ramp is approximated by more
  stages. What matters thermodynamically is the *set of conditions visited, in order* — which holds
  capture exactly. (The :mod:`engine.process` analog ramps T linearly because its kernel integrates
  a continuous anneal; reachability is a fixed-point at a condition, so set-points are the right unit.)

Determinism (spec §14): reachability is deterministic; stage order is fixed; the species lookup is
keyed canonically. :meth:`Route.signature` folds the stages' quantized condition seed-keys, so a
route keys a cache reproducibly (mirroring :meth:`engine.process.Process.signature`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.rng import hash_ints

from .conditions import STANDARD_CHEM, ChemConditions
from .network import DEFAULT_RATE_CUTOFF, ReactionNetwork, SpeciesKey, species_key
from .reaction import Species


@dataclass(frozen=True)
class Route:
    """An ordered synthesis schedule: condition set-points threaded in sequence (spec §13).

    Each stage is a :class:`~chemistry.conditions.ChemConditions` hold; the inventory is grown by
    the C5 reachability closure at each, accumulating. Build directly or via :func:`isothermal` /
    :func:`heat_quench`.
    """

    stages: tuple[ChemConditions, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a route needs at least one stage")

    def signature(self) -> int:
        """Stable 64-bit hash of the schedule (quantized stage conditions; ``name`` excluded)."""
        ints: list[int] = []
        for c in self.stages:
            ints.extend(c.seed_key())
        return hash_ints(ints)


@dataclass(frozen=True)
class SynthesisResult:
    """The outcome of running a :class:`Route`: what was attainable, and where it first appeared.

    ``reachable`` is the cumulative set of species keys; ``first_stage`` maps each *synthesised*
    species (not in the starting inventory) to the index of the stage that first made it; ``species``
    maps every reachable key to its :class:`~chemistry.reaction.Species` object.
    """

    route: Route
    reachable: frozenset[SpeciesKey]
    first_stage: dict[SpeciesKey, int]
    species: dict[SpeciesKey, Species] = field(default_factory=dict)

    def made(self, target: Species) -> bool:
        """True iff ``target`` is attainable along this route."""
        return species_key(target) in self.reachable

    def stage_made(self, target: Species) -> int | None:
        """Index of the stage that first made ``target`` (``None`` if inventory or never made)."""
        return self.first_stage.get(species_key(target))

    def products(self) -> tuple[SpeciesKey, ...]:
        """The synthesised species (everything reachable that wasn't in the starting inventory),
        in canonical key order."""
        return tuple(sorted(self.first_stage))


def synthesize(network: ReactionNetwork, inventory, route: Route, *,
               rate_cutoff: float = DEFAULT_RATE_CUTOFF,
               require_rate: bool = True) -> SynthesisResult:
    """Run ``route`` over ``network`` from ``inventory``; return what the trajectory can yield.

    Folds :meth:`ReactionNetwork.reachable` across the route's stages, accumulating the inventory
    (cumulative attainability — see the module docstring). ``inventory`` is any iterable of
    :class:`~chemistry.reaction.Species`. Deterministic in ``(network, inventory, route, gates)``.
    """
    lookup: dict[SpeciesKey, Species] = {species_key(s): s for s in network.species()}
    pool: dict[SpeciesKey, Species] = {}
    first_stage: dict[SpeciesKey, int] = {}
    for sp in inventory:
        k = species_key(sp)
        pool.setdefault(k, sp)
        lookup.setdefault(k, sp)

    for idx, cond in enumerate(route.stages):
        reached = network.reachable(list(pool.values()), cond,
                                    rate_cutoff=rate_cutoff, require_rate=require_rate)
        for k in sorted(reached):  # canonical order → deterministic first_stage assignment
            if k not in pool:
                pool[k] = lookup.get(k, pool.get(k))
                first_stage[k] = idx

    return SynthesisResult(route=route, reachable=frozenset(pool),
                           first_stage=first_stage, species=dict(pool))


# --- route presets --------------------------------------------------------------------

def isothermal(temperature: float, *, base: ChemConditions = STANDARD_CHEM,
               name: str = "isothermal") -> Route:
    """A one-stage hold at ``temperature`` — i.e. a plain C5 ``reachable`` as a (trivial) route."""
    return Route((base.with_temperature(temperature),), name=name)


def heat_quench(t_hot: float, t_cold: float, *, base: ChemConditions = STANDARD_CHEM,
                name: str = "heat-quench") -> Route:
    """Heat to ``t_hot`` (drive/dissociate), then quench to ``t_cold`` (capture) — the route that
    reaches trajectory-only targets like NO. Catalysts/pressure on ``base`` apply to both stages."""
    return Route((base.with_temperature(t_hot), base.with_temperature(t_cold)), name=name)
