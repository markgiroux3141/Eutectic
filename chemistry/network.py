"""Reaction network / tech tree: ``reachable(inventory, conditions)`` (spec §12 — C5).

Species and reactions form a **directed graph**; what you can make from a starting inventory
*emerges* from it. A reaction is **live** at the given conditions when it is both

* **thermodynamically feasible** — ΔG < 0 (C3, :mod:`chemistry.reaction`), and
* **kinetically fast enough** — its Arrhenius rate clears a reference cutoff at the available T,
  given any catalysts present (C4, :mod:`chemistry.kinetics`).

:meth:`ReactionNetwork.reachable` is the **transitive closure**: start from the inventory, fire
every live reaction whose reactants are all available, add the products, and repeat to a fixed
point. Prerequisites ("you must make Y before X"), condition-gating ("X only forms above some T,
or only with a catalyst you must first obtain"), and rarity (exotic species behind narrow windows
and long chains) are **not authored** — they fall out of the graph + the per-reaction gates.

What the gate is anchored on (the no-fudge discipline, spec §15 / §20)
----------------------------------------------------------------------
There are **two** gates, and they are not equally trustworthy — we say which is which:

* **ΔG sign-crossing = a GENUINE threshold.** ΔG is strictly monotone in T, and feasibility flips
  *at* the crossover temperature ``T* = ΔH/ΔS`` (C3). The reachability claims we *stand on* are
  anchored here. The headline demo: free radicals (O, Cl, H, N) are unreachable cold and become
  reachable as T climbs through each diatomic's dissociation T*, **in emergent order of bond
  strength** (Cl 4.03 < O 4.62 < H 7.27 < N 7.79 — the very ordering C3 proved), with no tuned
  numbers anywhere. Raising T grows the reachable set across real thresholds.

* **The rate cutoff = a SOFT dial, and we flag it as one.** "Fast enough" compares a smooth
  Arrhenius rate to a reference :data:`DEFAULT_RATE_CUTOFF`. Nudging that cutoff slides the
  unlock temperature *continuously* — the tell-tale of a non-transition (C4 pinned exactly this).
  So we do **not** dress a ``rate > cutoff`` unlock as a sharp tech-tree gate; where a demo gates
  on rate (e.g. the catalysed Haber unlock), we report the disguised-dial slide alongside it
  (:meth:`unlock_temperature` + the test ``..._rate_gate_is_a_soft_dial``). ``require_rate=False``
  drops the kinetic gate entirely, leaving the pure-ΔG threshold to stand on its own.

Honest findings from the C5 de-risk (reported, not buried — spec §20)
---------------------------------------------------------------------
* **Rarity is emergent: some targets are locked out at every temperature.** ``NO`` is in the
  graph but unreachable: the direct route ``N₂+O₂→2NO`` is endothermic with Δn_gas=0 (ΔG>0 at all
  T), and the radical route ``N+O→NO`` is exergonic only below T*≈5.4, which is *disjoint* from
  when atomic N exists (above T*≈7.8). No path, at any T — and nothing authored that as "rare".
* **A unique radical-only synthesis to a stable compound generally does not exist in this
  substrate.** Radical recombination shrinks gas moles (Δn_gas<0), so it turns endergonic above
  ``E_formed/|ΔS|``; that ceiling sits *below* the dissociation T* unless the formed bond beats
  the diatomic it came from. So strong diatomics (N₂) can't be liberated and usefully reassembled
  by heat alone. A genuine multi-step prerequisite *does* survive where only the weaker partner
  needs liberating and the product bond is strong — ``O₂→2O`` then ``O+H₂→H₂O`` (overlap window
  T∈[4.62, 6.09]) — so atomic O is a real, required prerequisite for that route.
* **Cross-character magnitudes stay out of the feasibility comparisons.** Every reaction in the
  curated demo network is gas-phase covalent (diatomics, H₂O/NH₃/HCl/NO, free atoms); no live/dead
  decision hinges on the uncalibrated ionic/metallic energy scale (the C2/C1 caveat).

Determinism (spec §14)
----------------------
Species identity is ``(formula, phase)``; reactions are ordered by their canonical id; the
fixed-point loop iterates that fixed order and grows an insertion-ordered pool. ``reachable``
returns a ``frozenset`` (order-independent equality), so "reachable twice ⇒ identical set" holds
by construction — pinned as a test from the start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conditions import STANDARD_CHEM, ChemConditions
from .kinetics import Kinetics, kinetics
from .reaction import Reaction, Species

# The reference "fast enough" rate (C4 model units). This is a SOFT DIAL, not a threshold —
# see the module docstring. It is a low floor: it only screens out reactions that are
# astronomically slow at the available T (the level below which "feasible in principle" means
# nothing on any timescale). The genuine reachability gate is the ΔG sign; this value never
# moves a dissociation onset (those clear it by ~50 orders of magnitude at their T*), it only
# decides where the soft kinetic unlocks (HCl, catalysed Haber) land — and that landing slides
# continuously when you nudge this number, which is exactly why we call it a dial, not a gate.
DEFAULT_RATE_CUTOFF: float = 1.0e-9

SpeciesKey = tuple[str, str]


def species_key(sp: Species) -> SpeciesKey:
    """Canonical identity of a species: ``(formula, phase)`` (spec §14.5)."""
    return (sp.formula, sp.phase.value)


@dataclass(frozen=True)
class NetworkReaction:
    """A reaction as a graph edge, with the species that catalyse it (spec §4, §12).

    Wraps a C3 :class:`~chemistry.reaction.Reaction` (which owns ΔG) and carries the authored
    catalyst set for its C4 :class:`~chemistry.kinetics.Kinetics` (which owns the rate). The
    edge is **live** when ΔG<0 and — unless ``require_rate`` is off — its rate clears the cutoff.
    """

    reaction: Reaction
    catalysts: frozenset[str] = field(default_factory=frozenset)

    @property
    def kinetics(self) -> Kinetics:
        return kinetics(self.reaction, self.catalysts)

    def is_feasible(self, conditions: ChemConditions = STANDARD_CHEM) -> bool:
        """The genuine gate: ΔG < 0 (a real sign-crossing, not a dial)."""
        return self.reaction.is_spontaneous(conditions)

    def is_fast_enough(self, conditions: ChemConditions = STANDARD_CHEM,
                       rate_cutoff: float = DEFAULT_RATE_CUTOFF) -> bool:
        """The soft gate: Arrhenius rate ≥ ``rate_cutoff`` (a dial — flagged, see module doc)."""
        return self.kinetics.rate(conditions) >= rate_cutoff

    def is_live(self, conditions: ChemConditions = STANDARD_CHEM, *,
                rate_cutoff: float = DEFAULT_RATE_CUTOFF, require_rate: bool = True) -> bool:
        """Live = feasible (ΔG<0) AND (``require_rate`` off, or fast enough) at the conditions."""
        if not self.is_feasible(conditions):
            return False
        if require_rate and not self.is_fast_enough(conditions, rate_cutoff):
            return False
        return True

    def reactant_keys(self) -> tuple[SpeciesKey, ...]:
        return tuple(species_key(sp) for sp, _ in self.reaction.reactants)

    def product_keys(self) -> tuple[SpeciesKey, ...]:
        return tuple(species_key(sp) for sp, _ in self.reaction.products)


def edge(reaction: Reaction, catalysts: frozenset[str] | set[str] | None = None) -> NetworkReaction:
    """Build a :class:`NetworkReaction` (a graph edge) from a reaction + optional catalysts."""
    return NetworkReaction(reaction=reaction, catalysts=frozenset(catalysts or ()))


@dataclass(frozen=True)
class ReactionNetwork:
    """A directed graph of species + reactions; reachability is its transitive closure (spec §12).

    Construct from :class:`NetworkReaction` edges (or use :func:`reaction_network`, which accepts
    bare reactions too). The reactions are stored in a fixed canonical order so every traversal is
    deterministic regardless of construction order (spec §14).
    """

    reactions: tuple[NetworkReaction, ...]

    def __post_init__(self) -> None:
        # Canonical, construction-order-independent iteration order (spec §14.2/§14.6).
        ordered = tuple(sorted(self.reactions, key=lambda nr: nr.reaction.canonical_id()))
        object.__setattr__(self, "reactions", ordered)

    def species(self) -> tuple[Species, ...]:
        """Every distinct species appearing in any reaction, in canonical (key-sorted) order."""
        seen: dict[SpeciesKey, Species] = {}
        for nr in self.reactions:
            for sp, _ in (*nr.reaction.reactants, *nr.reaction.products):
                seen.setdefault(species_key(sp), sp)
        return tuple(seen[k] for k in sorted(seen))

    def live_reactions(self, conditions: ChemConditions = STANDARD_CHEM, *,
                       rate_cutoff: float = DEFAULT_RATE_CUTOFF,
                       require_rate: bool = True) -> tuple[NetworkReaction, ...]:
        """All edges live at the conditions (ΔG<0, and fast enough unless ``require_rate`` off)."""
        return tuple(nr for nr in self.reactions
                     if nr.is_live(conditions, rate_cutoff=rate_cutoff, require_rate=require_rate))

    # --- reachability (the tech tree) -------------------------------------------------
    def _close(self, inventory, conditions: ChemConditions,
               rate_cutoff: float, require_rate: bool) -> dict[SpeciesKey, NetworkReaction | None]:
        """Fixed-point closure; returns ``key -> first reaction that produced it`` (None=inventory).

        Insertion order is deterministic: inventory first (in given order), then products in the
        canonical reaction order, each the first time it appears. A reaction fires when every
        reactant key is already in the pool and the edge is live.
        """
        pool: dict[SpeciesKey, NetworkReaction | None] = {}
        for sp in inventory:
            pool.setdefault(species_key(sp), None)
        changed = True
        while changed:
            changed = False
            for nr in self.reactions:  # fixed canonical order
                if not all(k in pool for k in nr.reactant_keys()):
                    continue
                if not nr.is_live(conditions, rate_cutoff=rate_cutoff, require_rate=require_rate):
                    continue
                for k in nr.product_keys():
                    if k not in pool:
                        pool[k] = nr
                        changed = True
        return pool

    def reachable(self, inventory, conditions: ChemConditions = STANDARD_CHEM, *,
                  rate_cutoff: float = DEFAULT_RATE_CUTOFF,
                  require_rate: bool = True) -> frozenset[SpeciesKey]:
        """The set of species attainable from ``inventory`` under ``conditions`` (spec §12).

        ``inventory`` is any iterable of :class:`~chemistry.reaction.Species`. Returns a
        ``frozenset`` of species keys — order-independent, so calling it twice yields an equal set
        (the determinism contract, §14). Pass ``require_rate=False`` to gate on the genuine ΔG
        threshold alone (dropping the soft kinetic dial).
        """
        return frozenset(self._close(inventory, conditions, rate_cutoff, require_rate))

    def first_producers(self, inventory, conditions: ChemConditions = STANDARD_CHEM, *,
                        rate_cutoff: float = DEFAULT_RATE_CUTOFF,
                        require_rate: bool = True) -> dict[SpeciesKey, NetworkReaction | None]:
        """Like :meth:`reachable`, but maps each reached species to the reaction that first made it
        (``None`` for inventory species). The provenance the explorer uses to draw the tech tree."""
        return self._close(inventory, conditions, rate_cutoff, require_rate)

    def can_reach(self, target: Species, inventory, conditions: ChemConditions = STANDARD_CHEM, *,
                  rate_cutoff: float = DEFAULT_RATE_CUTOFF, require_rate: bool = True) -> bool:
        """True iff ``target`` is reachable from ``inventory`` under ``conditions``."""
        return species_key(target) in self.reachable(
            inventory, conditions, rate_cutoff=rate_cutoff, require_rate=require_rate)

    def unlock_temperature(self, target: Species, inventory, *,
                           t_lo: float = 0.25, t_hi: float = 40.0, tol: float = 1.0e-3,
                           rate_cutoff: float = DEFAULT_RATE_CUTOFF,
                           require_rate: bool = True,
                           base: ChemConditions = STANDARD_CHEM) -> float | None:
        """The lowest temperature at which ``target`` becomes reachable, or ``None`` within range.

        A bisection over T (reachability is monotone-onset for the genuine-threshold demos). This
        is the instrument for the disguised-dial check: re-run it across ``rate_cutoff`` values —
        if the unlock T slides *smoothly*, that gate is the soft kinetic dial, not a real
        threshold (the no-fudge tell, C4). Deterministic: fixed bracket, fixed iteration count.
        """
        def reachable_at(t: float) -> bool:
            return self.can_reach(target, inventory, base.with_temperature(t),
                                  rate_cutoff=rate_cutoff, require_rate=require_rate)

        if reachable_at(t_lo):
            return t_lo
        if not reachable_at(t_hi):
            return None
        lo, hi = t_lo, t_hi
        # ~16 halvings takes the 40-wide bracket below tol; fixed count keeps it deterministic.
        for _ in range(64):
            if hi - lo <= tol:
                break
            mid = 0.5 * (lo + hi)
            if reachable_at(mid):
                hi = mid
            else:
                lo = mid
        return round(hi, 6)


def reaction_network(edges) -> ReactionNetwork:
    """Build a :class:`ReactionNetwork` from edges (``NetworkReaction`` or bare ``Reaction``)."""
    norm = tuple(e if isinstance(e, NetworkReaction) else edge(e) for e in edges)
    return ReactionNetwork(reactions=norm)


# --- a curated demo network (the C5 tech tree the explorer + tests exercise) -----------
# This is example/reference data, NOT engine state — a hand-picked, all-gas-phase-covalent slice
# (so no feasibility decision touches the uncalibrated ionic/metallic scale). The *structure*
# (prerequisites, gates, the locked NO target) is emergent from the thermodynamics, not authored.

def demo_network() -> ReactionNetwork:
    """The C5 demonstration tech tree (see the module docstring for the honest findings).

    Inventory for the keystone is the diatomic gases {H₂, O₂, N₂, Cl₂}. Edges:

    * synthesis (spontaneous cold, rate-trapped until warm): ``2H₂+O₂→2H₂O``, ``H₂+Cl₂→2HCl``;
    * Haber (spontaneous cold, deeply rate-trapped; **Fe** catalyses): ``N₂+3H₂→2NH₃``;
    * dissociations (the GENUINE ΔG gates, ordered by bond strength): ``Cl₂→2Cl``, ``O₂→2O``,
      ``H₂→2H``, ``N₂→2N``;
    * a surviving radical prerequisite: ``O+H₂→H₂O`` (needs atomic O first);
    * the locked target NO: ``N₂+O₂→2NO`` (blocked) and ``N+O→NO`` (window disjoint from atomic N).
    """
    from . import reaction as rx

    H2, O2, N2, Cl2 = (rx.diatomic(s) for s in ("H", "O", "N", "Cl"))
    H2O, HCl, NH3, NO = rx.binary("H", "O"), rx.binary("H", "Cl"), rx.binary("N", "H"), rx.binary("N", "O")
    aH, aO, aN, aCl = (rx.atom(s) for s in ("H", "O", "N", "Cl"))

    edges = [
        edge(rx.reaction(((H2, 2), (O2, 1)), ((H2O, 2),))),
        edge(rx.reaction(((H2, 1), (Cl2, 1)), ((HCl, 2),))),
        edge(rx.reaction(((N2, 1), (H2, 3)), ((NH3, 2),)), catalysts={"Fe"}),
        edge(rx.reaction(((Cl2, 1),), ((aCl, 2),))),
        edge(rx.reaction(((O2, 1),), ((aO, 2),))),
        edge(rx.reaction(((H2, 1),), ((aH, 2),))),
        edge(rx.reaction(((N2, 1),), ((aN, 2),))),
        edge(rx.reaction(((aO, 1), (H2, 1)), ((H2O, 1),))),
        edge(rx.reaction(((N2, 1), (O2, 1)), ((NO, 2),))),
        edge(rx.reaction(((aN, 1), (aO, 1)), ((NO, 1),))),
    ]
    return reaction_network(edges)


def demo_inventory():
    """The starting inventory for the demo keystone: the diatomic gases {H₂, O₂, N₂, Cl₂}."""
    from . import reaction as rx

    return [rx.diatomic("H"), rx.diatomic("O"), rx.diatomic("N"), rx.diatomic("Cl")]
