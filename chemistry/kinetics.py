"""Reaction kinetics: Arrhenius rate, activation energy, catalysts (spec §11 — milestone C4).

Feasibility (ΔG, C3) says a reaction *can* go; **rate** says whether it actually does on any
useful timescale. The rate is the same Boltzmann/exponential machinery the materials engine
already uses for Metropolis acceptance, applied to a transition-state barrier:

* **Arrhenius rate** ``rate = A · exp(−Ea / RT)`` — exponentially sensitive to both the barrier
  and the temperature (the same ``exp(−E/kT)`` form as a spin flip's acceptance).
* **Activation energy Ea** from a distilled transition-state estimate: a fixed fraction of the
  energy of the **reactant** bonds that must break on the way to products
  (``BARRIER_FRACTION · Σ E(reactant bonds)``). A reaction with no bonds to break (radical
  recombination ``2A→A₂``) has Ea≈0 and is barrierless — exactly as in reality.
* **Catalysts** open an alternative pathway with a **lower** barrier (``CATALYST_BARRIER_FACTOR
  · Ea``). A catalyst is authored per-reaction domain data (which species catalyses *this*
  reaction — like knowing Pt lights off H₂+O₂); it affects **rate only**, never ΔG/K, and is
  not consumed (it does not appear in the stoichiometry).

**Keystone (C4):** the "thermodynamically favourable but kinetically trapped" case — an
exergonic reaction (ΔG<0) whose rate is negligible cold because Ea is high, until it is
**ignited** (raise T) or **catalysed** (lower Ea); the catalyst measurably moves the temperature
needed for a given rate **without changing ΔG**.

**Honest caveats (no-fudge norm, flagged not buried):**

* **The "ignition threshold" is not a true transition.** ``rate(T)`` is a smooth, monotone
  exponential; it climbs by orders of magnitude over a narrow window, but there is no genuine
  sign-crossing or singularity here. Real ignition is *sharp* because of **thermal runaway**
  (released heat raises T, which raises rate — a feedback loop) that this single-reaction model
  does **not** include. So we report rate ratios and the temperature-for-a-given-rate; we do
  **not** dress a rate>cutoff boolean as a phase transition (that cutoff is a soft dial — nudging
  it slides the "ignition" temperature smoothly, the tell-tale of a non-transition).
* **Absolute rates are uncalibrated** (``A`` and ``Ea`` are in model units). What is emergent:
  the *exponential* sensitivity to Ea and T, and the catalyst shifting the whole rate(T) curve
  by a fixed factor in the exponent.
* Ea is estimated from reactant bond energies, so it inherits the C1 linear-in-bond-order
  overcount (memory ``c1-bond-order-fix-needed``) — harmless to the keystone (rate gates
  regardless of ΔH sign) but noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .conditions import STANDARD_CHEM, ChemConditions
from .reaction import GAS_CONSTANT, Reaction

# --- distilled kinetic constants (one global set, like the C3 entropy units) ----------
# Ea as a fraction of the reactant bonds that must break (a Hammond-ish transition-state
# estimate); the catalyst path is a fixed fraction of that barrier; A is the attempt frequency.
BARRIER_FRACTION: float = 0.4
CATALYST_BARRIER_FACTOR: float = 0.5
ARRHENIUS_A: float = 1.0e3

_QUANT_DECIMALS: int = 6


def _q(x: float) -> float:
    return round(float(x), _QUANT_DECIMALS)


def _q_sig(x: float, sig: int = 6) -> float:
    """Round to ``sig`` significant figures — rates span many orders of magnitude, so a
    fixed-decimal quantize would flush small rates to zero. Keeps determinism without that."""
    if x == 0.0:
        return 0.0
    d = sig - 1 - math.floor(math.log10(abs(x)))
    return round(x, d)


@dataclass(frozen=True)
class Kinetics:
    """The kinetic layer over a :class:`~chemistry.reaction.Reaction` (spec §11).

    ``catalysts`` is the authored set of species that catalyse *this* reaction (lower its
    barrier when present in the conditions). Thermodynamics is untouched — ``reaction`` owns
    ΔG/K; this object owns Ea and rate.
    """

    reaction: Reaction
    catalysts: frozenset[str] = field(default_factory=frozenset)

    @property
    def reactant_bond_energy(self) -> float:
        """Σ bond energy of the reactants — the bonds that must break in the transition state."""
        return _q(sum(sp.bond_energy * c for sp, c in self.reaction.reactants))

    def base_activation_energy(self) -> float:
        """Uncatalysed Ea = ``BARRIER_FRACTION · Σ E(reactant bonds)`` (transition-state est.)."""
        return _q(BARRIER_FRACTION * self.reactant_bond_energy)

    def is_catalyzed(self, conditions: ChemConditions = STANDARD_CHEM) -> bool:
        """True when a species that catalyses this reaction is present in the conditions."""
        return bool(self.catalysts & conditions.catalysts)

    def activation_energy(self, conditions: ChemConditions = STANDARD_CHEM) -> float:
        """Effective Ea at the conditions: the catalysed (lower) barrier if a catalyst is present."""
        ea = self.base_activation_energy()
        if self.is_catalyzed(conditions):
            ea *= CATALYST_BARRIER_FACTOR
        return _q(ea)

    def rate(self, conditions: ChemConditions = STANDARD_CHEM) -> float:
        """Arrhenius rate ``A · exp(−Ea/RT)`` at the conditions (spec §11). Higher T, lower Ea → faster."""
        ea = self.activation_energy(conditions)
        return _q_sig(ARRHENIUS_A * math.exp(-ea / (GAS_CONSTANT * conditions.temperature)))

    def temperature_for_rate(self, target_rate: float,
                             conditions: ChemConditions = STANDARD_CHEM) -> float | None:
        """The T at which the rate reaches ``target_rate`` (a reference 'fast enough' rate).

        Solving ``A·exp(−Ea/RT) = target`` gives ``T = Ea / (R·ln(A/target))``. This is *not* an
        ignition transition — it is the temperature for a chosen reference rate, and it slides
        smoothly as that reference moves (see the module caveat). Returns ``None`` if the target
        is unreachable (≥ A, the ceiling rate). The catalysed barrier lowers this temperature.
        """
        if target_rate >= ARRHENIUS_A or target_rate <= 0.0:
            return None
        ea = self.activation_energy(conditions)
        denom = GAS_CONSTANT * math.log(ARRHENIUS_A / target_rate)
        return _q(ea / denom)


def kinetics(reaction: Reaction, catalysts: frozenset[str] | set[str] | None = None) -> Kinetics:
    """Build a :class:`Kinetics` for a reaction, optionally with the species that catalyse it."""
    return Kinetics(reaction=reaction, catalysts=frozenset(catalysts or ()))
