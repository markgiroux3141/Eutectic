"""Reaction thermodynamics: ΔH (Hess) / ΔS / ΔG / K, with T/P/concentration (spec §10 — C3).

Feasibility is **measured** from energetics, never assigned:

* **ΔH** by Hess's law over the C1 bonding model: ``ΔH = Σ E(bonds broken) − Σ E(bonds
  formed)`` — exothermic (ΔH<0) when the products are more strongly bound. Bond energies come
  straight from :mod:`chemistry.bonding`; nothing is re-fit here.
* **ΔS** from a distilled, *directional* entropy estimate: each species contributes a molar
  entropy set by its :class:`~chemistry.conditions.Phase` (gas ≫ liquid > solid), so
  ``ΔS = Σ_products c·S(phase) − Σ_reactants c·S(phase)``. More gas moles ⇒ higher S — the
  dominant real-world term, kept deliberately coarse (spec §10, §20).
* **ΔG = ΔH − T·ΔS**, with the **mass-action / Le Chatelier** term folded in:
  ``ΔG = ΔG° + R·T·ln Q``, where the reaction quotient ``Q`` carries the pressure dependence of
  gas species (``P^Δn_gas``) and the concentration dependence (``activity^Δn``). Spontaneous when
  ΔG<0. Equilibrium constant ``K = exp(−ΔG/RT)`` (so K=1 exactly at the feasibility boundary).

**The honest C3 finding (de-risked, reported not buried — spec §20, no-fudge norm).** Hess's
law is only as good as the bond energies under it, and the C1 covalent energy is **linear in
bond order** (``E ∝ order``). Real double/triple bonds are *less* than 2×/3× a single bond (π
bonds are weaker), so the C1 model **overstates** O=O, N≡N, C=O. The consequence, measured in
the de-risk: reactions that *break* a multiple bond come out with the **wrong ΔH sign** —
``2H₂+O₂→2H₂O`` reads +68 (endothermic) where reality is strongly exothermic; likewise the
Haber synthesis and ``H₂+Cl₂→2HCl``. This is the C1 *cross-bond-calibration* caveat
(:mod:`chemistry.bonding`) surfacing one level up. We **do not retune** the bond model to force
these signs (that would be the fudge the project forbids); instead the C3 keystone is anchored
on the class where the sign is **robust to that calibration**:

* **dissociation / recombination** (``A₂ ⇌ 2A``): ΔH = ±E(A–A) — its sign cannot depend on
  comparing *different* bonds, because only one bond type is involved. These are also the
  textbook *entropy-driven* reactions (thermal dissociation), so they carry the keystone's
  T-threshold sign-flip honestly;
* reactions that only **form** multiple bonds (``C+O₂→CO₂``, ``C+2H₂→CH₄``) — no multiple bond
  is broken, so the overcount can't flip the sign.

The wrong-sign reactions are **pinned as tests** (recording the divergence, not hiding it).

**Units & what is / isn't calibrated.** Energies are in C1 model units; entropy in model
entropy units (the ``S_*`` constants below — one global set, like ``COVALENT_K``, not per-
reaction dials). So *absolute* T magnitudes of the ΔG=0 crossings are **uncalibrated**. What is
emergent and parameter-free: the *existence* of a finite positive crossing for entropy-favored
endothermic reactions, the **ordering** of those crossings (weaker bond ⇒ lower threshold:
Cl₂ < H₂ < O₂ in the de-risk), and the **direction** of every Le Chatelier shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.rng import hash_str

from . import bonding
from .atoms import get
from .conditions import STANDARD_CHEM, ChemConditions, Phase
from .molecule import Molecule

# --- distilled entropy units (one global set; gas ≫ liquid > solid) -------------------
# Molar entropy a species contributes by phase. The gas term dominates ΔS (translational
# disorder), exactly as in real chemistry; the absolute scale is a unit choice, not a dial —
# it slides all crossover temperatures together and changes no sign or ordering.
S_GAS: float = 60.0
S_LIQUID: float = 10.0
S_SOLID: float = 2.0
_PHASE_ENTROPY: dict[Phase, float] = {
    Phase.GAS: S_GAS,
    Phase.LIQUID: S_LIQUID,
    Phase.SOLID: S_SOLID,
}

# Model gas constant for K = exp(−ΔG/RT). The real value (so K=1 at ΔG=0, the feasibility
# boundary); with ΔG in model energy units it sets only how sharply K swings, no sign.
GAS_CONSTANT: float = 8.314

# Thermo quantities are quantized before storage/return so float dust can't perturb a
# comparison or a determinism check (spec §14.4); a touch finer than the materials engine's 4.
_QUANT_DECIMALS: int = 6


def _q(x: float) -> float:
    return round(float(x), _QUANT_DECIMALS)


# --- species --------------------------------------------------------------------------

@dataclass(frozen=True)
class Species:
    """A reacting species: its formula, total bond energy, and phase (spec §4).

    ``bond_energy`` is the sum of bond strengths in **one formula unit** (0 for a lone atom —
    it has no bonds), the quantity Hess's law sums over. ``phase`` sets the species' entropy
    contribution. Build via :func:`from_molecule`, :func:`atom`, or :func:`binary`.
    """

    formula: str
    bond_energy: float
    phase: Phase = Phase.GAS

    def molar_entropy(self) -> float:
        return _PHASE_ENTROPY[self.phase]


def from_molecule(mol: Molecule, phase: Phase = Phase.GAS) -> Species:
    """A species from a formed :class:`~chemistry.molecule.Molecule` (its summed bond energy)."""
    return Species(formula=mol.formula, bond_energy=_q(sum(b.energy for b in mol.bonds)), phase=phase)


def atom(symbol: str, phase: Phase = Phase.GAS) -> Species:
    """A free monatomic species (no bonds → zero bond energy), e.g. atomic H, O, Cl."""
    get(symbol)  # validate the symbol exists
    return Species(formula=symbol, bond_energy=0.0, phase=phase)


def binary(symbol_a: str, symbol_b: str, phase: Phase = Phase.GAS) -> Species:
    """A species formed from two elements via the C1 former (e.g. ``binary('H','O')`` → H₂O)."""
    from .molecule import form_binary

    mol = form_binary(symbol_a, symbol_b)
    if not isinstance(mol, Molecule):
        raise ValueError(f"{symbol_a}+{symbol_b} forms no compound: {mol.reason}")
    return from_molecule(mol, phase)


def diatomic(symbol: str, phase: Phase = Phase.GAS) -> Species:
    """The homonuclear diatomic of an element (H→H₂, O→O₂, N→N₂, Cl→Cl₂)."""
    return binary(symbol, symbol, phase)


# --- reactions ------------------------------------------------------------------------

_Side = tuple[tuple[Species, int], ...]


@dataclass(frozen=True)
class Reaction:
    """A balanced reaction: reactant and product ``(species, coefficient)`` multisets (spec §4).

    Thermodynamics is measured, not stored: :attr:`delta_H` and :attr:`delta_S` are derived
    from the species; :meth:`delta_G`, :meth:`equilibrium_constant`, :meth:`is_spontaneous`,
    and :meth:`crossover_temperature` evaluate them at given conditions.
    """

    reactants: _Side
    products: _Side

    # --- ΔH / ΔS (measured from the species, condition-independent) ---
    @property
    def delta_H(self) -> float:
        """Hess's law: ``Σ E(bonds broken) − Σ E(bonds formed)`` (spec §10). ΔH<0 = exothermic."""
        broken = sum(sp.bond_energy * c for sp, c in self.reactants)
        formed = sum(sp.bond_energy * c for sp, c in self.products)
        return _q(broken - formed)

    @property
    def delta_S(self) -> float:
        """Distilled entropy change: ``Σ_products c·S(phase) − Σ_reactants c·S(phase)`` (spec §10)."""
        s_prod = sum(sp.molar_entropy() * c for sp, c in self.products)
        s_react = sum(sp.molar_entropy() * c for sp, c in self.reactants)
        return _q(s_prod - s_react)

    @property
    def delta_n_gas(self) -> int:
        """Change in number of gas-phase moles (the Le Chatelier pressure lever)."""
        g_prod = sum(c for sp, c in self.products if sp.phase.is_gas)
        g_react = sum(c for sp, c in self.reactants if sp.phase.is_gas)
        return g_prod - g_react

    @property
    def delta_n_total(self) -> int:
        """Change in total number of moles (the concentration / mass-action lever)."""
        n_prod = sum(c for _, c in self.products)
        n_react = sum(c for _, c in self.reactants)
        return n_prod - n_react

    # --- ΔG and K at conditions ---
    def _ln_quotient(self, cond: ChemConditions) -> float:
        """``ln Q`` from activities: gas species scale with P (``P^Δn_gas``), all with activity.

        ``ΔG = ΔG° + R·T·ln Q`` is the principled mass-action form; with every gas species at
        pressure ``P`` and activity ``concentration``, ``ln Q = Δn_gas·ln P + Δn_total·ln c``.
        At standard conditions (P=1, c=1) this is 0 and ΔG reduces to ΔG° = ΔH − T·ΔS.
        """
        ln_p = math.log(cond.pressure) if cond.pressure > 0 else 0.0
        ln_c = math.log(cond.concentration)
        return self.delta_n_gas * ln_p + self.delta_n_total * ln_c

    def delta_G_standard(self, temperature: float) -> float:
        """ΔG° = ΔH − T·ΔS at standard activity (the bare temperature dependence)."""
        return _q(self.delta_H - temperature * self.delta_S)

    def delta_G(self, conditions: ChemConditions = STANDARD_CHEM) -> float:
        """ΔG = ΔG° + R·T·ln Q at the given conditions (spec §10). Spontaneous when < 0."""
        t = conditions.temperature
        return _q(self.delta_H - t * self.delta_S + GAS_CONSTANT * t * self._ln_quotient(conditions))

    def is_spontaneous(self, conditions: ChemConditions = STANDARD_CHEM) -> bool:
        """Feasibility boolean: ΔG < 0 (a hard threshold — the genuine sign-crossing, not a dial)."""
        return self.delta_G(conditions) < 0.0

    def equilibrium_constant(self, conditions: ChemConditions = STANDARD_CHEM) -> float:
        """``K = exp(−ΔG/RT)`` (spec §10). K>1 favours products; K=1 at the ΔG=0 boundary."""
        t = conditions.temperature
        return _q(math.exp(-self.delta_G(conditions) / (GAS_CONSTANT * t)))

    def crossover_temperature(self, conditions: ChemConditions = STANDARD_CHEM) -> float | None:
        """The T where ΔG changes sign, or ``None`` if it never crosses for T>0 (spec §10).

        Grouping the temperature-linear terms, ``ΔG = ΔH − T·[ΔS − R·ln Q_unit]`` where
        ``ln Q_unit = Δn_gas·ln P + Δn_total·ln c`` (the per-unit-T quotient slope). The single
        root is ``T* = ΔH / (ΔS − R·ln Q_unit)``; it is a real crossing only when positive (ΔH
        and the effective entropy share a sign). For an entropy-favored endothermic reaction
        (ΔH>0, effective ΔS>0) this is the temperature at which heat switches the reaction on.
        """
        ln_p = math.log(conditions.pressure) if conditions.pressure > 0 else 0.0
        ln_c = math.log(conditions.concentration)
        ln_q_unit = self.delta_n_gas * ln_p + self.delta_n_total * ln_c
        eff_dS = self.delta_S - GAS_CONSTANT * ln_q_unit
        if eff_dS == 0.0:
            return None  # ΔG is flat in T — no crossing
        t_star = self.delta_H / eff_dS
        return _q(t_star) if t_star > 0.0 else None

    # --- identity (deterministic; construction-order-independent) ---
    def canonical_id(self) -> str:
        """Stable id from the sorted reactant/product multisets (spec §14.6)."""
        def side_key(side: _Side) -> str:
            return ";".join(sorted(f"{sp.formula}:{sp.phase.value}*{c}" for sp, c in side))

        key = f"{side_key(self.reactants)}=>{side_key(self.products)}"
        return f"rxn_{hash_str(key):016x}"


def reaction(reactants: _Side, products: _Side) -> Reaction:
    """Build a :class:`Reaction` from ``(species, coefficient)`` reactant/product tuples."""
    return Reaction(reactants=tuple(reactants), products=tuple(products))
