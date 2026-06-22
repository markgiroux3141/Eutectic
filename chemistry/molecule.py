"""Molecule formation by constrained energy minimization (chemistry-engine-spec §8 — C1c).

The "combine" analog at the molecular level: given two element types, find the bonding
configuration (and the stoichiometry) that **minimizes total energy** subject to valence /
charge satisfaction (spec §8). Stoichiometry is *not* assigned — it is the ratio at which the
energy is lowest, which is the ratio at which valence/charge balance.

**Stable compound ⇔ energy minimum.** For the covalent case this is made literal: the formula
unit is chosen by an actual minimization over candidate ratios (:func:`_covalent_energy`), and
the textbook ratio wins because leaving valence unsatisfied (dangling) costs energy. For the
ionic case the minimum is the charge-neutral ratio (any other ratio carries net charge).

Keystone (C1c): Na+Cl→NaCl (1:1), Mg+Cl→MgCl₂ (1:2), H+O→H₂O (2:1), C+O→CO₂ (1:2),
parameter-free; noble gases refuse to bond; a non-opposing pairing has no stabilizing minimum.

**Scope at C1 (flagged):** covalent formation uses the single-central-atom rule (the
higher-capacity atom is central, terminals ring it). Multi-center molecules (C₂H₆, chains,
rings) need the general combinatorial/anneal search (spec §19.2) and are out of the C1
keystone; :func:`form_binary` reports when a ratio would require them rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from engine.rng import hash_str

from . import bonding, orbitals
from .atoms import Atom, get
from .bonding import Bond, BondCharacter
from .orbitals import Geometry

# Energy charged per unit of unsatisfied (dangling) valence in the covalent search. Large
# enough that a fully-satisfied ratio always beats a partially-bonded one — this is what makes
# the textbook stoichiometry the genuine energy minimum (verified in the C1 de-risk).
DANGLING_PENALTY: float = 50.0


@dataclass(frozen=True)
class Molecule:
    """A formula unit: its atoms, bonds, charges, geometry, and measured formation energy.

    ``atoms`` is one element symbol per site index; ``bonds`` reference those indices. For an
    ionic unit the "bonds" are notional cation–anion contacts (the real crystal is C2) and
    ``geometry`` is ``None``; for a covalent molecule with a clear central atom ``geometry``
    carries its VSEPR shape. ``formation_energy`` is negative (stabilizing) — the negated sum
    of bond strengths (spec §4).
    """

    atoms: tuple[str, ...]
    bonds: tuple[Bond, ...]
    formal_charges: tuple[int, ...]
    character: BondCharacter
    counts: dict[str, int]
    formula: str
    formation_energy: float
    geometry: Geometry | None

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def canonical_id(self) -> str:
        """Stable identity from a graph-canonical key (spec §14.5).

        Construction-order-independent: the canonical key is the sorted ``(symbol, count)``
        multiset, the character, and the sorted bond-order multiset, so the same compound
        always hashes identically regardless of how it was built.
        """
        counts_key = ";".join(f"{s}{self.counts[s]}" for s in sorted(self.counts))
        order_key = ",".join(str(o) for o in sorted(b.order for b in self.bonds))
        key = f"{counts_key}|{self.character.value}|{order_key}"
        return f"mol_{hash_str(key):016x}"


def _formula_from_order(counts: dict[str, int], order: list[str]) -> str:
    """Formula string with elements in the given symbol order (1-counts omitted)."""
    return "".join(sym if counts[sym] == 1 else f"{sym}{counts[sym]}" for sym in order)


def hill_formula(counts: dict[str, int]) -> str:
    """Canonical covalent formula in Hill notation (C first, H second, then alphabetical)."""
    def order_key(sym: str) -> tuple[int, str]:
        if "C" in counts:
            if sym == "C":
                return (0, "")
            if sym == "H":
                return (1, "")
        return (2, sym)

    return _formula_from_order(counts, sorted(counts, key=order_key))


def _covalent_energy(central: Atom, term: Atom, n_term: int) -> float:
    """Per-atom formation energy of ``central`` ringed by ``n_term`` terminals (lower = stabler).

    Each terminal forms ``min(term_cap, central_cap // n_term)`` bonds to the centre; whatever
    capacity is left unsatisfied on either side is dangling and penalised. This is the function
    minimized to *discover* the stoichiometry (not a closed-form ratio) — the textbook ratio
    wins because it is the one that leaves nothing dangling.
    """
    cc, ct = central.bonding_capacity, term.bonding_capacity
    bonds_per_term = min(ct, cc // n_term)
    used_central = bonds_per_term * n_term
    if bonds_per_term == 0:
        return float("inf")  # central can't reach this many terminals at all
    e_bonds = -bonding.covalent_bond_energy(central, term, bonds_per_term) * n_term
    dangling = (cc - used_central) + n_term * (ct - bonds_per_term)
    return (e_bonds + DANGLING_PENALTY * dangling) / (1 + n_term)


def _build_covalent(central: Atom, term: Atom, n_term: int) -> Molecule:
    """Assemble the covalent molecule: central at index 0, ``n_term`` terminals after it."""
    cc, ct = central.bonding_capacity, term.bonding_capacity
    order = min(ct, cc // n_term)
    syms = (central.symbol,) + (term.symbol,) * n_term
    bonds = tuple(
        bonding.make_bond(central, term, 0, i + 1, order) for i in range(n_term)
    )
    counts = {central.symbol: 1}
    counts[term.symbol] = counts.get(term.symbol, 0) + n_term
    formation_energy = -sum(b.energy for b in bonds)
    # Geometry only when there is a genuine central atom with >= 2 directions (an angle).
    geometry = (
        Geometry.from_counts(central.symbol, sigma_bonds=n_term, lone_pairs=central.lone_pairs)
        if n_term >= 2
        else None
    )
    return Molecule(
        atoms=syms,
        bonds=bonds,
        formal_charges=(0,) * len(syms),
        character=bond_character_of(central, term),
        counts=counts,
        formula=hill_formula(counts),
        formation_energy=formation_energy,
        geometry=geometry,
    )


def bond_character_of(a: Atom, b: Atom) -> BondCharacter:
    char = bonding.bond_character(a, b)
    assert char is not None  # callers guard noble gases first
    return char


def _build_ionic(cation: Atom, anion: Atom, n_cat: int, n_an: int) -> Molecule:
    """Assemble the ionic formula unit: one cation site bonded to each anion (notional)."""
    # Place cations then anions; bond each cation to the anions it neutralises (notional).
    syms = (cation.symbol,) * n_cat + (anion.symbol,) * n_an
    charges = (cation.ion_charge,) * n_cat + (anion.ion_charge,) * n_an
    bonds = []
    # Notional contacts: connect every cation to every anion in the unit (the real
    # coordination geometry is the C2 crystal). Energy per contact is the ion-pair Coulomb.
    for ci in range(n_cat):
        for ai in range(n_an):
            bonds.append(
                bonding.make_bond(cation, anion, ci, n_cat + ai, order=1)
            )
    counts = {}
    counts[cation.symbol] = counts.get(cation.symbol, 0) + n_cat
    counts[anion.symbol] = counts.get(anion.symbol, 0) + n_an
    formation_energy = -sum(b.energy for b in bonds)
    # Ionic compounds are conventionally written cation-first (not strict Hill alphabetical).
    return Molecule(
        atoms=syms,
        bonds=tuple(bonds),
        formal_charges=charges,
        character=BondCharacter.IONIC,
        counts=counts,
        formula=_formula_from_order(counts, [cation.symbol, anion.symbol]),
        formation_energy=formation_energy,
        geometry=None,
    )


@dataclass(frozen=True)
class NoCompound:
    """Why a pair does not form a compound (so callers see the reason, spec §8)."""

    reason: str


def form_binary(symbol_a: str, symbol_b: str) -> Molecule | NoCompound:
    """Form the minimum-energy compound of two element types, or report why none forms.

    The stoichiometry *emerges*: ionic → the charge-neutral ratio; covalent → the ratio that
    minimizes :func:`_covalent_energy`. Noble gases (zero capacity) and same-sign ion pairs do
    not form. Deterministic and side-effect-free (spec §14).
    """
    a, b = get(symbol_a), get(symbol_b)
    if a.bonding_capacity == 0 or b.bonding_capacity == 0:
        return NoCompound("a noble gas (full shell) gains nothing by bonding")

    char = bonding.bond_character(a, b)
    if char is None:
        return NoCompound("no electronegativity difference defined (noble gas)")

    if char is BondCharacter.IONIC:
        # Charge balance: n_cat·q_cat = n_an·|q_an|, reduced to lowest terms.
        if a.ion_charge > 0 > b.ion_charge:
            cat, an = a, b
        elif b.ion_charge > 0 > a.ion_charge:
            cat, an = b, a
        else:
            return NoCompound("ionic but the preferred charges do not oppose")
        qc, qa = cat.ion_charge, -an.ion_charge
        g = gcd(qc, qa)
        return _build_ionic(cat, an, n_cat=qa // g, n_an=qc // g)

    if char is BondCharacter.METALLIC:
        # Metallic bonding is delocalized; a discrete molecule is only notional here and the
        # real structure is the bulk lattice (C2). Return a 2-atom unit so the character/energy
        # are inspectable, with no stoichiometry search.
        bond = bonding.make_bond(a, b, 0, 1, order=1)
        counts = {a.symbol: 1}
        counts[b.symbol] = counts.get(b.symbol, 0) + 1
        return Molecule(
            atoms=(a.symbol, b.symbol),
            bonds=(bond,),
            formal_charges=(0, 0),
            character=BondCharacter.METALLIC,
            counts=counts,
            formula=hill_formula(counts),
            formation_energy=-bond.energy,
            geometry=None,
        )

    # Covalent / polar covalent: minimize over candidate ratios (central = higher capacity).
    central, term = (a, b) if a.bonding_capacity >= b.bonding_capacity else (b, a)
    cc, ct = central.bonding_capacity, term.bonding_capacity
    best_n, best_e = None, float("inf")
    for n in range(1, cc + 1):
        e = _covalent_energy(central, term, n)
        if e < best_e:
            best_e, best_n = e, n
    if best_n is None:
        return NoCompound("no stabilizing covalent configuration")
    # If the winning ratio still leaves the centre's valence unsatisfied, the keystone
    # single-central model can't represent it (a multi-center molecule would) — say so.
    if cc % best_n != 0 and min(ct, cc // best_n) * best_n < cc:
        return NoCompound(
            f"{central.symbol}/{term.symbol} needs a multi-center structure "
            f"(non-integer single-central ratio) — out of C1 scope"
        )
    return _build_covalent(central, term, best_n)
