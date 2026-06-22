"""Bonding model: character, order, energy (chemistry-engine-spec §7 — milestone C1b).

A bond's **character** *emerges* from the electronegativity difference ΔEN (Pauling), its
**order** from the electron pairs the two atoms share, and its **energy** from a distilled,
character-specific model. Nothing here is assigned per-pair; the thresholds and the one or
two calibration constants are the only authored numbers (spec §7).

The ΔEN read here is the **authored Pauling** electronegativity, *not* the derived Z_eff
proxy — the C0 de-risk showed the proxy is trend-correct but not Pauling-calibrated, so the
quantitative ΔEN thresholds below would be wrong on the proxy scale (see
:mod:`chemistry.atoms`). Trends were the proxy's job; absolute ΔEN is the authored value's.

**Honest scope at C1 (flagged, carried to C2):** the ionic energy here is the single ion-pair
Coulomb term ``z⁺z⁻/(r⁺+r⁻)``. The large "ionic-strong" *lattice* energy is the Madelung sum
over the crystal — a C2 deliverable. So covalent and ionic energies are on **separate, not yet
cross-calibrated scales**; C1 claims only *within-character* ordering (e.g. triple > double >
single), never "ionic vs covalent" magnitude comparison.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .atoms import Atom

# --- ΔEN thresholds (Pauling-style, spec §7) ------------------------------------------
IONIC_DELTA_EN: float = 1.7          # ΔEN >= this -> ionic (electron transfer)
POLAR_DELTA_EN: float = 0.4          # ΔEN in [this, IONIC) -> polar covalent
# Below POLAR_DELTA_EN the bond is covalent UNLESS both atoms are electropositive (metals),
# in which case it is metallic. The discriminator is absolute electronegativity: two atoms
# both below this ceiling are metals sharing a delocalized electron sea (Cu-Cu), while two
# above it share localized pairs (Cl-Cl). One calibrated constant near the metalloid line.
METALLIC_EN_CEILING: float = 2.0

# --- energy calibration constants (spec §7: "a fixed scale, not a per-bond dial") ------
# Each character has ONE proportionality constant. They are NOT mutually calibrated at C1
# (see module docstring) — cross-character magnitude comparison waits for C2's Madelung.
COVALENT_K: float = 100.0
IONIC_K: float = 100.0
METALLIC_K: float = 100.0


class BondCharacter(enum.Enum):
    """How a bond holds together, from ΔEN + electropositivity (spec §7)."""

    IONIC = "ionic"
    POLAR_COVALENT = "polar_covalent"
    COVALENT = "covalent"
    METALLIC = "metallic"

    @property
    def is_covalent_like(self) -> bool:
        """Shares electron pairs (covalent or polar) rather than transferring/delocalizing."""
        return self in (BondCharacter.COVALENT, BondCharacter.POLAR_COVALENT)


def delta_en(a: Atom, b: Atom) -> float | None:
    """Pauling electronegativity difference, or ``None`` if either atom has no Pauling EN."""
    if a.electronegativity is None or b.electronegativity is None:
        return None
    return abs(a.electronegativity - b.electronegativity)


def bond_character(a: Atom, b: Atom) -> BondCharacter | None:
    """Bond character from ΔEN, with the covalent/metallic split by absolute EN (spec §7).

    Returns ``None`` when ΔEN is undefined (a noble gas with no Pauling EN) — those do not
    bond. Keystone (C1b): NaCl ionic, Cl₂ covalent, Cu metallic.
    """
    den = delta_en(a, b)
    if den is None:
        return None
    if den >= IONIC_DELTA_EN:
        return BondCharacter.IONIC
    if den >= POLAR_DELTA_EN:
        return BondCharacter.POLAR_COVALENT
    # ΔEN < 0.4: covalent unless both atoms are electropositive metals.
    if max(a.electronegativity, b.electronegativity) < METALLIC_EN_CEILING:  # type: ignore[arg-type]
        return BondCharacter.METALLIC
    return BondCharacter.COVALENT


# --- bond energy by character (distilled; MEASURED, not assigned) ---------------------

def covalent_bond_energy(a: Atom, b: Atom, order: int) -> float:
    """Orbital-overlap model: ``K · order · EN_avg / (r_a + r_b)`` (spec §7).

    Closer, higher-order, more-electronegative pairs bond harder. Positive = bond strength.
    """
    en_avg = (a.electronegativity + b.electronegativity) / 2.0  # type: ignore[operator]
    return COVALENT_K * order * en_avg / (a.covalent_radius + b.covalent_radius)


def ionic_bond_energy(cation: Atom, anion: Atom, q_cation: int, q_anion: int) -> float:
    """Single ion-pair Coulomb energy: ``K · z⁺z⁻ / (r⁺ + r⁻)`` (spec §7).

    ``q_anion`` is passed as a positive magnitude. NB this is the *pair* energy; the large
    lattice (Madelung) energy is C2. Positive = bond strength.
    """
    return IONIC_K * (q_cation * q_anion) / (cation.covalent_radius + anion.covalent_radius)


def metallic_bond_energy(a: Atom, b: Atom) -> float:
    """Electron-sea cohesion ``∝ valence-electron density × overlap`` (spec §7).

    Notional at C1 (a single-pair stand-in); real metallic cohesion is the delocalized bulk
    lattice (C2). Positive = bond strength.
    """
    ve = (a.valence_electrons + b.valence_electrons) / 2.0
    return METALLIC_K * ve / (a.covalent_radius + b.covalent_radius)


@dataclass(frozen=True)
class Bond:
    """A bond between two atoms in a molecule (spec §4): indices, order, character, energy."""

    a_index: int
    b_index: int
    order: int
    character: BondCharacter
    energy: float       # bond strength (positive); formation energy negates the sum


def make_bond(a: Atom, b: Atom, a_index: int, b_index: int, order: int) -> Bond:
    """Construct a :class:`Bond`, measuring character + energy from the two atoms (spec §7)."""
    char = bond_character(a, b)
    if char is None:
        raise ValueError(f"{a.symbol}-{b.symbol} do not bond (noble gas / no ΔEN)")
    if char is BondCharacter.IONIC:
        # orient by ion sign; magnitudes from preferred ion charges
        if a.ion_charge > 0 > b.ion_charge:
            energy = ionic_bond_energy(a, b, a.ion_charge, -b.ion_charge)
        elif b.ion_charge > 0 > a.ion_charge:
            energy = ionic_bond_energy(b, a, b.ion_charge, -a.ion_charge)
        else:
            energy = ionic_bond_energy(a, b, abs(a.ion_charge), abs(b.ion_charge))
    elif char is BondCharacter.METALLIC:
        energy = metallic_bond_energy(a, b)
    else:
        energy = covalent_bond_energy(a, b, order)
    return Bond(a_index=a_index, b_index=b_index, order=order, character=char, energy=energy)
