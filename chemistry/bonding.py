"""Bonding model: character, order, energy (chemistry-engine-spec §7 — milestone C1b).

A bond's **character** *emerges* from the electronegativity difference ΔEN (Pauling), its
**order** from the electron pairs the two atoms share, and its **energy** from a distilled,
character-specific model. Nothing here is assigned per-pair; the thresholds and the one or
two calibration constants are the only authored numbers (spec §7).

The ΔEN read here is the **authored Pauling** electronegativity, *not* the derived Z_eff
proxy — the C0 de-risk showed the proxy is trend-correct but not Pauling-calibrated, so the
quantitative ΔEN thresholds below would be wrong on the proxy scale (see
:mod:`chemistry.atoms`). Trends were the proxy's job; absolute ΔEN is the authored value's.

**Covalent energy recalibration (post-C4, the C3↔C1 fix).** The covalent bond energy is the
**Pauling model** in real kJ/mol — ``√(E_AA·E_BB) + k·(Δχ)²`` times a sublinear bond-order
factor (see :func:`covalent_bond_energy`). It replaced the original ``order·EN_avg/(r)`` form
after a de-risk showed that form correlated only **r≈0.32** with real single-bond energies and
was linear in bond order, which flipped the ΔH sign of every combustion/synthesis reaction in
C3 (2H₂+O₂ read endothermic). The replacement was the *honest* fix the no-fudge norm demanded:
homonuclear single-bond energies are authored real reference data, the ionic constant (96.5)
and bond-order α (~0.9) are calibrated from **independent** bond data, held-out heteronuclear
bonds are then predicted at **r≈0.97**, and the correct combustion signs fall out as a
*consequence* — never fitted to a reaction target. (A bond-order-only patch was rejected: at the
honest α≈0.9, combustion stayed endothermic — the form, not the order, was the real culprit.)

**Honest scope still carried:** (1) O=O / N≡N are built from the *anomalously weak* O–O / N–N
single bonds × the order factor, so multiple bonds on O/N are **underestimated** (magnitudes
off, signs right). (2) the **ionic** energy here is the single ion-pair Coulomb term
``z⁺z⁻/(r⁺+r⁻)`` and **metallic** its own electron-sea term; neither is cross-calibrated to the
covalent kJ/mol scale (the Madelung lattice energy is still future work). So only the *covalent*
channel is now on a real scale; cross-character magnitude comparison remains out of scope.
"""

from __future__ import annotations

import enum
import math
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
# Ionic / metallic each keep ONE proportionality constant; they are NOT cross-calibrated to
# the covalent kJ/mol scale (cross-character magnitude comparison still waits for a Madelung
# treatment — see module docstring). The covalent energy is now the Pauling model (below).
IONIC_K: float = 100.0
METALLIC_K: float = 100.0

# --- covalent bond energy: the Pauling model (real kJ/mol) ----------------------------
# Authored homonuclear single-bond energies (kJ/mol) — distilled REAL reference data, exactly
# as ``atomic_mass`` is. The Pauling bond-energy equation builds every heteronuclear single
# bond from these plus an ionic-resonance term, predicting held-out real bonds at r≈0.97 (vs
# r≈0.32 for the old EN_avg/(r) form — see the recalibration note in the module docstring).
# Defined for the covalent-forming elements; metals fall back to a coarse radius estimate
# (flagged in :func:`_homonuclear_energy`) since their real bonding is metallic/ionic.
HOMONUCLEAR_SINGLE_BOND_ENERGY: dict[str, float] = {
    "H": 436.0, "B": 293.0, "C": 346.0, "N": 167.0, "O": 146.0, "F": 155.0,
    "Si": 222.0, "P": 201.0, "S": 266.0, "Cl": 242.0, "Br": 193.0, "I": 151.0,
}
# Pauling's ionic-resonance constant (kJ/mol): the extra stabilisation a polar bond gains,
# ``k·(Δχ)²``. A published physical constant, not a tuned dial.
IONIC_RESONANCE_K: float = 96.5
# Each bond beyond the first adds ~α× the first bond's energy (the π bonds are weaker than the
# σ). α≈0.9 is the mean of the clean σ/π ratios in real C–C/C=C/C≡C, C–N…, C–O… series
# (range 0.71–1.09) — calibrated from INDEPENDENT bond data, never from a reaction target.
BOND_ORDER_ALPHA: float = 0.9
# Coarse fallback E(A–A) ≈ scale / covalent_radius for elements with no authored homonuclear
# value (only un-tabulated metals, reached only via the rare polar metal+nonmetal path; never
# a keystone). Tuned once to the order of the authored set, flagged as approximate.
_FALLBACK_HOMONUCLEAR_SCALE: float = 250.0


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

def _homonuclear_energy(a: Atom) -> float:
    """Authored E(A–A) single-bond energy (kJ/mol), or a coarse radius fallback (flagged).

    The fallback (``scale / covalent_radius``) is only reached for un-tabulated elements —
    in practice metals, and only via the rare polar metal+nonmetal covalent path; no keystone
    uses it. Their real bonding is metallic/ionic (their own energy functions).
    """
    e = HOMONUCLEAR_SINGLE_BOND_ENERGY.get(a.symbol)
    if e is not None:
        return e
    return _FALLBACK_HOMONUCLEAR_SCALE / a.covalent_radius


def covalent_bond_energy(a: Atom, b: Atom, order: int) -> float:
    """Pauling bond-energy model (real kJ/mol): ``√(E_AA·E_BB) + k·(Δχ)²``, × a bond-order factor.

    The geometric mean of the two homonuclear single-bond energies (the covalent contribution)
    plus an ionic-resonance term that grows with the electronegativity difference — Pauling's
    classic equation. The bond-order factor ``(1 + α·(order−1))`` makes each additional (π) bond
    weaker than the first (σ). Positive = bond strength.

    This *replaces* the old ``order·EN_avg/(r)`` form, which correlated only r≈0.32 with real
    single-bond energies (it over-rewarded high-EN/small-radius pairs like F–F/O–O and was
    linear in order). The Pauling form predicts held-out heteronuclear bonds at r≈0.97 (MAE
    ~27 kJ/mol) and — as a *consequence*, not a fit — gives the correct sign for combustion/
    synthesis enthalpies (2H₂+O₂ ≈ −453 kJ/mol vs real −482), which C3 previously got wrong.

    **Residual honest limits (carried):** O=O and N≡N are built from the *anomalously weak*
    O–O / N–N single bonds × the order factor, so multiple bonds on O/N are **underestimated**
    (magnitudes off — e.g. O₂ dissociation ~277 vs real 498 — though the reaction *signs* stay
    right). And ionic/metallic energies remain on their own (uncalibrated) scales: only the
    covalent channel is now real kJ/mol.
    """
    base = math.sqrt(_homonuclear_energy(a) * _homonuclear_energy(b))
    if a.electronegativity is not None and b.electronegativity is not None:
        base += IONIC_RESONANCE_K * (a.electronegativity - b.electronegativity) ** 2
    return base * (1 + BOND_ORDER_ALPHA * (order - 1))


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
