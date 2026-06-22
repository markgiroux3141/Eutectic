"""Hybridization + VSEPR geometry (chemistry-engine-spec §6 — milestone C1a).

Molecular geometry *emerges from electron counting*, not assignment (spec §6):

1. **Steric number** = σ-bonds + lone pairs on the central atom (a double/triple bond is
   one σ for VSEPR — it occupies one direction).
2. **Hybridization** from the steric number (2→sp … 6→sp³d²); the d-bearing 5/6 cases are
   why the orbital model (:func:`chemistry.atoms.available_orbitals`) matters.
3. **Idealized bond angle** from the steric number, **compressed by lone-pair repulsion** —
   one calibrated constant (~2.5°/lone pair) reproduces both NH₃ (107°) and H₂O (104.5°)
   from the same tetrahedral 109.5° base.

Keystone (C1a): CH₄ tetrahedral 109.5°, H₂O bent ~104.5°, CO₂ linear 180°, NH₃ trigonal
pyramidal — all from steric number + lone pairs, no per-molecule rule.
"""

from __future__ import annotations

from dataclasses import dataclass

# Steric number -> (hybridization, idealized dominant bond angle in degrees).
_STERIC: dict[int, tuple[str, float]] = {
    2: ("sp", 180.0),       # linear
    3: ("sp2", 120.0),      # trigonal planar
    4: ("sp3", 109.5),      # tetrahedral
    5: ("sp3d", 120.0),     # trigonal bipyramidal (equatorial angle)
    6: ("sp3d2", 90.0),     # octahedral
}

# Degrees a single lone pair compresses the bond angle (VSEPR: lp-bp repulsion > bp-bp).
# One calibrated constant; reproduces NH₃ 107° (1 lp) and H₂O 104.5° (2 lp) off 109.5°.
LONE_PAIR_COMPRESSION: float = 2.5

# Shape name from (steric_number, lone_pairs) — the recognizable VSEPR labels.
_SHAPE: dict[tuple[int, int], str] = {
    (2, 0): "linear",
    (3, 0): "trigonal planar", (3, 1): "bent",
    (4, 0): "tetrahedral", (4, 1): "trigonal pyramidal", (4, 2): "bent",
    (5, 0): "trigonal bipyramidal", (5, 1): "seesaw", (5, 2): "t-shaped", (5, 3): "linear",
    (6, 0): "octahedral", (6, 1): "square pyramidal", (6, 2): "square planar",
}


def steric_number(sigma_bonds: int, lone_pairs: int) -> int:
    """σ-bonds + lone pairs on the central atom (multiple bonds count as one σ)."""
    return sigma_bonds + lone_pairs


def hybridization(sn: int) -> str:
    """Hybridization label from the steric number (2→sp … 6→sp³d²)."""
    try:
        return _STERIC[sn][0]
    except KeyError:
        raise ValueError(f"steric number {sn} outside the modelled 2..6 range") from None


def ideal_angle(sn: int) -> float:
    """Idealized (lone-pair-free) bond angle from the steric number, in degrees."""
    try:
        return _STERIC[sn][1]
    except KeyError:
        raise ValueError(f"steric number {sn} outside the modelled 2..6 range") from None


def bond_angle(sn: int, lone_pairs: int) -> float:
    """Bond angle with lone-pair compression applied (spec §6)."""
    return ideal_angle(sn) - LONE_PAIR_COMPRESSION * lone_pairs


def shape(sigma_bonds: int, lone_pairs: int) -> str:
    """VSEPR shape name from σ-bonds + lone pairs (e.g. 'bent', 'trigonal pyramidal')."""
    sn = steric_number(sigma_bonds, lone_pairs)
    return _SHAPE.get((sn, lone_pairs), f"SN{sn}-lp{lone_pairs}")


@dataclass(frozen=True)
class Geometry:
    """The VSEPR geometry of a molecule's central atom (spec §4, §6)."""

    central: str            # central atom symbol
    sigma_bonds: int
    lone_pairs: int
    steric_number: int
    hybridization: str
    bond_angle: float       # degrees, lone-pair-compressed
    shape: str

    @classmethod
    def from_counts(cls, central: str, sigma_bonds: int, lone_pairs: int) -> "Geometry":
        sn = steric_number(sigma_bonds, lone_pairs)
        return cls(
            central=central,
            sigma_bonds=sigma_bonds,
            lone_pairs=lone_pairs,
            steric_number=sn,
            hybridization=hybridization(sn),
            bond_angle=bond_angle(sn, lone_pairs),
            shape=shape(sigma_bonds, lone_pairs),
        )
