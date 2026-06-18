"""Composite-armor worked example (spec §8) — the machine that *solves* the M8 dilemma.

M8 showed strength and ductility **anti-correlate** (corr ≈ −0.81): a material that resists
deformation (high strength → a hard projectile-defeating face) tends to be brittle (low ductility
→ it shatters), and a material that absorbs energy without cracking (high ductility) tends to be
weak. No single material is both. Real armor solves this with a **composite**: a hard face on a
ductile backing. So this machine has *two* roles, deliberately pulling the two opposite ends of the
anti-correlation:

* ``hard_face``      — resists penetration; reads ``strength`` (+ ``bulk_modulus`` stiffness). M8.
* ``ductile_backing`` — holds the cracked face together and catches fragments; reads ``ductility``
  (and prefers low ``density`` for weight). M8.

Performance::

    face_resistance  = strength_face + FACE_STIFF_W * bulk_modulus_face
    backing_support  = clamp(ductility_backing / DUCT_REF, 0, 1)   # 0 if the backing can't deform
    protection       = PROTECT_K * face_resistance * backing_support
    areal_mass       = density_face + density_backing
    specific_protection = protection / areal_mass
    stopped          = protection >= threat
    penetration      = max(0, 1 - protection / threat)             # 0 = fully stopped

The dilemma is *visible and solved*: protection needs **both** a hard face and a ductile backing,
so the best plate combines a brittle-strong material (tungsten/carbon face) with a soft-ductile one
(aluminium/gold backing). Using one material for both is strictly worse — a tungsten backing barely
deforms (low support), an aluminium face has no strength. No gates: a missing property just zeroes
its term and the protection collapses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .roles import Blueprint, Requirement, Role, _props

ARMOR = Blueprint(
    name="composite_armor",
    roles=(
        Role(
            name="hard_face",
            description="defeats/blunts the projectile; wants high strength + stiffness (brittle is fine)",
            requirements=(
                Requirement("strength", ref=0.15, description="resists penetration"),
                Requirement("bulk_modulus", ref=0.30, description="stiffness: spreads the load"),
            ),
        ),
        Role(
            name="ductile_backing",
            description="absorbs residual energy without cracking; wants high ductility, low weight",
            requirements=(
                Requirement("ductility", ref=0.40, description="deforms without shattering"),
                Requirement("density", ref=30.0, higher_is_better=False, description="lighter is better"),
            ),
        ),
    ),
)

# Fixed design/calibration scales (not per-material dials).
PROTECT_K: float = 2.5     # overall protection scale
FACE_STIFF_W: float = 0.3  # how much bulk_modulus adds to the face's penetration resistance
DUCT_REF: float = 0.40     # ductility at which the backing fully supports the face (support -> 1)


@dataclass(frozen=True)
class ArmorPerformance:
    protection: float
    specific_protection: float       # protection per unit areal mass — the figure of merit
    areal_mass: float
    face_resistance: float
    backing_support: float           # 0..1: the backing's ability to hold the face together
    stopped: bool                    # protection >= threat
    penetration: float               # 0 (stopped) .. 1 (defeated)
    suitabilities: dict[str, float]

    def summary(self) -> str:
        flag = "STOPPED" if self.stopped else f"PENETRATED ({self.penetration:.0%})"
        suit = "  ".join(f"{k}={v:.2f}" for k, v in self.suitabilities.items())
        return (
            f"protection={self.protection:.4f}  vs threat -> {flag}\n"
            f"face_resistance={self.face_resistance:.4f}  backing_support={self.backing_support:.3f}\n"
            f"areal_mass={self.areal_mass:.3f}  specific_protection (per mass) = {self.specific_protection:.5f}\n"
            f"suitability: {suit}"
        )


def build_armor(hard_face: Any, ductile_backing: Any, threat: float = 0.5) -> ArmorPerformance:
    """Build a composite plate from a face + a backing material; compute its protection (spec §8).

    Pure function of ``(hard_face, ductile_backing, threat)``; consumes ``Material.properties`` only.
    """
    face = _props(hard_face)
    backing = _props(ductile_backing)

    face_resistance = (
        float(face.get("strength", 0.0)) + FACE_STIFF_W * float(face.get("bulk_modulus", 0.0))
    )
    backing_support = max(0.0, min(1.0, float(backing.get("ductility", 0.0)) / DUCT_REF))
    protection = PROTECT_K * face_resistance * backing_support
    areal_mass = float(face.get("density", 0.0)) + float(backing.get("density", 0.0))
    specific_protection = protection / areal_mass if areal_mass > 0.0 else 0.0
    stopped = protection >= threat
    penetration = max(0.0, 1.0 - protection / threat) if threat > 0.0 else 0.0

    return ArmorPerformance(
        protection=protection,
        specific_protection=specific_protection,
        areal_mass=areal_mass,
        face_resistance=face_resistance,
        backing_support=backing_support,
        stopped=stopped,
        penetration=penetration,
        suitabilities=ARMOR.suitabilities(
            {"hard_face": face, "ductile_backing": backing}
        ),
    )
