"""The thin role / requirement framework for the machine layer (spec §8).

A machine is an *assembly*: it declares named **roles**, each role reads a few of a
material's measured properties and reports a soft ``0..1`` **suitability** — "how well does
this material fit this slot".

Design choice for this layer (a deliberate fork): **requirements are emergent, not gates.**
The suitability score never *blocks* a build. A wrong material is punished by the machine's
own performance equations (a non-conducting wire carries no current → zero torque; a
non-ferromagnetic core has no flux), exactly the way the engine lets properties *emerge*
from structure rather than assigning them (spec §1, in spirit). ``suitability`` is a
UX/legibility readout only — it tells a player *why* a slot is a poor fit before they see
the performance collapse, without ever overriding the physics.

This module is engine-agnostic: it consumes ``Material.properties`` (a plain ``dict``) and
nothing else, so the engine never has to import it (spec §2, §8). Anything with a
``.properties`` mapping (a :class:`engine.material.Material`) or a bare dict works.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _props(material_or_props: Any) -> Mapping[str, float]:
    """Accept either a ``Material`` (duck-typed: has ``.properties``) or a bare dict."""
    props = getattr(material_or_props, "properties", material_or_props)
    if not isinstance(props, Mapping):
        raise TypeError(
            "expected a Material (with .properties) or a properties mapping, "
            f"got {type(material_or_props).__name__}"
        )
    return props


@dataclass(frozen=True)
class Requirement:
    """One property a role cares about, and the scale at which it is ~satisfied.

    ``ref`` is the property value at which this requirement's term reaches ``1.0`` (fully
    met). ``higher_is_better`` chooses the direction: for ``True`` the term is
    ``value / ref`` clamped to ``[0, 1]`` (more is better, e.g. magnetism); for ``False``
    it is ``ref / value`` clamped (less is better, e.g. a raw resistance). The term is a
    pure, dimensionless ``0..1`` reading — never a gate.
    """

    prop: str
    ref: float
    higher_is_better: bool = True
    weight: float = 1.0
    description: str = ""

    def term(self, props: Mapping[str, float]) -> float:
        """This requirement's ``0..1`` satisfaction term for a property mapping."""
        value = float(props.get(self.prop, 0.0))
        if self.higher_is_better:
            t = value / self.ref if self.ref > 0.0 else 0.0
        else:
            t = self.ref / value if value > 0.0 else 1.0
        return max(0.0, min(1.0, t))


@dataclass(frozen=True)
class Role:
    """A named slot in an assembly and the requirements a material fills it against."""

    name: str
    requirements: tuple[Requirement, ...]
    description: str = ""

    def terms(self, material_or_props: Any) -> dict[str, float]:
        """Per-requirement ``0..1`` terms — the legible breakdown behind the score."""
        props = _props(material_or_props)
        return {r.prop: r.term(props) for r in self.requirements}

    def suitability(self, material_or_props: Any) -> float:
        """Soft ``0..1`` fit score: the weighted **geometric** mean of the terms.

        A geometric mean drives the whole score to ``0`` if *any* requirement is unmet
        (a term hits ``0``), expressing "all requirements must hold" — but it is still only
        a readout, never a gate (see the module docstring).
        """
        if not self.requirements:
            return 1.0
        props = _props(material_or_props)
        weight_sum = sum(r.weight for r in self.requirements)
        log_sum = 0.0
        for r in self.requirements:
            t = r.term(props)
            if t <= 0.0:
                return 0.0
            log_sum += r.weight * math.log(t)
        return math.exp(log_sum / weight_sum)


@dataclass(frozen=True)
class Blueprint:
    """An assembly template: a set of named roles. Performance lives in the machine module.

    The blueprint knows *what slots exist and what each wants*; how the assigned materials
    combine into stats (the real-ish equations) is the concrete machine's job
    (e.g. :func:`machines.motor.build_motor`). This split keeps the framework reusable for a
    second machine without baking motor physics into it.
    """

    name: str
    roles: tuple[Role, ...]

    def role(self, name: str) -> Role:
        for r in self.roles:
            if r.name == name:
                return r
        raise KeyError(f"{self.name} has no role {name!r}")

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.roles)

    def suitabilities(self, assignment: Mapping[str, Any]) -> dict[str, float]:
        """Map each role name to the suitability of the material assigned to it.

        ``assignment`` maps role name → ``Material`` (or properties dict). Missing roles are
        an error (the assembly is incomplete); extra keys are ignored.
        """
        missing = [name for name in self.role_names if name not in assignment]
        if missing:
            raise KeyError(f"{self.name} assignment missing role(s): {missing}")
        return {r.name: r.suitability(assignment[r.name]) for r in self.roles}
