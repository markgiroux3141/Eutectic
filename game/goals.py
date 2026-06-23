"""The progression: discovery + build goals that auto-complete (materials-engine-spec sec 11).

A goal is a named objective with a predicate over the :class:`~game.state.GameState`. Goals
complete the moment their predicate holds (checked after every discover/build), and a goal only
becomes *active* once its prerequisites are done — a light tech-tree that steers the player from
"make your first alloy" toward "build a motor that beats the iron-and-copper baseline".

**Discovery goals check combinations only** (materials the player actually made, lineage length
2): owning iron at the start shouldn't tick "find a magnet". Build goals read the best headline
the player has achieved for that machine (the engine/machines compute it; the goal just compares).
Thresholds are calibrated against the real property/performance ranges (not magic numbers): e.g.
the motor baseline is iron-core / copper-coil / iron-shaft ~ 0.116 torque, copper heat-sink
conductance ~ 0.131 — goals ask you to beat them.

This module imports nothing from the game (predicates take the state duck-typed), so there is no
import cycle with :mod:`game.state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Goal:
    """One objective: a predicate over the game state, gated by prerequisites."""

    id: str
    title: str
    description: str
    check: Callable[[Any], bool]
    prereqs: tuple[str, ...] = field(default_factory=tuple)


# --- predicate helpers ----------------------------------------------------------------

def _any_combo(state: Any, pred: Callable[[dict], bool]) -> bool:
    """True iff any DISCOVERED COMBINATION's properties satisfy ``pred``."""
    return any(pred(m.properties) for m in state.combos())


def _machine_at_least(state: Any, machine: str, threshold: float) -> bool:
    best = state.best_machine(machine)
    return best is not None and best >= threshold


# Calibrated reference points (see module docstring / the de-risk calibration run).
_MOTOR_BASELINE = 0.116      # iron core, copper coil, iron shaft
_HEATSINK_COPPER = 0.131     # copper fin conductance
_DEFAULT_THREAT = 0.5        # armor's default threat level


# --- the goal catalog (a light prerequisite tree) -------------------------------------

GOALS: tuple[Goal, ...] = (
    Goal("first_alloy", "First Alloy",
         "Combine two materials to discover a new one.",
         lambda s: any(True for _ in s.combos())),

    Goal("conductor", "Spark of Life",
         "Discover an alloy that conducts electricity.",
         lambda s: _any_combo(s, lambda p: p.get("conductivity", 0) >= 1.0),
         prereqs=("first_alloy",)),

    Goal("lodestone", "Lodestone",
         "Discover a strongly magnetic alloy (magnetism > 0.5).",
         lambda s: _any_combo(s, lambda p: p.get("magnetism", 0) > 0.5),
         prereqs=("first_alloy",)),

    Goal("featherweight", "Featherweight",
         "Discover a very light alloy (density < 1.0).",
         lambda s: _any_combo(s, lambda p: 0 < p.get("density", 0) < 1.0),
         prereqs=("first_alloy",)),

    Goal("refractory", "Forged in Fire",
         "Discover a refractory alloy (melting point > 3.0).",
         lambda s: _any_combo(s, lambda p: p.get("melting_temperature", 0) > 3.0),
         prereqs=("conductor",)),

    Goal("permanent_magnet", "Permanent Magnet",
         "Discover an alloy that stays magnetic when hot (magnetism > 0.4 and Curie > 1.5).",
         lambda s: _any_combo(s, lambda p: p.get("magnetism", 0) > 0.4
                              and p.get("curie_temperature", 0) > 1.5),
         prereqs=("lodestone",)),

    Goal("titan", "Titan",
         "Discover a strong alloy (shear strength > 0.12).",
         lambda s: _any_combo(s, lambda p: p.get("strength", 0) > 0.12),
         prereqs=("refractory",)),

    # --- build goals ---
    Goal("working_motor", "It Turns!",
         "Build a motor that produces torque.",
         lambda s: _machine_at_least(s, "motor", 1e-6),
         prereqs=("conductor",)),

    Goal("power_motor", "Power Plant",
         f"Build a motor that beats the iron-and-copper baseline (torque > {_MOTOR_BASELINE}).",
         lambda s: _machine_at_least(s, "motor", _MOTOR_BASELINE + 1e-9),
         prereqs=("working_motor",)),

    Goal("lifter", "Heavy Lifter",
         "Build an electromagnet that generates lift.",
         lambda s: _machine_at_least(s, "electromagnet", 1e-6),
         prereqs=("working_motor",)),

    Goal("cool_runner", "Cooler Than Copper",
         f"Build a heat sink that beats copper (conductance > {_HEATSINK_COPPER}).",
         lambda s: _machine_at_least(s, "heatsink", _HEATSINK_COPPER + 1e-9),
         prereqs=("first_alloy",)),

    Goal("aegis", "Aegis",
         f"Build composite armor that stops the standard threat (protection >= {_DEFAULT_THREAT}).",
         lambda s: _machine_at_least(s, "armor", _DEFAULT_THREAT),
         prereqs=("titan",)),
)

BY_ID: dict[str, Goal] = {g.id: g for g in GOALS}


def active_goals(completed: set[str]) -> list[Goal]:
    """Goals whose prerequisites are all complete and which are not themselves complete."""
    return [g for g in GOALS
            if g.id not in completed and all(p in completed for p in g.prereqs)]
