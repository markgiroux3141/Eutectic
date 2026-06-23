"""Game shell tests (materials-engine-spec sec 11): discovery, goals, building, save/load.

The game is a thin deterministic layer over the engine, so the tests assert the player-facing
contracts: alchemy is reproducible (same combination -> same material AND same coined name),
goals complete from real measured properties, machines build through the catalog, and a save
round-trips exactly (it replays the combo list, never serializing lattices).
"""

import pytest

from game.naming import coin_name
from game.session import Session
from game.state import GameState
from game import goals as goals_mod


# --- naming / determinism -------------------------------------------------------------

def test_coined_name_is_deterministic_and_pronounceable():
    assert coin_name("m_93ef3ea683c87cad") == coin_name("m_93ef3ea683c87cad")
    name = coin_name("m_93ef3ea683c87cad")
    assert name.isalpha() and len(name) == 5


def test_discovery_is_deterministic_across_sessions():
    a, b = GameState(), GameState()
    ca, _ = a.discover("iron", "copper")
    cb, _ = b.discover("iron", "copper")
    assert ca.id == cb.id                       # same engine material
    assert a.handle_of(ca.id) == b.handle_of(cb.id)   # ...and the same coined handle


def test_rediscovery_is_not_new():
    s = Session()
    first = s.discover("iron", "copper")
    again = s.discover("iron", "copper")
    assert any("discovered" in ln for ln in first)
    assert any("already known" in ln for ln in again)


# --- goals complete from real measured properties -------------------------------------

def test_first_alloy_and_conductor_goals_complete():
    s = Session()
    s.discover("iron", "copper")        # a conducting alloy (cond=1) per calibration
    assert "first_alloy" in s.state.completed
    assert "conductor" in s.state.completed


def test_goal_prereqs_gate_activation():
    state = GameState()
    # with nothing done, only the prereq-free goals are active
    active_ids = {g.id for g in goals_mod.active_goals(state.completed)}
    assert "first_alloy" in active_ids
    assert "refractory" not in active_ids        # needs 'conductor' (which needs 'first_alloy')


def test_build_motor_records_score_and_completes_goal():
    s = Session()
    s.discover("iron", "copper")                 # completes 'conductor' -> unlocks 'working_motor'
    out = s.build("motor", "iron", "copper", "iron")
    assert s.state.best_machine("motor") > 0.0
    assert "working_motor" in s.state.completed
    assert any("torque" in ln for ln in out)


def test_build_wrong_arity_is_a_friendly_error():
    s = Session()
    out = s.build("motor", "iron")               # motor needs 3 materials
    assert any("needs 3" in ln or "roles" in ln for ln in out)


def test_unknown_handle_and_machine_are_graceful():
    s = Session()
    assert any("no material" in ln for ln in s.inspect("nonsuch"))
    assert any("unknown machine" in ln for ln in s.build("frobnicator", "iron"))


# --- save / load round-trips exactly --------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    s = Session()
    s.discover("iron", "copper")
    s.discover("iron", "gold")
    s.build("motor", "iron", "copper", "iron")
    before_combos = sorted(m.id for m in s.state.combos())
    before_completed = set(s.state.completed)
    before_best = dict(s.state._best_machines)
    before_handles = {m.id: s.state.handle_of(m.id) for m in s.state.combos()}

    path = tmp_path / "save.json"
    s.save(str(path))
    loaded = GameState.load(str(path))

    assert sorted(m.id for m in loaded.combos()) == before_combos
    assert loaded.completed == before_completed
    assert loaded._best_machines == before_best
    # coined handles are recomputed deterministically -> identical after load
    assert {m.id: loaded.handle_of(m.id) for m in loaded.combos()} == before_handles


def test_loaded_session_continues_consistently(tmp_path):
    s = Session()
    s.discover("iron", "copper")
    path = tmp_path / "s.json"
    s.save(str(path))
    s2 = Session()
    s2.load(str(path))
    # the same re-discovery on the loaded session is recognized as already known
    assert any("already known" in ln for ln in s2.discover("iron", "copper"))
