"""C0 keystone: valence + periodic trends, parameter-free (chemistry-engine-spec §5, §15).

The keystone discipline (no-fudge norm): every claim is pressure-tested against the textbook
value with *zero* per-element tuning. Valence/capacity/lone-pairs are derived from a
Madelung-filled configuration; EN/radius *trends* from a Slater Z_eff model. If any of these
needed a hand-authored exception, the emergence would be a fiction — so the tests below assert
the whole keystone set against one formula each.
"""

import pytest

from chemistry import atoms


# --- table integrity ------------------------------------------------------------------

def test_table_covers_the_keystone_minimum():
    # Spec §5 minimum slice: H, C, N, O, Na, Cl, Mg, Fe, Cu, Si, + metals.
    for sym in ("H", "C", "N", "O", "Na", "Cl", "Mg", "Fe", "Cu", "Si"):
        assert sym in atoms.ATOMS, sym
    assert 20 <= len(atoms.ATOMS) <= 60


def test_symbols_ordered_by_z_and_unique():
    syms = atoms.all_symbols()
    zs = [atoms.get(s).z for s in syms]
    assert zs == sorted(zs)
    assert len(syms) == len(set(syms))


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        atoms.get("Xx")


def test_authored_values_are_sane():
    for a in atoms.ATOMS.values():
        assert a.z >= 1
        assert a.atomic_mass > 0
        assert a.covalent_radius > 0
        assert a.ionization_energy > 0
        if a.electronegativity is not None:
            assert 0.5 <= a.electronegativity <= 4.0, a.symbol


# --- KEYSTONE: valence / bonding capacity, parameter-free -----------------------------

# (symbol, expected common valence / bonding capacity) — spec §8 / §15.
_VALENCE_KEYSTONE = {
    "H": 1, "C": 4, "N": 3, "O": 2, "F": 1,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 3, "S": 2, "Cl": 1,
}


@pytest.mark.parametrize("symbol,expected", sorted(_VALENCE_KEYSTONE.items()))
def test_bonding_capacity_matches_textbook(symbol, expected):
    assert atoms.get(symbol).bonding_capacity == expected


def test_carbon_capacity_is_the_promotion_case():
    # Carbon's ground state has only 2 unpaired electrons (2s²2p²) but its valence is 4 —
    # the octet-deficit rule captures promotion without a per-element special case.
    assert atoms.get("C").bonding_capacity == 4


# --- KEYSTONE: ion charge sign + magnitude for the clear ionic cases ------------------

@pytest.mark.parametrize("symbol,expected", [
    ("Na", +1), ("Mg", +2), ("Al", +3), ("K", +1), ("Ca", +2),
    ("Cl", -1), ("O", -2), ("F", -1),
])
def test_ion_charge_falls_out(symbol, expected):
    assert atoms.get(symbol).ion_charge == expected


def test_carbon_prefers_covalent_neither_ion():
    # Equidistant from both closed shells -> no preferred ionic direction.
    assert atoms.get("C").ion_charge == 0


# --- KEYSTONE: noble gases inert ------------------------------------------------------

@pytest.mark.parametrize("symbol", ["He", "Ne", "Ar"])
def test_noble_gases_are_inert(symbol):
    a = atoms.get(symbol)
    assert a.bonding_capacity == 0
    assert a.is_noble_gas


# --- lone pairs (the VSEPR input for C1a) ---------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    ("C", 0), ("N", 1), ("O", 2), ("F", 3), ("H", 0), ("Cl", 3),
])
def test_lone_pairs(symbol, expected):
    assert atoms.get(symbol).lone_pairs == expected


# --- configuration derivation ---------------------------------------------------------

def test_configuration_examples():
    assert atoms.configuration_string(8) == "1s2 2s2 2p4"        # oxygen
    assert atoms.configuration_string(11) == "1s2 2s2 2p6 3s1"   # sodium


def test_configuration_electron_count_conserved():
    # Every electron is placed: the configuration sums back to Z.
    for a in atoms.ATOMS.values():
        assert sum(c for (_n, _l, c) in a.configuration) == a.z


def test_configuration_is_deterministic():
    # Pure function of Z -> identical, hashable value every call (spec §14).
    assert atoms.configuration(26) == atoms.configuration(26)
    assert isinstance(atoms.configuration(26), tuple)


# --- orbital availability gates hypervalency later ------------------------------------

def test_available_orbitals_gate_d_by_period():
    assert atoms.get("C").available_orbitals == ("s", "p")        # period 2: no d
    assert "d" in atoms.get("S").available_orbitals               # period 3: d accessible
    assert "d" in atoms.get("P").available_orbitals


# --- KEYSTONE (bonus): periodic trends emerge from derived Z_eff ----------------------

def _ens(symbols):
    return [atoms.get(s).en_proxy for s in symbols]


def _radii(symbols):
    return [atoms.get(s).radius_proxy for s in symbols]


def _strictly_rising(xs):
    return all(a < b for a, b in zip(xs, xs[1:]))


def _strictly_falling(xs):
    return all(a > b for a, b in zip(xs, xs[1:]))


def test_en_rises_and_radius_falls_across_period_2():
    period2 = ["Li", "Be", "B", "C", "N", "O", "F"]
    assert _strictly_rising(_ens(period2))
    assert _strictly_falling(_radii(period2))


@pytest.mark.parametrize("group", [
    ["Li", "Na", "K"],            # group 1
    ["F", "Cl", "Br", "I"],       # group 17
])
def test_en_falls_and_radius_rises_down_a_group(group):
    assert _strictly_falling(_ens(group))
    assert _strictly_rising(_radii(group))


def test_derived_en_is_ordinal_not_pauling():
    # Honest caveat made executable: the proxy is trend-correct but NOT on the Pauling scale,
    # so quantitative work must read the authored value. (Guards against anyone later treating
    # en_proxy as if it were calibrated.)
    assert atoms.get("H").en_proxy != pytest.approx(atoms.get("H").electronegativity, abs=0.5)
