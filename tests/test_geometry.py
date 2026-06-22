"""C1a keystone: VSEPR geometry from steric number + lone pairs (spec §6, §15).

CH₄ / H₂O / CO₂ / NH₃ angles and shapes from electron counting alone — one calibrated
constant (lone-pair compression) shared across all of them, no per-molecule rule.
"""

import pytest

from chemistry import orbitals


# --- KEYSTONE: hybridization + angle + shape ------------------------------------------
# (label, central, sigma_bonds, lone_pairs, hyb, angle, shape)
_GEOMETRY_KEYSTONE = [
    ("CH4", "C", 4, 0, "sp3", 109.5, "tetrahedral"),
    ("H2O", "O", 2, 2, "sp3", 104.5, "bent"),
    ("CO2", "C", 2, 0, "sp",  180.0, "linear"),
    ("NH3", "N", 3, 1, "sp3", 107.0, "trigonal pyramidal"),
]


@pytest.mark.parametrize("label,central,sigma,lp,hyb,angle,shape", _GEOMETRY_KEYSTONE)
def test_geometry_keystone(label, central, sigma, lp, hyb, angle, shape):
    g = orbitals.Geometry.from_counts(central, sigma_bonds=sigma, lone_pairs=lp)
    assert g.hybridization == hyb, label
    assert g.bond_angle == pytest.approx(angle, abs=0.05), label
    assert g.shape == shape, label


def test_water_more_compressed_than_ammonia():
    # Two lone pairs squeeze harder than one — the angle ordering is itself a keystone.
    h2o = orbitals.Geometry.from_counts("O", 2, 2).bond_angle
    nh3 = orbitals.Geometry.from_counts("N", 3, 1).bond_angle
    ch4 = orbitals.Geometry.from_counts("C", 4, 0).bond_angle
    assert h2o < nh3 < ch4


def test_steric_number_and_hybridization_table():
    assert orbitals.hybridization(2) == "sp"
    assert orbitals.hybridization(3) == "sp2"
    assert orbitals.hybridization(4) == "sp3"
    assert orbitals.hybridization(5) == "sp3d"
    assert orbitals.hybridization(6) == "sp3d2"


def test_steric_number_out_of_range_raises():
    with pytest.raises(ValueError):
        orbitals.hybridization(7)
