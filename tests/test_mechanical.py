"""The M8 keystone: strength + ductility emerge from a central-force spring network (spec §5.7).

Same discipline as the Curie (M4) and melting (M5) keystones, but for *mechanics*. Strength is
the shear modulus of the bond network (the elastic analogue of M3's Laplacian conductance);
ductility is the coordination deficit (the density of slip-enabling under-coordinated sites). The
validations:

1. RIGIDITY TRANSITION — a fully-occupied lattice is over-constrained → rigid (finite shear
   modulus, exactly zero floppy modes beyond rigid-body); diluting it drives the modulus to zero
   and opens floppy modes. The shear-rigidity onset is a real, parameter-free transition.
2. ANTI-CORRELATION (the §5.7 headline) — across the dilution sweep AND across real elements,
   strength rises as ductility falls. It is NOT hardcoded: strength is an elastic-energy solve,
   ductility an independent geometric count; the tradeoff emerges through coordination.
3. REAL ORDERING — refractory/dense (tungsten, carbon) come out strong + brittle; soft/porous
   (lead, mercury) weak + ductile — the mechanical analogue of iron>cobalt>nickel for Curie.
4. CHEAP↔EXACT — the stored (cheap) coordination-deficit ductility tracks the exact (expensive)
   floppy-mode fraction, justifying the "coarse stored value, accurate instrument" split.
5. gated stored property + determinism (spec §6).
"""

import numpy as np

from engine import elements
from engine.lattice import Lattice
from engine.material import MECH_GATE_FLOOR, from_element
from engine.properties import mechanical


def _uniform_lattice(shape, cohesion=1.0, fill=1.0, seed=0):
    """A uniform-cohesion lattice at a given fill (random vacancies) — the clean spring network."""
    rng = np.random.default_rng(seed)
    occ = (rng.random(shape) < fill).astype(np.uint8)
    return Lattice(
        occupied=occ,
        atom_type=occ.astype(np.int8),
        spin=np.ones(shape, np.int8),
        cohesion=np.full(shape, cohesion, np.float32),
    )


# --- KEYSTONE 1: the rigidity transition (parameter-free) ------------------------------


def test_full_lattice_is_rigid_no_floppy_modes():
    """A fully-occupied NN+diagonal lattice is over-constrained: rigid, zero floppy modes."""
    lat = _uniform_lattice((24, 24), cohesion=1.0, fill=1.0)
    assert mechanical.shear_modulus(lat) > 0.0
    # Beyond the 3 rigid-body modes there are NO zero-energy deformations.
    assert mechanical.floppy_fraction(lat) == 0.0


def test_shear_modulus_collapses_and_floppy_opens_with_dilution():
    """Diluting the network drives shear modulus -> 0 and opens floppy modes (the transition)."""
    dense = _uniform_lattice((28, 28), cohesion=1.0, fill=1.0, seed=1)
    sparse_ = _uniform_lattice((28, 28), cohesion=1.0, fill=0.50, seed=1)
    assert mechanical.shear_modulus(dense) > mechanical.shear_modulus(sparse_)
    assert mechanical.floppy_fraction(sparse_) > mechanical.floppy_fraction(dense)
    # Near/below the connectivity limit the shear modulus is essentially gone (a mechanism).
    assert mechanical.shear_modulus(sparse_) < 0.05 * mechanical.shear_modulus(dense) + 1e-9


def test_strength_monotonic_in_fill():
    """Strength rises monotonically with coordination (fill) — the rigidity order parameter."""
    fills = [0.55, 0.70, 0.85, 1.00]
    strengths = [mechanical.shear_modulus(_uniform_lattice((26, 26), 1.0, f, seed=2)) for f in fills]
    assert strengths == sorted(strengths), f"strength not monotonic in fill: {strengths}"


# --- KEYSTONE 2: the strength<->ductility anti-correlation EMERGES ----------------------


def test_anticorrelation_emerges_over_dilution():
    """As fill rises, strength up and ductility down — measured, not wired in."""
    fills = [0.55, 0.70, 0.85, 1.00]
    strength, ductility = [], []
    for f in fills:
        lat = _uniform_lattice((26, 26), 1.0, f, seed=3)
        strength.append(mechanical.shear_modulus(lat))
        ductility.append(mechanical.ductility(lat))
    assert strength == sorted(strength)              # increasing
    assert ductility == sorted(ductility, reverse=True)  # decreasing
    assert np.corrcoef(strength, ductility)[0, 1] < -0.9


def test_anticorrelation_across_real_elements():
    """Across real elements the stored strength and ductility are strongly anti-correlated."""
    ids = ["tungsten", "carbon", "titanium", "iron", "copper", "gold",
           "aluminium", "lead", "mercury"]
    strength, ductility = [], []
    for i in ids:
        lat = from_element(elements.get(i), shape=(32, 32)).lattice
        strength.append(mechanical.shear_modulus(lat))
        ductility.append(mechanical.ductility(lat))
    assert np.corrcoef(strength, ductility)[0, 1] < -0.8


# --- KEYSTONE 3: real ordering recovered ------------------------------------------------


def test_real_strength_and_ductility_ordering():
    """Refractory/dense -> strong+brittle; soft/porous -> weak+ductile (the real ordering)."""
    def mech(eid):
        lat = from_element(elements.get(eid), shape=(32, 32)).lattice
        return mechanical.shear_modulus(lat), mechanical.ductility(lat)

    w_s, w_d = mech("tungsten")
    fe_s, fe_d = mech("iron")
    hg_s, hg_d = mech("mercury")
    assert w_s > fe_s > hg_s, f"strength order wrong: W={w_s:.3f} Fe={fe_s:.3f} Hg={hg_s:.3f}"
    assert w_d < fe_d < hg_d, f"ductility order wrong: W={w_d:.3f} Fe={fe_d:.3f} Hg={hg_d:.3f}"


# --- KEYSTONE 4: the cheap stored ductility tracks the exact floppy-mode mechanics ------


def test_cheap_ductility_tracks_exact_floppy_fraction():
    """The stored coordination-deficit ductility tracks the exact floppy-mode fraction.

    Validated over the real-element set (where coordination spans a useful range): the cheap O(N)
    proxy preserves the exact (expensive ``eigvalsh``) mechanics — Pearson ≈ 0.93, Spearman ≈ 0.98
    — which is what licenses storing the cheap measure and keeping the exact one on demand.
    """
    ids = ["tungsten", "carbon", "titanium", "silicon", "iron", "copper", "gold",
           "aluminium", "zinc", "lead", "mercury", "hydrogen"]
    cheap, exact = [], []
    for i in ids:
        lat = from_element(elements.get(i), shape=(28, 28)).lattice
        cheap.append(mechanical.ductility(lat))
        exact.append(mechanical.floppy_fraction(lat))
    assert np.corrcoef(cheap, exact)[0, 1] > 0.85


# --- the gated stored property (material pipeline) -------------------------------------


def test_stored_mechanical_gating():
    """A connected solid stores strength>0; a dispersed structure stores 0 + high ductility."""
    iron = from_element(elements.get("iron"), shape=(40, 40))
    hydrogen = from_element(elements.get("hydrogen"), shape=(40, 40))
    assert iron.properties["strength"] > 0.0
    assert iron.properties["largest_cluster_fraction"] >= MECH_GATE_FLOOR
    assert "ductility" in iron.properties and "bulk_modulus" in iron.properties
    # Dispersed: no load-bearing solid -> strength gated to 0, but it is honestly very ductile.
    assert hydrogen.properties["strength"] == 0.0
    assert hydrogen.properties["largest_cluster_fraction"] < MECH_GATE_FLOOR
    assert hydrogen.properties["ductility"] > iron.properties["ductility"]


# --- determinism (spec §6) ------------------------------------------------------------


def test_mechanical_is_deterministic():
    """Same structure -> byte-identical mechanical observables (no RNG, fixed solves)."""
    lat = from_element(elements.get("iron"), shape=(28, 28)).lattice
    assert mechanical.measure(lat) == mechanical.measure(lat)
    assert mechanical.floppy_fraction(lat) == mechanical.floppy_fraction(lat)
