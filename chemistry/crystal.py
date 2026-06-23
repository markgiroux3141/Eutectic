"""Compound → crystal lattice: the bridge to the materials engine (spec §9 — milestone C2).

This is the load-bearing integration. A compound's bulk structure is a crystal; this module
emits an :class:`engine.lattice.Lattice` whose **per-cell fields are set by the bonding**, so
the *existing* materials extractors measure it **unchanged** (spec §9). The dependency arrow is
``chemistry → engine.lattice`` only; nothing here imports the extractors for its own sake.

Per-cell fields, all sourced from chemistry (spec §9):

* **packing** chosen by bond character — ionic → alternating-charge sublattice (rock-salt);
  metallic → close-packed; covalent-network → coordination tiling. (Molecular packing for
  small covalent molecules — dry ice, ice — is noted as future work; the C2 keystone is the
  network/ionic/metallic crystals.)
* **occupied** — a crystal is dense, so every cell is filled (fill fraction 1.0). Conduction is
  then gated by *metallicity*, not by fill — exactly the M6b split.
* **atom_type** — the species arrangement (sublattices for ionic).
* **metallicity** — from bond character: metallic → conductive (charge backbone); ionic /
  covalent → not. *This* is what makes NaCl an insulator and Cu a conductor (M6b).
* **moment** — from **unpaired electrons** (real electron structure, not an authored tendency):
  Fe's 4 unpaired → high moment → magnetic; Cu's 1 → low → not (spec §9, M3/M4).
* **cohesion** — from bond strength (drives melting M5, strength M8, phonon κ M6b).
* **mass** — from atomic masses (drives density M2, phonon κ).
* **site_potential** — from species **electronegativity** (ionic only): a ``±Δ`` stagger on the
  cation/anion sublattices, ``Δ ∝ ΔEN``. This is what makes NaCl an *insulator* with a real band
  gap (= 2Δ) and a uniform metal a *conductor* with none — the spectral M7a property. Uniform
  (element/metallic/covalent) crystals leave it 0 (no ionic gap; covalent gaps await M7b).

**Affinity-derivation finding (the make-or-break, reported not buried — spec §2, §20).** We
de-risked deriving the *existing elements'* authored affinities from these descriptors. Result:
conduction correlates moderately with low ionization energy (+0.66) but noble metals (Cu/Ag/Au)
underpredict; **magnetism does not derive** — ferromagnetism is the Stoner criterion, not the
unpaired-electron count, so moment-from-unpaired would wrongly make Cr/W/U magnetic and drop
Co/Ni; and cohesion (a many-body effect) anti-correlates with simple descriptors. Silently
swapping the affinities would break M3/M4/M5/M8. So we **keep the authored element affinities**
(legitimate reference data, like ``atomic_mass``) and apply the *chemistry-derived* fields only
to **new compounds** built here. ``engine/elements.py`` is untouched → M0–M8 stay byte-identical.

**Magnetism caveat (carried).** ``moment`` from unpaired electrons is the right local-moment
*magnitude*, but this Ising substrate has only ferromagnetic coupling (J>0), so any connected
high-moment crystal orders ferromagnetically. That is correct for Fe/Co/Ni but would overpredict
for an antiferromagnet (Cr, FeO) — the same limitation the element layer sidesteps by authoring.
"""

from __future__ import annotations

import numpy as np

from engine.lattice import COH_HI, COH_LO, DEFAULT_SHAPE_2D, MOMENT_HI, MOMENT_LO, Lattice

from . import atoms, bonding
from .atoms import Atom
from .bonding import BondCharacter
from .molecule import Molecule

# Unpaired-electron count mapped to the moment range. Divisor sized so Fe's 4 unpaired clears
# the ordering threshold (critical moment ≈ 0.94) and Cu's 1 stays well below it.
_UNPAIRED_REFERENCE: float = 5.0
# Self-bond energy (within a character) mapped to cohesion. The scale is coarse and heuristic
# at C2 (cross-character bond energies are not yet calibrated — see chemistry.bonding); it only
# needs a sensible monotone ordering, e.g. diamond (stiff C network) above the soft metals.
_COHESION_ENERGY_SCALE: float = 200.0
# Electronegativity -> on-site potential scale (M7, the band gap). A site's potential is
# ``scale·(χ_site − χ̄)``; on a 1:1 ionic checkerboard this is a ``±scale·ΔEN/2`` stagger, opening a
# tight-binding gap ``= 2Δ = scale·ΔEN``. ONE fixed constant: it sets only the absolute (uncalibrated)
# eV scale, never the ordering or the conductor/insulator split — the same "one calibrated constant"
# stance as the C1 bond energy and C3 entropy units (see engine.properties.spectral). scale=1 →
# gap ≈ ΔEN (Pauling units), e.g. NaCl ≈ 2.23.
_EN_POTENTIAL_SCALE: float = 1.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _moment_from_unpaired(z: int, charge: int = 0) -> float:
    """Per-cell moment from an atom/ion's *localized* (d/f) unpaired electrons (spec §9).

    Uses d/f unpaired electrons, not the total: in a solid the s/p electrons are quenched by
    bonding/delocalization, so a covalent or main-group ionic crystal (diamond, NaCl) is
    non-magnetic, while a d-electron metal (Fe) keeps its moment. See
    :func:`chemistry.atoms.localized_unpaired_electrons`.
    """
    up = atoms.localized_unpaired_electrons(z, charge)
    return MOMENT_LO + (MOMENT_HI - MOMENT_LO) * _clamp01(up / _UNPAIRED_REFERENCE)


def _cohesion_from_energy(bond_energy: float) -> float:
    """Per-cell cohesion (bond stiffness) from a representative bond energy (coarse, spec §9)."""
    return COH_LO + (COH_HI - COH_LO) * _clamp01(bond_energy / _COHESION_ENERGY_SCALE)


def _metallicity_for(character: BondCharacter) -> float:
    """Charge-carrier quality from bond character: metallic conducts, ionic/covalent does not."""
    return 1.0 if character is BondCharacter.METALLIC else 0.0


def _dense(shape, atom_type, metallicity, moment, mass, cohesion, site_potential=None) -> Lattice:
    """Assemble a fully-occupied (dense crystal) lattice from prebuilt per-cell arrays.

    ``site_potential`` (M7) defaults to ``None`` → zeros (no stagger → no band gap → conductor),
    which is the correct behaviour for a uniform element/metallic/covalent crystal. Only the ionic
    crystal supplies a real staggered potential (see :func:`_ionic_crystal`).
    """
    occ = np.ones(shape, dtype=np.uint8)
    spin = np.ones(shape, dtype=np.int8)   # relax() settles these; pinned +1 is a fine start
    pot = None if site_potential is None else np.asarray(site_potential, dtype=np.float32)
    return Lattice(
        occupied=occ,
        atom_type=np.asarray(atom_type, dtype=np.int8),
        spin=spin,
        mass=np.asarray(mass, dtype=np.float32),
        moment=np.asarray(moment, dtype=np.float32),
        cohesion=np.asarray(cohesion, dtype=np.float32),
        metallicity=np.asarray(metallicity, dtype=np.float32),
        site_potential=pot,
    )


def element_crystal(symbol: str, *, shape=DEFAULT_SHAPE_2D) -> Lattice:
    """Build an elemental crystal; packing follows the element's self-bond character (spec §9).

    Cu (metallic) → close-packed conductor; C (covalent) → stiff insulating network; Fe
    (metallic) → conductor that also magnetises from its unpaired electrons. A noble gas (no
    self-bond) raises.
    """
    a: Atom = atoms.get(symbol)
    character = bonding.bond_character(a, a)
    if character is None:
        raise ValueError(f"{symbol} is a noble gas — forms no crystal")

    shape = tuple(int(s) for s in shape)
    metallicity = np.full(shape, _metallicity_for(character), dtype=np.float32)
    atom_type = np.ones(shape, dtype=np.int8)
    mass = np.full(shape, a.atomic_mass, dtype=np.float32)
    moment = np.full(shape, _moment_from_unpaired(a.z), dtype=np.float32)

    # representative self-bond energy for cohesion (within-character magnitude)
    if character is BondCharacter.METALLIC:
        self_energy = bonding.metallic_bond_energy(a, a)
    else:
        self_energy = bonding.covalent_bond_energy(a, a, order=a.bonding_capacity)
    cohesion = np.full(shape, _cohesion_from_energy(self_energy), dtype=np.float32)

    return _dense(shape, atom_type, metallicity, moment, mass, cohesion)


def compound_crystal(molecule: Molecule, *, shape=DEFAULT_SHAPE_2D) -> Lattice:
    """Build a compound's crystal from a formed :class:`~chemistry.molecule.Molecule` (spec §9).

    Ionic compounds tile a rock-salt-style alternating-charge sublattice (insulating, hard);
    metallic compounds close-pack (conducting). Covalent/polar small molecules would crystallise
    *molecularly* (weak inter-unit bonds) — flagged as future work; for now they are treated as a
    covalent network (a reasonable bulk stand-in for network formers like SiO₂/SiC).
    """
    shape = tuple(int(s) for s in shape)
    char = molecule.character

    if char is BondCharacter.IONIC:
        return _ionic_crystal(molecule, shape)

    # metallic / covalent / polar: a uniform crystal of the compound's atoms.
    metallicity = np.full(shape, _metallicity_for(char), dtype=np.float32)
    atom_type = np.ones(shape, dtype=np.int8)
    # Mean atomic mass and mean moment over the formula unit's atoms (dense uniform crystal).
    syms = molecule.atoms
    charges = molecule.formal_charges
    masses = [atoms.get(s).atomic_mass for s in syms]
    moments = [_moment_from_unpaired(atoms.get(s).z, q) for s, q in zip(syms, charges)]
    mass = np.full(shape, float(np.mean(masses)), dtype=np.float32)
    moment = np.full(shape, float(np.mean(moments)), dtype=np.float32)
    mean_bond = float(np.mean([b.energy for b in molecule.bonds])) if molecule.bonds else 0.0
    cohesion = np.full(shape, _cohesion_from_energy(mean_bond), dtype=np.float32)
    return _dense(shape, atom_type, metallicity, moment, mass, cohesion)


def _ionic_crystal(molecule: Molecule, shape) -> Lattice:
    """Rock-salt-style ionic crystal: alternating cation/anion sublattices (insulating)."""
    # Identify the cation (positive formal charge) and anion species from the formula unit.
    cation_sym = next(s for s, q in zip(molecule.atoms, molecule.formal_charges) if q > 0)
    anion_sym = next(s for s, q in zip(molecule.atoms, molecule.formal_charges) if q < 0)
    cation, anion = atoms.get(cation_sym), atoms.get(anion_sym)
    q_cat = next(q for q in molecule.formal_charges if q > 0)
    q_an = next(q for q in molecule.formal_charges if q < 0)
    n_cat = molecule.counts[cation_sym]
    n_an = molecule.counts[anion_sym]

    ii, jj = np.indices(shape)
    if n_cat == n_an:
        is_cation = (ii + jj) % 2 == 0           # exact rock-salt checkerboard for 1:1
    else:
        # General stoichiometry on the 2D prototype: a deterministic stripe pattern that hits
        # the cation fraction n_cat/(n_cat+n_an). Not the true 3D structure (fluorite/rutile),
        # but charge-neutral and insulating — what the keystone needs (flagged approximate).
        period = n_cat + n_an
        is_cation = ((ii + jj) % period) < n_cat

    atom_type = np.where(is_cation, 1, 2).astype(np.int8)
    metallicity = np.zeros(shape, dtype=np.float32)              # ionic: no free carriers
    mass = np.where(is_cation, cation.atomic_mass, anion.atomic_mass).astype(np.float32)
    # Ion moments: closed-shell main-group ions → ~0; transition-metal ions keep d-moments.
    m_cat = _moment_from_unpaired(cation.z, q_cat)
    m_an = _moment_from_unpaired(anion.z, q_an)
    moment = np.where(is_cation, m_cat, m_an).astype(np.float32)
    cohesion = np.full(shape, COH_HI, dtype=np.float32)         # ionic crystals are hard
    # On-site potential (M7, the band gap): a per-sublattice stagger from electronegativity. The
    # electron localizes on the more electronegative anion → a real ionic gap. χ̄ is the cell mean
    # (count-weighted via the sublattice pattern), so a uniform lattice would give 0 — the gap is
    # set purely by ΔEN. χ for an ion is its parent atom's Pauling EN (ionic formers are not noble).
    chi = np.where(is_cation, cation.electronegativity, anion.electronegativity).astype(np.float64)
    site_potential = (_EN_POTENTIAL_SCALE * (chi - chi.mean())).astype(np.float32)
    return _dense(shape, atom_type, metallicity, moment, mass, cohesion, site_potential)
