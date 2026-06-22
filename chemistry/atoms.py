"""The atom model (chemistry-engine-spec §4, §5 — milestone C0).

The leaves of the chemistry hierarchy. A root :class:`Atom` carries a handful of **distilled
quantum descriptors** (electron configuration, electronegativity, radius, ionization energy,
electron affinity — spec §3) and from them an enormous amount of chemistry *emerges* by
electron counting, never by assignment (the one principle, spec §1).

What is **authored** vs **derived** (the no-fudge line):

* **Authored reference data** (distilled real values, exactly as ``atomic_mass`` already is —
  not a fudge): atomic number ``Z``, ``atomic_mass``, Pauling ``electronegativity``,
  ``covalent_radius``, ``ionization_energy``, ``electron_affinity``. These are measured
  physical constants; we look them up, we do not invent them.

* **Derived parameter-free** (the emergence — see the de-risk readout below): the electron
  ``configuration`` (from ``Z`` by the aufbau/Madelung rule), and from it
  ``valence_electrons``, ``bonding_capacity``, ``lone_pairs``, ``ion_charge``,
  ``available_orbitals``, the Slater ``z_eff``, and the ordinal EN/radius *proxies*.

**C0 de-risk readout (numbers reported before this architecture was committed).** The
octet-deficit rule ``capacity = min(valence_e, octet − valence_e)`` over a Madelung-filled
configuration reproduces the keystone valences **parameter-free**: H→1, C→4 (the promotion
case falls out, no special-casing), N→3, O→2, Na→+1, Mg→+2, Cl→−1, noble gases→0 (inert);
lone pairs come out right (O→2, N→1 — what VSEPR needs in C1a). A Slater effective-nuclear-
charge model makes ``EN ∝ Z_eff/n²`` and ``radius ∝ n²/Z_eff`` reproduce every periodic
**trend** (EN rises across a period / falls down a group; radius the reverse). That trend is
a *bonus keystone* — but the derived EN is **ordinal, not Pauling-calibrated**, so the
quantitative ΔEN bond-character thresholds in C1b read the *authored* Pauling values; the
proxy exists only to carry the trend keystone honestly.

**Known limitations (flagged, not buried):** the outermost-sp valence definition does not
see d-shell multivalence, so transition metals (Fe, Cu) read a single common valence of 2;
and the plain Madelung fill misses the half/full-shell anomalies (it gives Cu 3d⁹4s², not
the real 3d¹⁰4s¹). Neither touches the main-group C0 keystone; both are recorded for the
transition-metal work later.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# --- aufbau / Madelung filling (parameter-free) ---------------------------------------
# Subshell electron capacities 2(2l+1): s=2, p=6, d=10, f=14.
_CAP: dict[int, int] = {0: 2, 1: 6, 2: 10, 3: 14}
_LNAME: dict[int, str] = {0: "s", 1: "p", 2: "d", 3: "f"}


@lru_cache(maxsize=1)
def _madelung_order() -> tuple[tuple[int, int], ...]:
    """``(n, l)`` subshells in Madelung (``n+l``, then ``n``) order — the aufbau rule.

    Generated, not tabulated: a subshell is admissible when ``0 <= l < n``; the closed-form
    fill order is just these sorted by ``(n+l, n)``. Covers well past ``Z=118``.
    """
    shells = [
        (n, l)
        for s in range(16)          # n + l, the Madelung energy index
        for n in range(1, s + 2)
        for l in (s - n,)
        if 0 <= l < n
    ]
    shells.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return tuple(shells)


@lru_cache(maxsize=128)
def configuration(z: int) -> tuple[tuple[int, int, int], ...]:
    """Distil ``Z`` to its electron configuration ``((n, l, count), ...)`` by aufbau.

    Parameter-free: fill ``Z`` electrons into subshells in Madelung order until exhausted.
    Returned sorted by ``(n, l)`` so the configuration is a canonical, hashable value.
    """
    if z < 1:
        raise ValueError(f"atomic number must be >= 1, got {z}")
    conf: list[tuple[int, int, int]] = []
    left = z
    for (n, l) in _madelung_order():
        if left <= 0:
            break
        c = min(_CAP[l], left)
        conf.append((n, l, c))
        left -= c
    conf.sort(key=lambda nlc: (nlc[0], nlc[1]))
    return tuple(conf)


def configuration_string(z: int) -> str:
    """Human-readable configuration, e.g. ``1s2 2s2 2p4`` for oxygen."""
    return " ".join(f"{n}{_LNAME[l]}{c}" for (n, l, c) in configuration(z))


def _max_shell(conf: tuple[tuple[int, int, int], ...]) -> int:
    """Highest occupied principal quantum number ``n`` (the valence shell)."""
    return max(n for (n, _l, _c) in conf)


def valence_electrons(z: int) -> int:
    """Main-group valence count: electrons in the outermost shell's s and p subshells.

    NB (known limitation): this deliberately ignores ``(n-1)d`` occupancy, so transition
    metals report only their ``ns`` count (a single common valence) — fine for the
    main-group C0 keystone, handled separately when transition-metal multivalence lands.
    """
    conf = configuration(z)
    mn = _max_shell(conf)
    return sum(c for (n, l, c) in conf if n == mn and l <= 1)


def octet_size(z: int) -> int:
    """Closed-shell target for the valence shell: a **duet** (2) for ``n=1``, else an octet."""
    return 2 if _max_shell(configuration(z)) == 1 else 8


def bonding_capacity(z: int) -> int:
    """Common valence: bonds (or transferred electrons) to reach the nearest closed shell.

    ``min(valence_e, octet − valence_e)`` — lose the few you have (cation) or gain the few
    you lack (anion), whichever is fewer. This implicitly accounts for promotion (carbon:
    ``min(4, 4) = 4``) without a per-element rule, and makes noble gases inert
    (``min(8, 0) = 0``). Expanded-octet / hypervalent capacity (PCl₅, SF₆) is **not** added
    here — it is gated by :func:`available_orbitals` and emerges in C1, where a keystone
    needs it; no C0 keystone atom is hypervalent.
    """
    ve = valence_electrons(z)
    return min(ve, octet_size(z) - ve)


def lone_pairs(z: int) -> int:
    """Non-bonding electron pairs on the neutral atom: ``(valence_e − capacity) / 2``.

    The VSEPR input for C1a: O→2, N→1, C→0, F/Cl→3 — exactly the counts that bend water
    and pyramidalise ammonia.
    """
    return (valence_electrons(z) - bonding_capacity(z)) // 2


def ion_charge(z: int) -> int:
    """Preferred monatomic ion charge: ``+ve`` if it loses electrons, ``−deficit`` if it gains.

    The sign falls out of which side of the shell is closer: Na(+1), Mg(+2), Cl(−1), O(−2),
    and 0 for the balanced (covalent-preferring) middle, e.g. carbon. The *magnitude* equals
    :func:`bonding_capacity`; this adds only the direction. (Whether a real bond is ionic is
    settled later by ΔEN with a partner — spec §7 / C1b — not by this single-atom preference.)
    """
    ve = valence_electrons(z)
    deficit = octet_size(z) - ve
    if ve < deficit:
        return +ve
    if deficit < ve:
        return -deficit
    return 0


def available_orbitals(z: int) -> tuple[str, ...]:
    """Orbital types accessible to the valence shell for hybridization (spec §5, §6).

    ``s`` always; ``p`` once the valence shell reaches ``n>=2``; ``d`` once ``n>=3`` (the
    period at which d-orbitals become energetically reachable for expansion). This is the
    gate the orbital model buys us: it is *why* expanded octets (sp³d, sp³d²) can emerge in
    C1 instead of being special-cased.
    """
    mn = _max_shell(configuration(z))
    orbs = ["s"]
    if mn >= 2:
        orbs.append("p")
    if mn >= 3:
        orbs.append("d")
    return tuple(orbs)


# --- Slater effective nuclear charge -> derived EN / radius proxies (trend keystone) --
# Slater shielding coefficients are distilled QM (a textbook approximation of electron
# screening), not tunable dials — the same stance as using real atomic masses.

def _slater_groups(conf: tuple[tuple[int, int, int], ...]) -> dict[tuple[int, str], int]:
    """Slater shielding groups: ``[1s] [2s2p] [3s3p] [3d] [4s4p] [4d] [4f] ...``."""
    groups: dict[tuple[int, str], int] = {}
    for (n, l, c) in conf:
        key = (n, "sp") if l <= 1 else (n, _LNAME[l])
        groups[key] = groups.get(key, 0) + c
    return groups


def z_eff(z: int) -> float:
    """Slater effective nuclear charge felt by an outermost (valence sp) electron.

    The screened pull that sets both electronegativity and size. Other electrons in the same
    group shield 0.35 (0.30 in the 1s group), the ``n-1`` shell shields 0.85, and everything
    deeper shields 1.00.
    """
    conf = configuration(z)
    mn = _max_shell(conf)
    groups = _slater_groups(conf)
    key = (mn, "sp")
    same = groups.get(key, 0)
    same_coeff = 0.30 if mn == 1 else 0.35
    shielding = same_coeff * (same - 1)
    for (gn, _gl), c in groups.items():
        if (gn, _gl) == key:
            continue
        if gn == mn - 1:
            shielding += 0.85 * c
        elif gn < mn - 1:
            shielding += 1.00 * c
    return z - shielding


def electronegativity_proxy(z: int) -> float:
    """Ordinal EN ``∝ Z_eff / n²`` (orbital-energy scaling). Trend-correct, not calibrated.

    Reproduces every periodic EN trend (the bonus keystone) but is **not** on the Pauling
    scale — quantitative ΔEN work reads the authored Pauling value instead (see module doc).
    """
    mn = _max_shell(configuration(z))
    return z_eff(z) / (mn * mn)


def covalent_radius_proxy(z: int) -> float:
    """Ordinal radius ``∝ n² / Z_eff`` (Bohr-like extent). Trend-correct, not calibrated."""
    mn = _max_shell(configuration(z))
    return (mn * mn) / z_eff(z)


# --- the Atom value object ------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """A root atom: authored descriptors + everything derived parameter-free from ``Z``.

    The authored fields are distilled reference data (spec §3). ``electronegativity`` is the
    Pauling scalar and is ``None`` for noble gases (no meaningful Pauling value — they do not
    bond). All other quantities are computed from ``z`` via the module-level derivations, so
    construction stays trivial and a forged inconsistent atom is impossible.
    """

    symbol: str
    name: str
    z: int
    atomic_mass: float
    electronegativity: float | None     # authored Pauling scalar; None for noble gases
    covalent_radius: float              # authored, Ångström
    ionization_energy: float            # authored, eV (first)
    electron_affinity: float            # authored, eV

    # --- derived (never authored) ---
    @property
    def configuration(self) -> tuple[tuple[int, int, int], ...]:
        return configuration(self.z)

    @property
    def configuration_string(self) -> str:
        return configuration_string(self.z)

    @property
    def valence_electrons(self) -> int:
        return valence_electrons(self.z)

    @property
    def octet_size(self) -> int:
        return octet_size(self.z)

    @property
    def bonding_capacity(self) -> int:
        return bonding_capacity(self.z)

    @property
    def lone_pairs(self) -> int:
        return lone_pairs(self.z)

    @property
    def ion_charge(self) -> int:
        return ion_charge(self.z)

    @property
    def available_orbitals(self) -> tuple[str, ...]:
        return available_orbitals(self.z)

    @property
    def is_noble_gas(self) -> bool:
        """A full valence shell -> zero bonding capacity (inert)."""
        return self.bonding_capacity == 0

    @property
    def z_eff(self) -> float:
        return z_eff(self.z)

    @property
    def en_proxy(self) -> float:
        return electronegativity_proxy(self.z)

    @property
    def radius_proxy(self) -> float:
        return covalent_radius_proxy(self.z)


# --- authored periodic reference table ------------------------------------------------
# Distilled real values (spec §5). Pauling EN is None for noble gases. Columns:
#   symbol, name, Z, atomic_mass, EN(Pauling), covalent_radius(Å), IE1(eV), EA(eV)
# Seeded by the existing engine element set and extended to a usable main-group slice
# (spec §5 minimum: H, C, N, O, Na, Cl, Mg, Fe, Cu, Si, plus the current metals).

_TABLE: tuple[tuple[str, str, int, float, float | None, float, float, float], ...] = (
    ("H",  "Hydrogen",    1,   1.008,   2.20, 0.31, 13.60, 0.75),
    ("He", "Helium",      2,   4.0026, None,  0.28, 24.59, 0.00),
    ("Li", "Lithium",     3,   6.94,    0.98, 1.28,  5.39, 0.62),
    ("Be", "Beryllium",   4,   9.0122,  1.57, 0.96,  9.32, 0.00),
    ("B",  "Boron",       5,  10.81,    2.04, 0.84,  8.30, 0.28),
    ("C",  "Carbon",      6,  12.011,   2.55, 0.76, 11.26, 1.26),
    ("N",  "Nitrogen",    7,  14.007,   3.04, 0.71, 14.53, 0.00),
    ("O",  "Oxygen",      8,  15.999,   3.44, 0.66, 13.62, 1.46),
    ("F",  "Fluorine",    9,  18.998,   3.98, 0.57, 17.42, 3.40),
    ("Ne", "Neon",       10,  20.180,  None,  0.58, 21.56, 0.00),
    ("Na", "Sodium",     11,  22.990,   0.93, 1.66,  5.14, 0.55),
    ("Mg", "Magnesium",  12,  24.305,   1.31, 1.41,  7.65, 0.00),
    ("Al", "Aluminium",  13,  26.982,   1.61, 1.21,  5.99, 0.43),
    ("Si", "Silicon",    14,  28.085,   1.90, 1.11,  8.15, 1.39),
    ("P",  "Phosphorus", 15,  30.974,   2.19, 1.07, 10.49, 0.75),
    ("S",  "Sulfur",     16,  32.06,    2.58, 1.05, 10.36, 2.08),
    ("Cl", "Chlorine",   17,  35.45,    3.16, 1.02, 12.97, 3.61),
    ("Ar", "Argon",      18,  39.948,  None,  1.06, 15.76, 0.00),
    ("K",  "Potassium",  19,  39.098,   0.82, 2.03,  4.34, 0.50),
    ("Ca", "Calcium",    20,  40.078,   1.00, 1.76,  6.11, 0.02),
    ("Ti", "Titanium",   22,  47.867,   1.54, 1.60,  6.83, 0.08),
    ("Cr", "Chromium",   24,  51.996,   1.66, 1.39,  6.77, 0.67),
    ("Fe", "Iron",       26,  55.845,   1.83, 1.32,  7.90, 0.15),
    ("Co", "Cobalt",     27,  58.933,   1.88, 1.26,  7.88, 0.66),
    ("Ni", "Nickel",     28,  58.693,   1.91, 1.24,  7.64, 1.16),
    ("Cu", "Copper",     29,  63.546,   1.90, 1.32,  7.73, 1.24),
    ("Zn", "Zinc",       30,  65.38,    1.65, 1.22,  9.39, 0.00),
    ("Br", "Bromine",    35,  79.904,   2.96, 1.20, 11.81, 3.36),
    ("Nb", "Niobium",    41,  92.906,   1.60, 1.64,  6.76, 0.92),
    ("Ag", "Silver",     47, 107.868,   1.93, 1.45,  7.58, 1.30),
    ("Sn", "Tin",        50, 118.710,   1.96, 1.39,  7.34, 1.11),
    ("I",  "Iodine",     53, 126.904,   2.66, 1.39, 10.45, 3.06),
    ("W",  "Tungsten",   74, 183.84,    2.36, 1.62,  7.98, 0.82),
    ("Pt", "Platinum",   78, 195.084,   2.28, 1.36,  8.96, 2.13),
    ("Au", "Gold",       79, 196.967,   2.54, 1.36,  9.23, 2.31),
    ("Hg", "Mercury",    80, 200.592,   2.00, 1.32, 10.44, 0.00),
    ("Pb", "Lead",       82, 207.2,     2.33, 1.46,  7.42, 0.36),
    ("U",  "Uranium",    92, 238.029,   1.38, 1.96,  6.19, 0.00),
)


ATOMS: dict[str, Atom] = {
    sym: Atom(
        symbol=sym, name=name, z=z, atomic_mass=mass,
        electronegativity=en, covalent_radius=r,
        ionization_energy=ie, electron_affinity=ea,
    )
    for (sym, name, z, mass, en, r, ie, ea) in _TABLE
}


def get(symbol: str) -> Atom:
    """Look up a root atom by element symbol, with a helpful error on a typo."""
    try:
        return ATOMS[symbol]
    except KeyError:
        raise KeyError(
            f"unknown atom {symbol!r}; known: {', '.join(all_symbols())}"
        ) from None


def all_symbols() -> list[str]:
    """All atom symbols, ordered by atomic number (canonical, deterministic — spec §14)."""
    return [a.symbol for a in sorted(ATOMS.values(), key=lambda a: a.z)]
