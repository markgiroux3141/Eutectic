"""Chemistry explorer / verification harness (chemistry-engine-spec §16).

Mirrors :mod:`tools.explorer` (ASCII stdout; matplotlib reserved for later sweeps). The
whole no-fudge bet at this level is "does real chemistry *emerge* from the distilled
descriptors?" — this tool is how we see it before trusting it.

Commands grow with the C0..C5 ladder. C0 ships:

* ``list``           — the root-atom table (authored descriptors at a glance).
* ``inspect-atom``   — one atom's descriptors + everything derived parameter-free from Z
                       (configuration, valence, capacity, lone pairs, ion charge, orbitals,
                       and the Z_eff-derived EN/radius trend proxies).
* ``build-molecule`` — form the minimum-energy compound of two elements: the settled bonding
                       graph, stoichiometry/formula, bond characters/orders/energies, VSEPR
                       geometry, and formation energy (spec §16).
* ``measure-compound`` — the integration view (C2): build a compound's crystal lattice and run
                       the **materials** extractors on it — chemistry in, bulk properties out.

Usage::

    python -m tools.chem_explorer list
    python -m tools.chem_explorer inspect-atom O
    python -m tools.chem_explorer build-molecule H O
    python -m tools.chem_explorer measure-compound Cu        # element crystal
    python -m tools.chem_explorer measure-compound Na Cl     # binary compound
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from chemistry import atoms, crystal, molecule
from chemistry.molecule import Molecule


def _cmd_list(_args: argparse.Namespace) -> int:
    print(f"{'sym':>3} {'name':<11} {'Z':>3} {'mass':>8} {'EN':>5} {'r':>5} "
          f"{'val':>3} {'cap':>3} {'lp':>3} {'ion':>4}")
    print("-" * 60)
    for sym in atoms.all_symbols():
        a = atoms.get(sym)
        en = "  -  " if a.electronegativity is None else f"{a.electronegativity:>5.2f}"
        print(f"{a.symbol:>3} {a.name:<11} {a.z:>3} {a.atomic_mass:>8.3f} {en} "
              f"{a.covalent_radius:>5.2f} {a.valence_electrons:>3} {a.bonding_capacity:>3} "
              f"{a.lone_pairs:>3} {a.ion_charge:>+4d}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        a = atoms.get(args.symbol)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    en = "undefined (noble gas)" if a.electronegativity is None else f"{a.electronegativity:.2f}"
    print(f"{a.symbol} - {a.name}  (Z={a.z})")
    print("=" * 56)
    print("authored descriptors (distilled reference data):")
    print(f"  atomic_mass        {a.atomic_mass:.3f}")
    print(f"  electronegativity  {en}  (Pauling)")
    print(f"  covalent_radius    {a.covalent_radius:.2f} A")
    print(f"  ionization_energy  {a.ionization_energy:.2f} eV")
    print(f"  electron_affinity  {a.electron_affinity:.2f} eV")
    print()
    print("derived parameter-free from Z (the emergence):")
    print(f"  configuration      {a.configuration_string}")
    print(f"  valence_electrons  {a.valence_electrons}")
    print(f"  octet_size         {a.octet_size}  ({'duet' if a.octet_size == 2 else 'octet'})")
    print(f"  bonding_capacity   {a.bonding_capacity}  (common valence)")
    print(f"  lone_pairs         {a.lone_pairs}")
    print(f"  ion_charge         {a.ion_charge:+d}")
    print(f"  available_orbitals {', '.join(a.available_orbitals)}")
    print(f"  is_noble_gas       {a.is_noble_gas}")
    print()
    print("derived trend proxies (Slater Z_eff - ordinal, NOT Pauling-calibrated):")
    print(f"  z_eff              {a.z_eff:.2f}")
    print(f"  en_proxy           {a.en_proxy:.3f}  (~ Z_eff / n^2)")
    print(f"  radius_proxy       {a.radius_proxy:.3f}  (~ n^2 / Z_eff)")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    result = molecule.form_binary(args.a, args.b)
    if not isinstance(result, Molecule):
        print(f"{args.a} + {args.b}: no compound forms - {result.reason}")
        return 0
    m = result
    print(f"{args.a} + {args.b}  ->  {m.formula}   ({m.character.value})")
    print("=" * 56)
    print(f"  stoichiometry      {m.counts}")
    print(f"  formation_energy   {m.formation_energy:.2f}  (negative = stable)")
    print(f"  canonical_id       {m.canonical_id()}")
    print()
    print("  sites (index: symbol, formal charge):")
    for i, (sym, q) in enumerate(zip(m.atoms, m.formal_charges)):
        print(f"    [{i}] {sym}{'' if q == 0 else f' {q:+d}'}")
    print()
    print("  bonds:")
    for bd in m.bonds:
        print(f"    {m.atoms[bd.a_index]}({bd.a_index})-{m.atoms[bd.b_index]}({bd.b_index})  "
              f"order={bd.order}  {bd.character.value}  E~{bd.energy:.1f}")
    if m.geometry is not None:
        g = m.geometry
        print()
        print(f"  geometry (central {g.central}):")
        print(f"    steric_number    {g.steric_number}  ({g.sigma_bonds} sigma + {g.lone_pairs} lp)")
        print(f"    hybridization    {g.hybridization}")
        print(f"    shape            {g.shape}")
        print(f"    bond_angle       {g.bond_angle:.1f} deg")
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    # Local imports: only this command touches the materials engine (chemistry -> engine bridge).
    from engine import material
    from engine.lattice import relax

    if args.b is None:
        try:
            lat = crystal.element_crystal(args.a)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        label = args.a
    else:
        m = molecule.form_binary(args.a, args.b)
        if not isinstance(m, Molecule):
            print(f"{args.a} + {args.b}: no compound forms - {m.reason}")
            return 0
        lat = crystal.compound_crystal(m)
        label = m.formula

    # Settle spins before measuring (a crystal is a settled lattice too, spec §1).
    settled = relax(lat, seed=0xC2)
    props = material.measure_properties(settled)

    print(f"compound crystal: {label}")
    print("=" * 56)
    print("  measured by the EXISTING materials extractors (unchanged):")
    for key in ("fill_fraction", "density", "atomic_mass", "conductivity",
                "conductivity_continuous", "thermal_conductivity", "thermal_conductivity_phonon",
                "magnetism", "curie_temperature", "melting_temperature", "strength", "ductility"):
        if key in props:
            print(f"    {key:<28} {props[key]}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chem_explorer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="the root-atom table")
    p_list.set_defaults(func=_cmd_list)

    p_inspect = sub.add_parser("inspect-atom", help="one atom's descriptors + derived chemistry")
    p_inspect.add_argument("symbol", help="element symbol, e.g. O, Cl, Fe")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_build = sub.add_parser("build-molecule", help="form the compound of two elements")
    p_build.add_argument("a", help="first element symbol")
    p_build.add_argument("b", help="second element symbol")
    p_build.set_defaults(func=_cmd_build)

    p_meas = sub.add_parser("measure-compound",
                            help="build a compound's crystal and run the materials extractors")
    p_meas.add_argument("a", help="element symbol (single -> element crystal)")
    p_meas.add_argument("b", nargs="?", default=None, help="optional second element (binary compound)")
    p_meas.set_defaults(func=_cmd_measure)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
