"""Chemistry explorer / verification harness (chemistry-engine-spec §16).

Mirrors :mod:`tools.explorer` (ASCII stdout; matplotlib reserved for later sweeps). The
whole no-fudge bet at this level is "does real chemistry *emerge* from the distilled
descriptors?" — this tool is how we see it before trusting it.

Commands grow with the C0..C5 ladder. C0 ships:

* ``list``          — the root-atom table (authored descriptors at a glance).
* ``inspect-atom``  — one atom's descriptors + everything derived parameter-free from Z
                      (configuration, valence, capacity, lone pairs, ion charge, orbitals,
                      and the Z_eff-derived EN/radius trend proxies).

Usage::

    python -m tools.chem_explorer list
    python -m tools.chem_explorer inspect-atom O
    python -m tools.chem_explorer inspect-atom Cl
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from chemistry import atoms


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chem_explorer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="the root-atom table")
    p_list.set_defaults(func=_cmd_list)

    p_inspect = sub.add_parser("inspect-atom", help="one atom's descriptors + derived chemistry")
    p_inspect.add_argument("symbol", help="element symbol, e.g. O, Cl, Fe")
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
