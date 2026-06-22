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
* ``react``          — (C3) a reaction's ΔH/ΔS/ΔG/K at given conditions, whether it proceeds,
                       and (for an entropy-favored one) the temperature threshold T* where ΔG
                       flips sign.
* ``condition-sweep`` — (C3) ΔG(T) across a temperature range, marking the ΔG=0 crossing; with
                       ``--pressure`` shows the Le Chatelier shift of that threshold.
* ``kinetics``       — (C4) a reaction's Ea and Arrhenius rate(T), the favourable-but-trapped
                       story, and how a ``--catalyst`` lowers the barrier (rate up, ΔG unchanged).

Reactions are given as ``REACTANTS = PRODUCTS`` with ``+`` separators; each term is a species
token (see :func:`_parse_species`): an element symbol is its diatomic gas (``H``→H₂); ``A.B`` is
the binary compound (``H.O``→H₂O); a leading ``=`` count and trailing ``/phase`` are optional, and
``~SYM`` is a free atom (``~H``). Examples::

    python -m tools.chem_explorer react "2 H + O = 2 H.O"      # 2H₂ + O₂ -> 2H₂O
    python -m tools.chem_explorer react "O = 2 ~O" --temperature 9   # O₂ -> 2O (dissociation)
    python -m tools.chem_explorer condition-sweep "Cl = 2 ~Cl"       # ΔG(T), Cl₂ dissociation
    python -m tools.chem_explorer condition-sweep "O = 2 ~O" --pressure 10  # Le Chatelier

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


def _parse_species(token: str):
    """Parse one species token into a :class:`chemistry.reaction.Species`.

    Grammar (suffix ``/phase`` optional: gas|liquid|solid, default gas):
      ``H``       → the element's diatomic gas (H₂)
      ``H.O``     → the binary compound of two elements (H₂O)
      ``~O``      → a free monatomic species (atomic O)
    """
    from chemistry import reaction as rx
    from chemistry.conditions import Phase

    phase = Phase.GAS
    if "/" in token:
        token, pname = token.split("/", 1)
        phase = Phase[pname.strip().upper()]
    token = token.strip()
    if token.startswith("~"):
        return rx.atom(token[1:], phase)
    if "." in token:
        a, b = token.split(".", 1)
        return rx.binary(a, b, phase)
    return rx.diatomic(token, phase)


def _parse_reaction(text: str):
    """Parse ``"2 A + B = 2 C"`` into a :class:`chemistry.reaction.Reaction`."""
    from chemistry import reaction as rx

    if "=" not in text:
        raise ValueError("reaction must contain '=' separating reactants from products")
    lhs, rhs = text.split("=", 1)

    def side(part: str):
        terms = []
        for chunk in part.split("+"):
            chunk = chunk.strip()
            if not chunk:
                continue
            bits = chunk.split(None, 1)
            if len(bits) == 2 and bits[0].isdigit():
                coef, tok = int(bits[0]), bits[1]
            else:
                coef, tok = 1, chunk
            terms.append((_parse_species(tok), coef))
        return tuple(terms)

    return rx.reaction(side(lhs), side(rhs))


def _conditions(args):
    from chemistry.conditions import ChemConditions

    return ChemConditions(
        temperature=args.temperature,
        pressure=getattr(args, "pressure", 0.0) or 1.0,
        concentration=getattr(args, "concentration", 1.0),
    )


def _fmt_side(side) -> str:
    return " + ".join(f"{c if c != 1 else ''}{sp.formula}".strip() for sp, c in side)


def _cmd_react(args: argparse.Namespace) -> int:
    try:
        r = _parse_reaction(args.reaction)
    except (ValueError, KeyError) as exc:
        print(f"could not parse reaction: {exc}", file=sys.stderr)
        return 2
    cond = _conditions(args)
    print(f"{_fmt_side(r.reactants)}  ->  {_fmt_side(r.products)}")
    print("=" * 56)
    print(f"  conditions         T={cond.temperature:g}  P={cond.pressure:g}  c={cond.concentration:g}")
    print(f"  delta_H            {r.delta_H:+.2f}   ({'exothermic' if r.delta_H < 0 else 'endothermic'})")
    print(f"  delta_S            {r.delta_S:+.2f}   (Dn_gas={r.delta_n_gas:+d}, Dn_total={r.delta_n_total:+d})")
    print(f"  delta_G            {r.delta_G(cond):+.2f}   ({'SPONTANEOUS' if r.is_spontaneous(cond) else 'not spontaneous'})")
    print(f"  K = exp(-dG/RT)    {r.equilibrium_constant(cond):.4g}   ({'products' if r.equilibrium_constant(cond) > 1 else 'reactants'} favored)")
    t_star = r.crossover_temperature(cond)
    if t_star is None:
        print("  crossover T*       none (delta_G does not change sign for T>0)")
    else:
        print(f"  crossover T*       {t_star:.3f}   (delta_G flips sign here; heat above to drive it)")
    print()
    print("  NB delta_H is Hess's law over the C1 bond energies, which are LINEAR in bond order")
    print("  (overstating double/triple bonds): reactions that BREAK a multiple bond (e.g.")
    print("  2H2+O2->2H2O) can read the wrong delta_H sign. Dissociation/recombination is robust.")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    try:
        r = _parse_reaction(args.reaction)
    except (ValueError, KeyError) as exc:
        print(f"could not parse reaction: {exc}", file=sys.stderr)
        return 2
    from chemistry.conditions import ChemConditions

    p = args.pressure or 1.0
    c = args.concentration
    t_star = r.crossover_temperature(ChemConditions(pressure=p, concentration=c))
    print(f"{_fmt_side(r.reactants)}  ->  {_fmt_side(r.products)}   (P={p:g}, c={c:g})")
    print("=" * 56)
    print(f"  delta_H={r.delta_H:+.2f}  delta_S={r.delta_S:+.2f}  ->  ", end="")
    print("T* = none" if t_star is None else f"T* = {t_star:.3f}")
    print(f"  {'T':>7} {'delta_G':>12}  spont?")
    print("  " + "-" * 32)
    lo, hi, n = args.t_min, args.t_max, args.steps
    prev_sign = None
    for i in range(n + 1):
        t = lo + (hi - lo) * i / n
        cond = ChemConditions(temperature=t, pressure=p, concentration=c)
        dg = r.delta_G(cond)
        spont = "yes" if dg < 0 else " no"
        mark = ""
        sign = dg < 0
        if prev_sign is not None and sign != prev_sign:
            mark = "  <== delta_G sign-crossing"
        prev_sign = sign
        print(f"  {t:>7.2f} {dg:>12.2f}   {spont}{mark}")
    return 0


def _cmd_kinetics(args: argparse.Namespace) -> int:
    try:
        r = _parse_reaction(args.reaction)
    except (ValueError, KeyError) as exc:
        print(f"could not parse reaction: {exc}", file=sys.stderr)
        return 2
    from chemistry import kinetics as kin
    from chemistry.conditions import ChemConditions

    cat = {args.catalyst} if args.catalyst else set()
    k = kin.kinetics(r, cat)
    base = ChemConditions(temperature=args.temperature)
    print(f"{_fmt_side(r.reactants)}  ->  {_fmt_side(r.products)}")
    print("=" * 56)
    print(f"  reactant_bond_energy   {k.reactant_bond_energy:.2f}  (the bonds that must break)")
    print(f"  Ea (uncatalyzed)       {k.base_activation_energy():.2f}")
    dg = r.delta_G(base)
    print(f"  delta_G @ T={args.temperature:g}          {dg:+.2f}  "
          f"({'favorable' if dg < 0 else 'unfavorable'} -- rate gates it regardless)")
    print()
    print(f"  {'T':>7} {'rate':>13} {'spont?':>8}")
    print("  " + "-" * 32)
    for t in (1.0, 2.0, 4.0, 8.0, 16.0):
        c = ChemConditions(temperature=t)
        spont = "yes" if r.is_spontaneous(c) else " no"
        print(f"  {t:>7.2f} {k.rate(c):>13.3e} {spont:>8}")
    if args.catalyst:
        print()
        plain = ChemConditions(temperature=args.temperature)
        with_cat = ChemConditions(temperature=args.temperature, catalysts=frozenset(cat))
        print(f"  catalyst {args.catalyst!r}: Ea {k.activation_energy(plain):.2f} -> "
              f"{k.activation_energy(with_cat):.2f}; rate "
              f"{k.rate(plain):.3e} -> {k.rate(with_cat):.3e} "
              f"({k.rate(with_cat) / max(k.rate(plain), 1e-300):.1f}x); "
              f"delta_G {r.delta_G(plain):+.2f} (UNCHANGED)")
        target = 1.0
        t0 = k.temperature_for_rate(target, plain)
        t1 = k.temperature_for_rate(target, with_cat)
        print(f"  temperature for rate>{target:g}:  {t0:.3f} -> {t1:.3f}  (catalyst lowers the threshold)")
    print()
    print("  NB rate(T) is a SMOOTH exponential, not a sharp transition: real ignition is thermal")
    print("  runaway (feedback) we do not model. Absolute rate is uncalibrated; the T-sensitivity")
    print("  and the catalyst's fixed barrier-shift are what's emergent.")
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

    p_react = sub.add_parser("react", help="a reaction's thermodynamics at given conditions")
    p_react.add_argument("reaction", help='e.g. "2 H + O = 2 H.O"  (H->H2, H.O->H2O, ~O->atom)')
    p_react.add_argument("--temperature", "-T", type=float, default=1.0, help="temperature (default 1.0)")
    p_react.add_argument("--pressure", "-P", type=float, default=1.0, help="pressure (default 1.0)")
    p_react.add_argument("--concentration", "-c", type=float, default=1.0, help="activity (default 1.0)")
    p_react.set_defaults(func=_cmd_react)

    p_sweep = sub.add_parser("condition-sweep", help="delta_G(T) across a temperature range")
    p_sweep.add_argument("reaction", help='e.g. "Cl = 2 ~Cl"  (Cl2 dissociation)')
    p_sweep.add_argument("--t-min", type=float, default=0.5, help="sweep start T (default 0.5)")
    p_sweep.add_argument("--t-max", type=float, default=12.0, help="sweep end T (default 12.0)")
    p_sweep.add_argument("--steps", type=int, default=12, help="number of sweep steps (default 12)")
    p_sweep.add_argument("--pressure", "-P", type=float, default=1.0, help="pressure (Le Chatelier)")
    p_sweep.add_argument("--concentration", "-c", type=float, default=1.0, help="activity (mass action)")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_kin = sub.add_parser("kinetics", help="Arrhenius rate, Ea, and catalyst effect")
    p_kin.add_argument("reaction", help='e.g. "C + O = C.O"  (C+O2->CO2, exergonic + trapped)')
    p_kin.add_argument("--temperature", "-T", type=float, default=1.0, help="temperature (default 1.0)")
    p_kin.add_argument("--catalyst", default=None, help="a species that catalyses this reaction, e.g. Pt")
    p_kin.set_defaults(func=_cmd_kinetics)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
