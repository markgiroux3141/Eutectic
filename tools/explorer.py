"""Explorer / verification harness (spec §7).

The whole bet of this project is "do interesting properties actually emerge and feel
good?" — and the explorer is how we *see* the property space before trusting it.

Commands: ``list`` / ``inspect`` (single element) and ``percolation-sweep`` (the
threshold experiment) since M0; ``combine`` since M1; ``distribution`` (population view)
since M2; ``magnetism-sweep`` (Ising transition) and ``connectivity-sweep`` (higher-order
percolation transitions behind superconductivity) since M3.

Usage::

    python -m tools.explorer list
    python -m tools.explorer inspect iron
    python -m tools.explorer inspect iron --plot          # save a heatmap PNG
    python -m tools.explorer inspect iron --shape 32 32    # smaller lattice
    python -m tools.explorer inspect iron --seed 7         # alternate universe seed
    python -m tools.explorer combine iron copper           # combine two roots
    python -m tools.explorer percolation-sweep --plot      # conductivity threshold
    python -m tools.explorer magnetism-sweep --plot        # magnetism critical transition
    python -m tools.explorer distribution --n 500 --plot   # property distributions (§7)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from engine import elements
from engine.lattice import (
    RELAX_STEPS,
    RELAX_TEMPERATURE,
    Lattice,
    generate_base,
    relax,
)
from engine.properties import ising, percolation
from engine.registry import Registry
from engine.rng import SplitMix64, mix

# ASCII ramp for the text render of atom types (0 = empty).
_RAMP = " .:-=+*#%@"


def _ascii_render(lattice: Lattice, max_width: int = 64) -> str:
    """Render a 2D lattice's atom types as ASCII; for 3D, render the middle slice."""
    atom = lattice.atom_type
    if lattice.dim == 3:
        atom = atom[atom.shape[0] // 2]  # middle z-slice
    rows, cols = atom.shape
    # Downsample columns if wider than the terminal budget (nearest-neighbour).
    if cols > max_width:
        idx = np.linspace(0, cols - 1, max_width).astype(int)
        atom = atom[:, idx]
        cols = max_width
    n_types = int(atom.max())
    lines = []
    for r in range(rows):
        chars = []
        for c in range(cols):
            v = int(atom[r, c])
            if v == 0:
                chars.append(_RAMP[0])
            else:
                # Spread atom types across the visible part of the ramp.
                k = 1 + (v - 1) * (len(_RAMP) - 2) // max(n_types, 1)
                chars.append(_RAMP[min(k, len(_RAMP) - 1)])
        lines.append("".join(chars))
    return "\n".join(lines)


def _summary(lattice: Lattice) -> str:
    occ = lattice.occupied
    spin = lattice.spin
    net_mag = float(spin[occ == 1].mean()) if occ.any() else 0.0
    type_counts = np.bincount(lattice.atom_type.reshape(-1).astype(int))
    return (
        f"shape={lattice.shape} dim={lattice.dim} size={lattice.size}\n"
        f"fill_fraction={lattice.fill_fraction:.4f}\n"
        f"net_spin(on occupied)={net_mag:+.4f}\n"
        f"atom_type counts (0=empty): {type_counts.tolist()}\n"
        f"structural_signature={lattice.structural_signature():#018x}"
    )


def _plot(lattice: Lattice, element_id: str, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")  # headless; we save to file
    import matplotlib.pyplot as plt

    atom = lattice.atom_type
    if lattice.dim == 3:
        atom = atom[atom.shape[0] // 2]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{element_id}_lattice.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(atom, interpolation="nearest", cmap="viridis")
    ax.set_title(f"{element_id} — atom types (0 = empty)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def cmd_list(_args: argparse.Namespace) -> int:
    for eid in elements.all_ids():
        el = elements.get(eid)
        aff = el.base_affinities
        print(
            f"{eid:12s} mass={el.atomic_mass:7.3f} "
            f"bond={aff.get('bond_energy', 0):.2f} "
            f"mag={aff.get('magnetic_tendency', 0):.2f} "
            f"cond={aff.get('conduction_tendency', 0):.2f}"
        )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    el = elements.get(args.element)
    shape = tuple(args.shape) if args.shape else (64, 64)
    lattice = el.lattice(shape=shape, universe_seed=args.seed)

    print(f"=== {el.display_name} ({el.id}) | universe_seed={args.seed} ===")
    print(_summary(lattice))
    print(
        f"conductivity(boolean)={percolation.conductivity_boolean(lattice)} "
        f"spanning_fraction={percolation.spanning_fraction(lattice):.4f} "
        f"largest_cluster_fraction={percolation.largest_cluster_fraction(lattice):.4f}"
    )
    print()
    print(_ascii_render(lattice))

    if args.plot:
        out = _plot(lattice, el.id, Path(args.out))
        print(f"\nsaved heatmap -> {out}")
    return 0


def cmd_percolation_sweep(args: argparse.Namespace) -> int:
    """Sweep fill fraction and measure spanning probability (the threshold experiment).

    This tests the core rarity claim (spec §1, §5.2) *without* needing combine(): if a
    spanning cluster appears suddenly around the critical density, the substrate has the
    sharp threshold the whole design depends on.
    """
    shape = tuple(args.shape) if args.shape else (64, 64)
    fills = np.linspace(args.lo, args.hi, args.steps)
    trials = args.trials

    print(f"shape={shape}  trials/point={trials}  (2D site p_c ~ 0.5927)")
    print(f"{'fill':>6}  {'P(span)':>8}  {'mean largest-cluster frac':>26}  bar")
    rows = []
    for fill in fills:
        spanned = 0
        largest = 0.0
        for t in range(trials):
            # Deterministic, varied per (fill, trial): no global RNG (spec §6).
            seed = SplitMix64(_seed_for(fill, t, args.seed)).next_u64()
            # Build a lattice at a *targeted* fill fraction directly, independent of
            # element affinities, so this isolates the percolation threshold itself.
            lat = _lattice_at_fill(seed, shape, float(fill))
            if percolation.percolates_any_axis(lat):
                spanned += 1
            largest += percolation.largest_cluster_fraction(lat)
        p_span = spanned / trials
        mean_largest = largest / trials
        bar = "#" * int(round(p_span * 40))
        rows.append((float(fill), p_span, mean_largest))
        print(f"{fill:6.3f}  {p_span:8.3f}  {mean_largest:26.4f}  {bar}")

    if args.plot:
        out = _plot_sweep(rows, Path(args.out))
        print(f"\nsaved sweep plot -> {out}")
    return 0


def _seed_for(fill: float, trial: int, universe_seed: int) -> int:
    """Deterministic seed for a (fill, trial) sample point."""
    from engine.rng import mix

    # Quantize fill so the seed is stable and reproducible.
    return mix(int(round(fill * 1_000_000)), trial, universe_seed)


def _lattice_at_fill(seed: int, shape: tuple[int, ...], fill: float) -> Lattice:
    """Generate a lattice with occupancy at a *targeted* fill fraction.

    Bypasses element affinities so the sweep measures the percolation threshold purely as
    a function of fill density (the cleanest test of the substrate).
    """
    gen = SplitMix64(seed).numpy_generator()
    occupied = (gen.random(shape) < fill).astype(np.uint8)
    atom_type = np.where(occupied == 1, 1, 0).astype(np.int8)
    spin = np.ones(shape, dtype=np.int8)
    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin)


def _plot_sweep(rows, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "percolation_sweep.png"
    fills = [r[0] for r in rows]
    p_span = [r[1] for r in rows]
    largest = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fills, p_span, "o-", label="P(spanning cluster)")
    ax.plot(fills, largest, "s--", label="mean largest-cluster fraction")
    ax.axvline(0.5927, color="r", ls=":", label="2D site p_c ≈ 0.5927")
    ax.set_xlabel("fill fraction")
    ax.set_ylabel("probability / fraction")
    ax.set_title("Percolation threshold sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def cmd_magnetism_sweep(args: argparse.Namespace) -> int:
    """Sweep the magnetic moment and measure spontaneous magnetization (the Ising analog).

    The companion to ``percolation-sweep``: it tests the magnetism critical claim (spec
    §5.5) *without* needing combine(). Each lattice starts fully aligned (symmetry already
    broken) at a high fill (so percolation isn't the limiter), with a uniform moment ``m``
    setting the structural coupling ``J = J0 * m^2``. We then relax at the engine's fixed
    temperature and measure ``|M|``: below the critical moment thermal fluctuations destroy
    the order (``|M| -> 0``); above it the order survives (``|M| -> 1``). A sharp rise is
    the critical transition.
    """
    shape = tuple(args.shape) if args.shape else (64, 64)
    moments = np.linspace(args.lo, args.hi, args.steps)
    trials = args.trials

    m_c = float(np.sqrt(0.4407 * RELAX_TEMPERATURE))  # pure-lattice estimate; dilution shifts it up
    print(
        f"shape={shape}  trials/point={trials}  T={RELAX_TEMPERATURE}  steps={RELAX_STEPS}  "
        f"fill={args.fill}  (pure-lattice m_c ~ {m_c:.3f})"
    )
    print(f"{'moment':>7}  {'|M|':>6}  bar")
    rows = []
    for moment in moments:
        total = 0.0
        for t in range(trials):
            seed = _seed_for(float(moment), t, args.seed)
            lat = _uniform_moment_lattice(seed, shape, float(moment), args.fill)
            settled = relax(lat, mix(seed, 0x52))
            total += ising.magnetism(settled)
        mean_mag = total / trials
        bar = "#" * int(round(mean_mag * 40))
        rows.append((float(moment), mean_mag))
        print(f"{moment:7.3f}  {mean_mag:6.3f}  {bar}")

    if args.plot:
        out = _plot_magnetism_sweep(rows, m_c, Path(args.out))
        print(f"\nsaved sweep plot -> {out}")
    return 0


def _uniform_moment_lattice(
    seed: int, shape: tuple[int, ...], moment: float, fill: float
) -> Lattice:
    """A high-fill lattice with uniform moment, spins started fully aligned.

    Aligned start + high fill isolates the Ising transition from percolation: the only
    control parameter is the moment (hence the coupling). Measures whether order *survives*
    relaxation at coupling ``J0 * moment^2``.
    """
    gen = SplitMix64(seed).numpy_generator()
    occupied = (gen.random(shape) < fill).astype(np.uint8)
    atom_type = np.where(occupied == 1, 1, 0).astype(np.int8)
    spin = np.ones(shape, dtype=np.int8)  # symmetry broken up
    moment_field = (occupied.astype(np.float32) * np.float32(moment))
    return Lattice(occupied=occupied, atom_type=atom_type, spin=spin, moment=moment_field)


def _plot_magnetism_sweep(rows, m_c: float, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "magnetism_sweep.png"
    moments = [r[0] for r in rows]
    mag = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(moments, mag, "o-", label="|M| (settled)")
    ax.axvline(m_c, color="r", ls=":", label=f"pure-lattice m_c ≈ {m_c:.3f}")
    ax.set_xlabel("magnetic moment (coupling = J0·moment²)")
    ax.set_ylabel("|net magnetization|")
    ax.set_title("Magnetism critical transition (Ising)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def cmd_connectivity_sweep(args: argparse.Namespace) -> int:
    """Sweep fill and measure P(min-cut >= k) for several k -- the higher-order transitions.

    This is the evidence that superconductivity rides a *genuine* transition (spec §5.4),
    not a tuned cut on a smooth tail. k-edge-connectivity percolation has its own critical
    density p_k: a backbone with k edge-disjoint spanning paths first appears at p_k, and
    p_k rises with k (k=1 reproduces the ordinary site percolation point p_c ~ 0.593). Each
    curve should be a sharp step at its own p_k, the steps marching right as k grows.
    """
    from engine.properties import conductance as conductance_mod

    shape = tuple(args.shape) if args.shape else (64, 64)
    ks = sorted(args.ks)
    fills = np.linspace(args.lo, args.hi, args.steps)
    trials = args.trials

    print(f"shape={shape}  trials/point={trials}  (site p_c ~ 0.5927)")
    print(f"{'fill':>6}  " + "  ".join(f"k>={k:<2d}" for k in ks))
    rows = []
    for fill in fills:
        counts = {k: 0 for k in ks}
        for t in range(trials):
            seed = SplitMix64(_seed_for(float(fill), t, args.seed)).next_u64()
            lat = _lattice_at_fill(seed, shape, float(fill))
            w = max(
                conductance_mod._bottleneck_width_axis(lat, ax) for ax in range(lat.dim)
            )
            for k in ks:
                counts[k] += int(w >= k)
        probs = {k: counts[k] / trials for k in ks}
        rows.append((float(fill), probs))
        print(f"{fill:6.3f}  " + "  ".join(f"{probs[k]:4.2f}" for k in ks))

    if args.plot:
        out = _plot_connectivity_sweep(rows, ks, Path(args.out))
        print(f"\nsaved sweep plot -> {out}")
    return 0


def _plot_connectivity_sweep(rows, ks, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "connectivity_sweep.png"
    fills = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in ks:
        ax.plot(fills, [r[1][k] for r in rows], "o-", label=f"P(min-cut ≥ {k})")
    ax.axvline(0.5927, color="r", ls=":", label="site p_c ≈ 0.5927")
    ax.set_xlabel("fill fraction")
    ax.set_ylabel("probability")
    ax.set_title("Higher-order (k-edge-connectivity) percolation transitions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _print_properties(props: dict) -> None:
    for key in sorted(props):
        print(f"  {key:24s} {props[key]}")


def cmd_combine(args: argparse.Namespace) -> int:
    """Combine two materials and inspect the child (spec §4)."""
    shape = tuple(args.shape) if args.shape else (64, 64)
    reg = Registry(universe_seed=args.seed, shape=shape)
    reg.add_element(args.a)
    reg.add_element(args.b)
    child = reg.combine(args.a, args.b)

    print(f"=== combine({args.a}, {args.b}) | universe_seed={args.seed} ===")
    print(f"child id: {child.id}")
    print(f"lineage:  {child.lineage}")
    print(f"structural_signature={child.lattice.structural_signature():#018x}")
    print("properties:")
    _print_properties(child.properties)
    print()
    print("parent property comparison:")
    a_mat, b_mat = reg.get(args.a), reg.get(args.b)
    keys = sorted(child.properties)
    print(f"  {'property':24s} {args.a:>12s} {args.b:>12s} {'child':>12s}")
    for k in keys:
        print(
            f"  {k:24s} {a_mat.properties[k]:>12} "
            f"{b_mat.properties[k]:>12} {child.properties[k]:>12}"
        )
    print()
    print(_ascii_render(child.lattice))

    if args.plot:
        out = _plot(child.lattice, child.id, Path(args.out))
        print(f"\nsaved heatmap -> {out}")
    return 0


def sample_population(
    reg: Registry, n: int, rng: SplitMix64, *, chain: bool = False
) -> list:
    """Generate ``n`` deterministic random combinations and return the child materials.

    Roots are assumed already seeded into ``reg``. With ``chain=True``, each child is
    added back to the candidate pool so deeper (multi-step) combinations appear — this
    exercises the recursive part of the material space (spec §7).
    """
    pool = reg.all_ids()
    children = []
    for _ in range(n):
        a = pool[rng.randint(0, len(pool))]
        b = pool[rng.randint(0, len(pool))]
        child = reg.combine(a, b)
        children.append(child)
        if chain:
            pool.append(child.id)
    return children


def _text_histogram(values, *, bins: int = 20, width: int = 50, lo=None, hi=None) -> str:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "  (no data)"
    lo = float(arr.min()) if lo is None else lo
    hi = float(arr.max()) if hi is None else hi
    if hi <= lo:
        return f"  all values = {lo:.4g}"
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    peak = max(int(counts.max()), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / peak * width))
        lines.append(f"  [{edges[i]:8.3g}, {edges[i + 1]:8.3g})  {c:5d} {bar}")
    return "\n".join(lines)


def _stats_line(values) -> str:
    arr = np.asarray(values, dtype=float)
    return (
        f"n={arr.size}  min={arr.min():.4g}  median={np.median(arr):.4g}  "
        f"mean={arr.mean():.4g}  max={arr.max():.4g}"
    )


def cmd_distribution(args: argparse.Namespace) -> int:
    """Population view: combine many random pairs and plot property distributions (§7).

    This is the project's central checkpoint — do interesting properties actually emerge?
    Specifically: is conductivity bimodal around the percolation threshold (so the
    population lands on *both* sides), and is density a smooth, legible spread?
    """
    shape = tuple(args.shape) if args.shape else (64, 64)
    reg = Registry(universe_seed=args.seed, shape=shape)
    reg.seed_elements()
    rng = SplitMix64(mix(args.seed, 0xD15C0))
    children = sample_population(reg, args.n, rng, chain=args.chain)

    props = sorted(children[0].properties)
    series = {k: [c.properties[k] for c in children] for k in props}

    print(f"=== distribution over {len(children)} combinations ===")
    print(f"shape={shape}  universe_seed={args.seed}  chain={args.chain}")

    cond = np.asarray(series["conductivity"])
    frac_cond = float((cond >= 0.5).mean())
    print(
        f"\nconductivity: {frac_cond*100:.1f}% conduct, "
        f"{(1-frac_cond)*100:.1f}% insulate  "
        f"(both present => threshold is being crossed)"
    )

    # Magnetism transition (spec §5.5): a disordered mode near 0 and an ordered tail.
    mag = np.asarray(series["magnetism"])
    print(
        f"magnetism:    {(mag < 0.15).mean()*100:.1f}% disordered (<0.15), "
        f"{(mag > 0.5).mean()*100:.1f}% ordered (>0.5), "
        f"{((mag >= 0.15) & (mag <= 0.5)).mean()*100:.1f}% in-between "
        f"(empty middle => critical transition)"
    )

    # Superconductivity (spec §5.4): spanning AND k-edge-connected (loss-free backbone).
    from engine.properties.conductance import required_connectivity

    sc = np.asarray(series["superconductor"])
    n_sc = int((sc > 0.5).sum())
    # Every superconductor must also be a conductor (the double threshold).
    sc_conducts = bool(((sc > 0.5) <= (cond >= 0.5)).all())
    k = required_connectivity(shape)
    print(
        f"superconductors: {n_sc}/{len(children)} ({sc.mean()*100:.2f}%) "
        f"-- thin tail: spans AND edge-connectivity >= {k} (p_k > p_c); "
        f"all-also-conduct={sc_conducts}"
    )

    # 0/1 boolean series: a histogram of two spikes isn't illuminating.
    boolean_keys = {"conductivity", "superconductor"}
    for k in props:
        print(f"\n{k}:")
        print("  " + _stats_line(series[k]))
        if k not in boolean_keys:
            print(_text_histogram(series[k]))

    if args.plot:
        out = _plot_distributions(series, Path(args.out))
        print(f"\nsaved distribution plots -> {out}")
    return 0


def _plot_distributions(series: dict, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "distributions.png"
    keys = sorted(series)
    ncols = 3
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        ax.hist(series[k], bins=25, color="steelblue", edgecolor="white")
        ax.set_title(k)
        ax.grid(alpha=0.2)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle("Property distributions over random combinations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="explorer", description="Materials engine explorer (spec §7)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list root elements + affinities")
    p_list.set_defaults(func=cmd_list)

    p_ins = sub.add_parser("inspect", help="inspect a single element's base lattice")
    p_ins.add_argument("element", help="element id (e.g. iron)")
    p_ins.add_argument(
        "--shape", type=int, nargs="+", default=None,
        help="lattice shape, e.g. --shape 64 64 or --shape 16 16 16 (default 64 64)",
    )
    p_ins.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_ins.add_argument("--plot", action="store_true", help="save a matplotlib heatmap")
    p_ins.add_argument("--out", default="out", help="output dir for plots (default out/)")
    p_ins.set_defaults(func=cmd_inspect)

    p_sweep = sub.add_parser(
        "percolation-sweep",
        help="sweep fill fraction; measure spanning probability (threshold experiment)",
    )
    p_sweep.add_argument("--lo", type=float, default=0.40, help="min fill fraction")
    p_sweep.add_argument("--hi", type=float, default=0.75, help="max fill fraction")
    p_sweep.add_argument("--steps", type=int, default=15, help="number of fill points")
    p_sweep.add_argument("--trials", type=int, default=40, help="lattices per fill point")
    p_sweep.add_argument(
        "--shape", type=int, nargs="+", default=None,
        help="lattice shape (default 64 64); larger -> sharper threshold",
    )
    p_sweep.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_sweep.add_argument("--plot", action="store_true", help="save a matplotlib plot")
    p_sweep.add_argument("--out", default="out", help="output dir for plots")
    p_sweep.set_defaults(func=cmd_percolation_sweep)

    p_msweep = sub.add_parser(
        "magnetism-sweep",
        help="sweep magnetic moment; measure |M| (the Ising critical-transition experiment)",
    )
    p_msweep.add_argument("--lo", type=float, default=0.30, help="min moment")
    p_msweep.add_argument("--hi", type=float, default=1.30, help="max moment")
    p_msweep.add_argument("--steps", type=int, default=15, help="number of moment points")
    p_msweep.add_argument("--trials", type=int, default=20, help="lattices per moment point")
    p_msweep.add_argument(
        "--fill", type=float, default=0.85,
        help="fill fraction (high, to isolate the Ising transition from percolation)",
    )
    p_msweep.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_msweep.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_msweep.add_argument("--plot", action="store_true", help="save a matplotlib plot")
    p_msweep.add_argument("--out", default="out", help="output dir for plots")
    p_msweep.set_defaults(func=cmd_magnetism_sweep)

    p_csweep = sub.add_parser(
        "connectivity-sweep",
        help="sweep fill; P(min-cut>=k) for several k (higher-order percolation transitions)",
    )
    p_csweep.add_argument("--lo", type=float, default=0.45, help="min fill fraction")
    p_csweep.add_argument("--hi", type=float, default=0.85, help="max fill fraction")
    p_csweep.add_argument("--steps", type=int, default=17, help="number of fill points")
    p_csweep.add_argument("--trials", type=int, default=30, help="lattices per fill point")
    p_csweep.add_argument(
        "--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="connectivity levels k"
    )
    p_csweep.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_csweep.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_csweep.add_argument("--plot", action="store_true", help="save a matplotlib plot")
    p_csweep.add_argument("--out", default="out", help="output dir for plots")
    p_csweep.set_defaults(func=cmd_connectivity_sweep)

    p_comb = sub.add_parser("combine", help="combine two materials and inspect the child")
    p_comb.add_argument("a", help="first element/material id")
    p_comb.add_argument("b", help="second element/material id")
    p_comb.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_comb.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_comb.add_argument("--plot", action="store_true", help="save a heatmap of the child")
    p_comb.add_argument("--out", default="out", help="output dir for plots")
    p_comb.set_defaults(func=cmd_combine)

    p_dist = sub.add_parser(
        "distribution",
        help="combine many random pairs; show property distributions (the §7 checkpoint)",
    )
    p_dist.add_argument("--n", type=int, default=500, help="number of combinations")
    p_dist.add_argument(
        "--chain", action="store_true",
        help="feed children back into the pool for multi-step combinations",
    )
    p_dist.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_dist.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_dist.add_argument("--plot", action="store_true", help="save histogram grid")
    p_dist.add_argument("--out", default="out", help="output dir for plots")
    p_dist.set_defaults(func=cmd_distribution)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
