"""Explorer / verification harness (spec §7).

The whole bet of this project is "do interesting properties actually emerge and feel
good?" — and the explorer is how we *see* the property space before trusting it.

M0 scope: ``inspect`` and ``list`` only — render/inspect a single element's generated
lattice (text always; optional matplotlib heatmap). Distribution plots over random
combinations arrive once ``combine()`` + properties exist (M2+).

Usage::

    python -m tools.explorer list
    python -m tools.explorer inspect iron
    python -m tools.explorer inspect iron --plot          # save a heatmap PNG
    python -m tools.explorer inspect iron --shape 32 32    # smaller lattice
    python -m tools.explorer inspect iron --seed 7         # alternate universe seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from engine import elements
from engine.lattice import Lattice, generate_base
from engine.properties import percolation
from engine.rng import SplitMix64

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

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
