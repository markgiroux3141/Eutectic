"""Explorer / verification harness (spec §7).

The whole bet of this project is "do interesting properties actually emerge and feel
good?" — and the explorer is how we *see* the property space before trusting it.

Commands: ``list`` / ``inspect`` (single element) and ``percolation-sweep`` (the
threshold experiment) since M0; ``combine`` since M1; ``distribution`` (population view)
since M2; ``magnetism-sweep`` (Ising transition) and ``connectivity-sweep`` (higher-order
percolation transitions behind superconductivity) since M3; ``temperature-sweep`` (Curie
point) and ``process-compare`` (synthesis trajectories) since M4; ``melting-sweep`` (the
occupancy order-disorder / melting transition) since M5; ``sc-sweep`` (phase-coherence /
BKT superconducting Tc) since M6.

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


def cmd_temperature_sweep(args: argparse.Namespace) -> int:
    """Sweep temperature for one material; show M(T) and C(T) and the Curie point (M4).

    The condition-as-measurement-axis view (docs §2/§4): a material *is* a structure, and we
    measure its order parameter and heat capacity as the thermal ensemble's temperature
    varies. The keystone is visible here — the heat-capacity peak (the universal transition
    detector) lands exactly where the magnetization collapses. Pass one id for a root, two
    to combine first.
    """
    from engine import thermal
    from engine.properties import ising

    shape = tuple(args.shape) if args.shape else (64, 64)
    reg = Registry(universe_seed=args.seed, shape=shape)
    reg.add_element(args.a)
    if args.b is not None:
        reg.add_element(args.b)
        mat = reg.combine(args.a, args.b)
        label = f"combine({args.a}, {args.b})"
    else:
        mat = reg.get(args.a)
        label = args.a

    ref_mag = ising.magnetism(mat.lattice)
    temps = np.linspace(args.lo, args.hi, args.steps)
    sweep = thermal.temperature_sweep(
        mat.lattice, temps, field=args.field,
        burn_in=args.burn_in, n_samples=args.samples, sample_every=args.sample_every,
    )
    mags = np.array([s.mean_abs_mag for s in sweep])
    caps = np.array([s.heat_capacity for s in sweep])
    peak_i = int(np.argmax(caps))
    tc = float(temps[peak_i]) if mags.max() >= 0.30 else None

    print(f"=== temperature-sweep: {label} | seed={args.seed} shape={shape} ===")
    print(f"reference magnetism (at standard T0): {ref_mag:.3f}   field H={args.field}")
    print(f"{'T':>6}  {'<|M|>':>6}  {'C':>8}  C-bar")
    cmax = max(float(caps.max()), 1e-9)
    for i, T in enumerate(temps):
        bar = "#" * int(round(caps[i] / cmax * 40))
        mark = "  <-- C peak (Tc)" if i == peak_i and tc is not None else ""
        print(f"{T:6.3f}  {mags[i]:6.3f}  {caps[i]:8.4f}  {bar}{mark}")

    if tc is None:
        print("\nno Curie point: the material never orders over this range (paramagnet).")
    else:
        # The keystone check, surfaced: does |M| collapse at the C peak?
        below = mags[temps < tc]
        above = mags[temps > tc]
        lo_m = below.mean() if below.size else mags[0]
        hi_m = above.mean() if above.size else mags[-1]
        print(
            f"\nTc = {tc:.3f} (heat-capacity peak). "
            f"order parameter: <|M|>={lo_m:.2f} below -> {hi_m:.2f} above "
            f"=> {'COLLAPSES at Tc (keystone holds)' if lo_m - hi_m > 0.3 else 'no clear collapse'}"
        )

    if args.plot:
        out = _plot_temperature_sweep(temps, mags, caps, tc, label, Path(args.out))
        print(f"\nsaved sweep plot -> {out}")
    return 0


def _plot_temperature_sweep(temps, mags, caps, tc, label, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "temperature_sweep.png"
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(temps, mags, "o-", color="steelblue", label="⟨|M|⟩ (order parameter)")
    ax1.set_xlabel("temperature T")
    ax1.set_ylabel("⟨|M|⟩", color="steelblue")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(temps, caps, "s--", color="crimson", label="C = Var(E)/(N·T²)")
    ax2.set_ylabel("heat capacity C", color="crimson")
    if tc is not None:
        ax1.axvline(tc, color="k", ls=":", label=f"Tc (C peak) = {tc:.3f}")
    ax1.set_title(f"Temperature sweep — {label}")
    ax1.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def cmd_melting_sweep(args: argparse.Namespace) -> int:
    """Sweep temperature for one material's OCCUPANCY; show ψ(T), C(T), ρ(T) and T_m (M5).

    The positional twin of ``temperature-sweep`` (docs §4/§5): occupancy is a thermal
    lattice gas, and its order-disorder transition *is* crystalline melting. The keystone is
    visible here three ways at once — the staggered (sublattice) order parameter ψ collapses,
    the heat-capacity peak (universal detector) lands at the same T, and the mean density ρ
    stays pinned at ½ across it (order lost at *fixed* density → melting, not sublimation).
    For a plain/uniform-cohesion material the peak sits at the textbook 2D point 2.269·J0·⟨c²⟩
    with no free parameter. Pass one id for a root, two to combine first.
    """
    from engine import thermal

    shape = tuple(args.shape) if args.shape else (64, 64)
    reg = Registry(universe_seed=args.seed, shape=shape)
    reg.add_element(args.a)
    if args.b is not None:
        reg.add_element(args.b)
        mat = reg.combine(args.a, args.b)
        label = f"combine({args.a}, {args.b})"
    else:
        mat = reg.get(args.a)
        label = args.a
    lat = mat.lattice

    coh2 = float((np.asarray(lat.cohesion, dtype=float) ** 2).mean())
    predicted = thermal.ISING_TC_2D * thermal.COHESION_J0 * coh2
    lo = args.lo if args.lo is not None else 0.35 * predicted
    hi = args.hi if args.hi is not None else 1.55 * predicted
    temps = np.linspace(lo, hi, args.steps)
    sweep = thermal.occupancy_temperature_sweep(
        lat, temps, pressure=args.pressure,
        burn_in=args.burn_in, n_samples=args.samples, sample_every=args.sample_every,
    )
    psis = np.array([s.staggered_order for s in sweep])
    caps = np.array([s.heat_capacity for s in sweep])
    rhos = np.array([s.mean_density for s in sweep])
    peak_i = int(np.argmax(caps))
    tm = float(temps[peak_i]) if psis.max() >= 0.30 else None

    print(f"=== melting-sweep: {label} | seed={args.seed} shape={shape} ===")
    print(f"mean cohesion = {float(np.asarray(lat.cohesion).mean()):.3f}   "
          f"analytic T_m ~ 2.269*J0*<c^2> = {predicted:.3f}   pressure P={args.pressure}")
    print(f"{'T':>6}  {'psi':>6}  {'rho':>6}  {'C':>8}  C-bar")
    cmax = max(float(caps.max()), 1e-9)
    for i, T in enumerate(temps):
        bar = "#" * int(round(caps[i] / cmax * 40))
        mark = "  <-- C peak (Tm)" if i == peak_i and tm is not None else ""
        print(f"{T:6.3f}  {psis[i]:6.3f}  {rhos[i]:6.3f}  {caps[i]:8.4f}  {bar}{mark}")

    if tm is None:
        print("\nno melting point: the structure never develops sublattice order in this range.")
    else:
        below = psis[temps < tm]; above = psis[temps > tm]
        lo_p = below.mean() if below.size else psis[0]
        hi_p = above.mean() if above.size else psis[-1]
        rho_span = float(rhos.max() - rhos.min())
        print(
            f"\nT_m = {tm:.3f} (heat-capacity peak). "
            f"order parameter psi: {lo_p:.2f} below -> {hi_p:.2f} above "
            f"=> {'COLLAPSES at Tm' if lo_p - hi_p > 0.3 else 'no clear collapse'}; "
            f"density rho stays {'FIXED' if rho_span < 0.05 else 'NOT fixed'} "
            f"(range {rho_span:.3f}) => crystalline melting, not sublimation."
        )

    if args.plot:
        out = _plot_melting_sweep(temps, psis, caps, rhos, tm, label, Path(args.out))
        print(f"\nsaved sweep plot -> {out}")
    return 0


def _plot_melting_sweep(temps, psis, caps, rhos, tm, label, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "melting_sweep.png"
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(temps, psis, "o-", color="seagreen", label="ψ (staggered/positional order)")
    ax1.plot(temps, rhos, "^-", color="goldenrod", label="ρ (density — stays ~½)")
    ax1.set_xlabel("temperature T")
    ax1.set_ylabel("ψ  /  ρ")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(temps, caps, "s--", color="crimson", label="C = Var(E)/(N·T²)")
    ax2.set_ylabel("heat capacity C", color="crimson")
    if tm is not None:
        ax1.axvline(tm, color="k", ls=":", label=f"Tm (C peak) = {tm:.3f}")
    ax1.set_title(f"Melting sweep (occupancy order-disorder) — {label}")
    ax1.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def cmd_sc_sweep(args: argparse.Namespace) -> int:
    """Sweep T for one material's phase coherence; show Y(T), the BKT line, and Tc (M6).

    Honest superconductivity: a conducting backbone carries an XY phase field, and its helicity
    modulus Y(T) (phase stiffness) is the order parameter. The transition Tc is where Y(T) crosses
    the BKT universal line Y=(2/pi)T -- NOT the heat-capacity peak, which for a BKT transition
    sits ABOVE Tc (we print both to show it). A fully-conducting lattice lands on the textbook
    0.893; a diluted/tortuous backbone coheres lower, so Tc emerges from structure. One id for a
    root, two to combine.
    """
    from engine import thermal

    shape = tuple(args.shape) if args.shape else (64, 64)
    reg = Registry(universe_seed=args.seed, shape=shape)
    reg.add_element(args.a)
    if args.b is not None:
        reg.add_element(args.b)
        mat = reg.combine(args.a, args.b)
        label = f"combine({args.a}, {args.b})"
    else:
        mat = reg.get(args.a)
        label = args.a
    lat = mat.lattice

    temps = np.linspace(args.lo, args.hi, args.steps)
    sweep = thermal.xy_temperature_sweep(
        lat, temps, burn_in=args.burn_in, n_samples=args.samples, sample_every=args.sample_every,
    )
    hel = np.array([s.helicity_modulus for s in sweep])
    caps = np.array([s.heat_capacity for s in sweep])
    line = (2.0 / np.pi) * temps
    tc = thermal._universal_crossing(temps, hel.tolist())
    c_peak_T = float(temps[int(np.argmax(caps))])

    print(f"=== sc-sweep: {label} | seed={args.seed} shape={shape} ===")
    print(f"conductivity(boolean)={mat.properties['conductivity']:.0f}  "
          f"edge_connectivity={mat.properties['edge_connectivity']:.0f}  "
          f"(textbook clean-lattice Tc = 0.893)")
    print(f"{'T':>6}  {'Y(stiffness)':>12}  {'(2/pi)T':>8}  {'C':>7}")
    for i, T in enumerate(temps):
        mark = "  <-- Y crosses line (Tc)" if (tc is not None and abs(T - tc) < (temps[1]-temps[0])) else ""
        cmark = "  [C peak]" if i == int(np.argmax(caps)) else ""
        print(f"{T:6.3f}  {hel[i]:12.3f}  {line[i]:8.3f}  {caps[i]:7.3f}{mark}{cmark}")

    if tc is None:
        print("\nno superconducting Tc: the backbone never phase-coheres (not a superconductor).")
    else:
        print(f"\nTc = {tc:.3f}  (helicity modulus crosses the BKT universal line). "
              f"C-peak at {c_peak_T:.3f} is ABOVE Tc -> the C-peak is NOT the SC detector (BKT).")

    if args.plot:
        out = _plot_sc_sweep(temps, hel, line, caps, tc, label, Path(args.out))
        print(f"\nsaved sweep plot -> {out}")
    return 0


def _plot_sc_sweep(temps, hel, line, caps, tc, label, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sc_sweep.png"
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(temps, hel, "o-", color="teal", label="Υ helicity modulus (phase stiffness)")
    ax1.plot(temps, line, "k--", label="BKT universal line Υ = (2/π)·T")
    ax1.set_xlabel("temperature T")
    ax1.set_ylabel("helicity modulus Υ")
    ax2 = ax1.twinx()
    ax2.plot(temps, caps, "s:", color="crimson", alpha=0.6, label="C (peak is ABOVE Tc)")
    ax2.set_ylabel("heat capacity C", color="crimson")
    if tc is not None:
        ax1.axvline(tc, color="seagreen", ls="-", alpha=0.7, label=f"Tc (BKT) = {tc:.3f}")
    ax1.set_title(f"Superconducting phase coherence (XY/BKT) — {label}")
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def cmd_process_compare(args: argparse.Namespace) -> int:
    """Run several synthesis processes on one structure and compare the results (process layer).

    The same ingredients, processed differently (anneal-slow / anneal-fast / quench /
    field-cool), settle into measurably different magnetic structures. This is the
    "wrong process = different material" view: remanence (net ⟨|M|⟩ retained), domain size,
    domain-wall density, and energy/cell all move with the cooling history. Pass one id for a
    root, two to combine first.
    """
    from engine import process as proc, thermal
    from engine.properties import ising, microstructure
    from engine.rng import mix

    shape = tuple(args.shape) if args.shape else (64, 64)
    reg = Registry(universe_seed=args.seed, shape=shape)
    reg.add_element(args.a)
    if args.b is not None:
        reg.add_element(args.b)
        mat = reg.combine(args.a, args.b)
        label = f"combine({args.a}, {args.b})"
    else:
        mat = reg.get(args.a)
        label = args.a
    lat = mat.lattice
    n_active = max(int((lat.occupied == 1).sum()), 1)

    budget = args.budget
    processes = [
        proc.anneal(budget=budget, name="anneal-slow"),
        proc.anneal(budget=budget, ramp_sweeps=max(budget // 6, 4), name="anneal-fast"),
        proc.quench(budget=budget, name="quench"),
        proc.field_cool(budget=budget, field=args.field, name="field-cool"),
    ]

    print(f"=== process-compare: {label} | seed={args.seed} shape={shape} budget={budget} ===")
    print(f"reference magnetism (default settle): {mat.properties['magnetism']:.3f}")
    print(f"{'process':12s} {'remanence':>9} {'domain':>7} {'walls':>6} {'E/cell':>8}")
    results = []
    for p in processes:
        out = proc.run_process(lat, p, seed=mix(args.seed, p.signature()))
        rem = ising.magnetism(out)
        dom = microstructure.domain_fraction(out)
        wall = microstructure.domain_wall_density(out)
        e = thermal.energy(out, out.spin) / n_active
        results.append((p.name, out, rem, dom, wall, e))
        print(f"{p.name:12s} {rem:9.3f} {dom:7.3f} {wall:6.3f} {e:8.3f}")

    print(
        "\nread: slower cooling -> lower E/cell, larger domain, fewer walls; "
        "field-cool -> high remanence (the process payoff)."
    )

    # --- M5: occupancy is thermal, so a process can also change a STRUCTURAL (non-magnetic)
    # property. Cool under a high vs low chemical potential (pressure) -> different density.
    from engine.process import Process, Stage
    from engine.properties import microstructure as micro

    def sinter(pressure: float, name: str) -> Process:
        return Process(
            (Stage(3.5, 3.5, 8, pressure=pressure),
             Stage(3.5, 0.4, budget - 8, pressure=pressure)),
            name=name, evolve_occupancy=True,
        )

    print(f"\n--- structural process-compare (M5: occupancy thermal) ---")
    print(f"{'process':14s} {'density':>8} {'pos.order':>9}")
    struct = []
    for p in (sinter(+6.0, "sinter-dense"), sinter(0.0, "sinter-std"), sinter(-6.0, "sinter-porous")):
        out = proc.run_process(lat, p, seed=mix(args.seed, p.signature()))
        dens = out.fill_fraction
        po = micro.positional_order(out)
        struct.append((p.name, dens, po))
        print(f"{p.name:14s} {dens:8.3f} {po:9.3f}")
    print("read: cooling under higher pressure (mu) freezes in a denser solid "
          "(density path-dependence -- the M5 structural payoff).")

    if args.plot:
        out = _plot_process_compare(results, label, Path(args.out))
        print(f"\nsaved spin-structure plot -> {out}")
    return 0


def _plot_process_compare(results, label, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "process_compare.png"
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, out, rem, dom, wall, e) in zip(axes, results):
        spin = out.spin
        if out.dim == 3:
            spin = spin[spin.shape[0] // 2]
        # show spin only on occupied cells (empty -> neutral gray via masked display)
        disp = np.where(out.occupied if out.dim == 2 else out.occupied[out.shape[0] // 2],
                        spin, 0)
        ax.imshow(disp, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"{name}\nrem={rem:.2f} dom={dom:.2f}")
        ax.axis("off")
    fig.suptitle(f"Spin microstructure by process — {label}")
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

    # Curie temperature (M4): the condition-dependent property — Tc>0 only where ordered.
    tc = np.asarray(series["curie_temperature"])
    has_tc = tc > 0
    if has_tc.any():
        tc_vals = tc[has_tc]
        print(
            f"curie point:  {has_tc.mean()*100:.1f}% have a Tc>0 "
            f"(ferromagnetic at standard conditions); "
            f"Tc range {tc_vals.min():.2f}-{tc_vals.max():.2f}, median {np.median(tc_vals):.2f} "
            f"(rest are paramagnets => Tc=0)"
        )
    else:
        print("curie point:  none in this sample (no ferromagnets drawn)")

    # Backbone redundancy (M6): edge-connectivity is the *structural input* to the
    # phase-coherence superconducting Tc (a redundant backbone is phase-stiff -> higher Tc). The
    # SC transition itself is measured on demand -- see the `sc-sweep` view.
    ec = np.asarray(series["edge_connectivity"])
    redundant = ec[ec >= 3]
    print(
        f"backbone:     {(ec > 0).mean()*100:.0f}% span (edge-connectivity>0), "
        f"{(ec >= 3).mean()*100:.0f}% redundant (>=3, high-Tc-capable); "
        f"max edge-connectivity {int(ec.max())}  -> SC Tc rides this (run `sc-sweep`)"
    )

    # 0/1 boolean series: a histogram of two spikes isn't illuminating.
    boolean_keys = {"conductivity"}
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

    p_tsweep = sub.add_parser(
        "temperature-sweep",
        help="sweep T for one material; show M(T), C(T) and the Curie point (M4 keystone)",
    )
    p_tsweep.add_argument("a", help="element/material id (or first of two to combine)")
    p_tsweep.add_argument("b", nargs="?", default=None, help="optional second id to combine")
    p_tsweep.add_argument("--lo", type=float, default=1.0, help="min temperature")
    p_tsweep.add_argument("--hi", type=float, default=4.0, help="max temperature")
    p_tsweep.add_argument("--steps", type=int, default=16, help="number of temperatures")
    p_tsweep.add_argument("--field", type=float, default=0.0, help="magnetic field H")
    p_tsweep.add_argument("--burn-in", type=int, default=80, help="equilibration sweeps")
    p_tsweep.add_argument("--samples", type=int, default=50, help="samples per temperature")
    p_tsweep.add_argument(
        "--sample-every", type=int, default=2, help="sweeps between samples"
    )
    p_tsweep.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_tsweep.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_tsweep.add_argument("--plot", action="store_true", help="save M(T)/C(T) plot")
    p_tsweep.add_argument("--out", default="out", help="output dir for plots")
    p_tsweep.set_defaults(func=cmd_temperature_sweep)

    p_melt = sub.add_parser(
        "melting-sweep",
        help="sweep T for one material's occupancy; ψ(T), C(T), ρ(T) and the melting point (M5)",
    )
    p_melt.add_argument("a", help="element/material id (or first of two to combine)")
    p_melt.add_argument("b", nargs="?", default=None, help="optional second id to combine")
    p_melt.add_argument("--lo", type=float, default=None, help="min T (default auto-bracket)")
    p_melt.add_argument("--hi", type=float, default=None, help="max T (default auto-bracket)")
    p_melt.add_argument("--steps", type=int, default=16, help="number of temperatures")
    p_melt.add_argument("--pressure", type=float, default=0.0, help="pressure dial P (μ offset)")
    p_melt.add_argument("--burn-in", type=int, default=80, help="equilibration sweeps")
    p_melt.add_argument("--samples", type=int, default=50, help="samples per temperature")
    p_melt.add_argument("--sample-every", type=int, default=2, help="sweeps between samples")
    p_melt.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_melt.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_melt.add_argument("--plot", action="store_true", help="save ψ(T)/C(T)/ρ(T) plot")
    p_melt.add_argument("--out", default="out", help="output dir for plots")
    p_melt.set_defaults(func=cmd_melting_sweep)

    p_sc = sub.add_parser(
        "sc-sweep",
        help="sweep T for one material's phase coherence; Y(T), BKT line, and superconducting Tc (M6)",
    )
    p_sc.add_argument("a", help="element/material id (or first of two to combine)")
    p_sc.add_argument("b", nargs="?", default=None, help="optional second id to combine")
    p_sc.add_argument("--lo", type=float, default=0.10, help="min temperature")
    p_sc.add_argument("--hi", type=float, default=1.20, help="max temperature")
    p_sc.add_argument("--steps", type=int, default=18, help="number of temperatures")
    p_sc.add_argument("--burn-in", type=int, default=300, help="equilibration sweeps (XY is slow)")
    p_sc.add_argument("--samples", type=int, default=120, help="samples per temperature")
    p_sc.add_argument("--sample-every", type=int, default=2, help="sweeps between samples")
    p_sc.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_sc.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_sc.add_argument("--plot", action="store_true", help="save Y(T)/BKT-line/Tc plot")
    p_sc.add_argument("--out", default="out", help="output dir for plots")
    p_sc.set_defaults(func=cmd_sc_sweep)

    p_proc = sub.add_parser(
        "process-compare",
        help="run anneal/quench/field-cool on one structure; compare the microstructures",
    )
    p_proc.add_argument("a", help="element/material id (or first of two to combine)")
    p_proc.add_argument("b", nargs="?", default=None, help="optional second id to combine")
    p_proc.add_argument("--budget", type=int, default=120, help="total sweeps per process")
    p_proc.add_argument("--field", type=float, default=0.5, help="field-cool field H")
    p_proc.add_argument(
        "--shape", type=int, nargs="+", default=None, help="lattice shape (default 64 64)"
    )
    p_proc.add_argument("--seed", type=int, default=0, help="UNIVERSE_SEED (default 0)")
    p_proc.add_argument("--plot", action="store_true", help="save spin-microstructure plot")
    p_proc.add_argument("--out", default="out", help="output dir for plots")
    p_proc.set_defaults(func=cmd_process_compare)

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
