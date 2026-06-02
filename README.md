# Emergent Materials Engine

A deterministic crafting/engineering substrate: root **elements** combine into new
**materials**, and each material's physical **properties** (strength, conductivity,
magnetism, superconductivity, …) are **measured from a settled lattice** — never assigned
by a black-box function.

See [materials-engine-spec.md](materials-engine-spec.md) for the full design rationale.
The one design principle: **measure properties from a structure; never assign them from a
function.**

## Layout

```
engine/        deterministic, headless materials engine (never imports game/UI code)
  rng.py         seeded PRNG (SplitMix64) + hashing/mixing
  lattice.py     Lattice type, generation, merge, relax
  elements.py    root element definitions + base lattice generation
  material.py    Material type + combine() pipeline (M1+)
  registry.py    discovered materials + lineage graph (M1+)
  properties/    pure Lattice -> float extractors (M2+)
machines/      worked-example performance equations (M5+); consume Material.properties only
tools/         explorer.py — distributions, single-material inspection, lattice render
tests/         determinism + distribution tests (run in CI from M1)
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Status

Following the milestones in spec §11.

- [x] **M0 — Foundations**: rng, Lattice, root elements, inspect a single element's lattice.
- [x] **M1 — Combination pipeline**: `merge` + `relax` + `combine()` + registry; determinism test passing.
- [x] **M2 — Legible properties + first threshold**: density/mass (`scalar.py`) + percolation/conductivity;
      explorer `distribution` view confirms conductivity is bimodal around the threshold.
- [x] **M3 — Emergent properties + rarity**: magnetism (`ising.py`, structure-derived spin
      coupling via a per-cell `moment` field) + resistance/superconductivity (`conductance.py`,
      Laplacian effective resistance + topological k-edge-connectivity). `magnetism-sweep` shows
      an Ising critical transition (and matches the parameter-free 2D Ising critical coupling);
      `connectivity-sweep` shows the higher-order percolation transitions superconductivity rides;
      the `distribution` view confirms magnetism is bimodal (disordered ↔ aligned) and
      superconductors are a thin ~2–3% tail conditional on conducting. See **M3 findings** below
      for the honest scope of each claim.
- [ ] M4 — Remaining properties (band gap, mechanical).

### M3 findings (what's genuinely emergent, and what isn't)

We pressure-tested M3 rather than just making the numbers look good. The honest summary:

- **Magnetism — genuine emergent transition.** Three independent checks: the sweep's critical
  point matches the parameter-free 2D Ising prediction `m_c = √(0.4407·T)`; weak-coupling
  materials thermalize to ~0 magnetism regardless of initial spins (the disorder is forced by
  physics, not chosen); and magnetism is a *nonlinear threshold* of composition (flat ~0 up to
  ~60% magnetic-material fraction, then jumps) — not a relabel of "iron content". Caveat: the
  `magnetism-sweep` starts from aligned spins, so it measures *order-survival* (the order
  parameter), not spontaneous symmetry-breaking from disorder; in the pipeline the symmetry is
  broken by magnetic elements' inherited spin bias. Physics gates *whether* a material can hold
  order; the bias sets the *direction*.
- **Superconductivity — honest thin tail riding real transitions, with one knob.** An earlier
  version gated superconductivity on an effective-resistance percentile. We rejected that: in a
  site-percolation substrate, conductivity (1/R) is *intrinsically smooth-tailed* — there is no
  second "loss-free phase", so any resistance cut is a disguised probability dial (the rate moved
  smoothly with the threshold). We also tested the obvious substrate fix (conserved Kawasaki
  phase-separation of `occupied` to grow dense domains): it *backfires* — surface tension balls
  the conductor into non-spanning islands, lowering conductivity and spanning. So we reframed
  superconductivity **topologically**: a backbone is loss-free iff it is *k-edge-connected*
  between faces (min-cut ≥ k). k-edge-connectivity percolation is a *genuine* higher-order
  transition with critical density `p_k > p_c` (k=1 reproduces site `p_c ≈ 0.593`), so
  superconductivity = "spans (p_c) AND k-connected (p_k)" is a real threshold-on-a-threshold.
  Remaining honesty: the redundancy level `k` (scaled with lattice width to stay size-robust) is
  the one tunable knob — it selects *which* p_k transition gates the flag. Unlike a resistance
  percentile, each k is a real transition, so the knob picks a transition, not a point on a
  smooth tail. We do **not** claim "no probability dial"; we claim a thin tail with a principled,
  topological criterion.
- [ ] M5 — Machine layer (motor).
- [ ] M6 — Game shell (separate effort).

## Running

```powershell
# Inspect a root element's generated lattice (text + optional plot)
python -m tools.explorer inspect iron

# Combine two materials and inspect the child
python -m tools.explorer combine iron copper

# See the percolation threshold (the core rarity mechanism)
python -m tools.explorer percolation-sweep --plot

# See the magnetism critical transition (Ising order parameter vs coupling)
python -m tools.explorer magnetism-sweep --plot

# See the higher-order percolation transitions superconductivity rides (P(min-cut>=k) vs fill)
python -m tools.explorer connectivity-sweep --plot

# Population view: distributions over many random combinations (spec §7 checkpoint)
python -m tools.explorer distribution --n 500 --plot

# Tests
pytest
```
