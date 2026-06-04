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
  conditions.py  Conditions(T, P, H) — the structure/state separation (M4+)
  thermal.py     thermal-ensemble engine: observables at conditions, Curie point (M4+)
  process.py     synthesis as a trajectory: anneal/quench/field-cool schedules (process layer)
  properties/    pure Lattice -> float extractors (M2+); microstructure.py = process readouts
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
- [x] **M4 — Conditions & thermodynamic state** (the enabler; supersedes the old spec-§11 M4,
      see `docs/conditions-and-properties.md`): `Conditions(T, P, H)` + `engine/thermal.py`, the
      thermal-ensemble engine. Properties become **ensemble observables of a structure at
      conditions** (`measure(structure, conditions)`), not frozen snapshot numbers. Curie
      temperature is now a first-class, *measured* property (`temperature-sweep` explorer view).
      **Keystone proven:** heat capacity `C(T) = Var(E)/(N·T²)` peaks at the textbook 2D-Ising
      `Tc ≈ 2.269` (no free parameters) exactly where the order parameter collapses. See
      **M4 findings** below for the honest scope (broadened dilute transitions; coarse stored Tc).

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

### M4 findings (the keystone, and the honest scope of a per-material Tc)

M4 separated **structure** (the material — geometry + couplings) from **state** (a thermal
configuration *at* conditions). A property is now an *ensemble observable*
`measure(structure, conditions)`, not a frozen snapshot. We pressure-tested it the same way:

- **Heat capacity as a universal transition detector — proven, parameter-free.** A
  fully-occupied, unit-moment lattice is plain 2D Ising. Its heat capacity `C(T)=Var(E)/(N·T²)`
  peaks at the textbook `Tc = 2/ln(1+√2) ≈ 2.269` (no fit, no knob), and the order parameter
  `⟨|M|⟩` collapses at that *same* temperature. That C-peak ↔ M-collapse ↔ textbook three-way
  coincidence is the proof the architecture works — we don't hand-define the Curie point; the
  fluctuation peak *is* it. (`tests/test_thermal.py`; see it live with
  `python -m tools.explorer temperature-sweep iron --plot`.)
- **Real materials have *broadened* transitions — honest, and physically correct.** The sharp
  clean peak is the full lattice. Site-diluted real materials (fill ~0.6, near percolation)
  show a *smeared* transition: high-quality C-peaks land at ~1.8/1.6/1.5 for iron/cobalt/nickel
  — a real ordering (correctly iron > cobalt > nickel), but a broad one. So a single
  per-material `Tc` is inherently an approximate midpoint of a smeared transition, not a razor
  edge. We say so rather than faking a crisp number.
- **Stored Tc is coarse by design; the explorer is the accurate instrument.** Storing Tc on
  every Material (a temperature sweep per material) is expensive, so it is **gated** by the
  reference order parameter — a material has a Curie point only if it's ferromagnetic at
  standard conditions, so the non-magnetic majority cost nothing (Tc=0). The stored value uses
  a lean sweep and is therefore coarse (~±0.3 vs the high-quality C-peak); `temperature-sweep`
  with high sampling gives the precise curve.
- **Rejected a cheaper locator (reported, not buried).** We tried locating Tc from the
  order-parameter's steepest descent (cheaper, smoother). It is *biased low* (~1.3 for all
  three ferromagnets — it catches the saturation roll-off, not the transition) and disagrees
  with the C-peak by up to 0.5. So the heat-capacity peak stays the canonical detector.

### Process layer findings (synthesis as a trajectory, Step 1)

Real materials are process-sensitive (neodymium magnets, quench-hardened steel): the *same*
ingredients yield different materials depending on temperature schedule, field, and sequence.
We capture this without new machinery — it's the existing relaxation kernel
(`engine.lattice.metropolis_sweep`) carried along a *schedule* of conditions instead of one
fixed-T settle. A `Process` is an ordered list of `Stage`s; `run_process` threads the live
spin state through them. The One Principle holds: the process shapes the *structure*;
properties are still measured (spec §1).

- **Path-dependence is real and monotonic (de-risked, then encoded as tests).** Slower
  cooling reaches a **lower-energy, larger-domain, fewer-domain-wall** structure than a quench
  — clean and monotonic across seeds (`tests/test_process.py`). Try it:
  `python -m tools.explorer process-compare iron chromium`.
- **The readout matters — `|M|` at zero field is the *wrong* one.** At `H=0` the 2D Ising
  picks a random domain sign and the net cancels, so annealed and quenched magnets both read
  `|M|≈0.1`. The cooling-rate signal lives in **microstructure** (`engine/properties/microstructure.py`:
  `domain_fraction`, `domain_wall_density`) and, under a field, in **remanence**. We measure
  those, not just net moment.
- **Field-cooling is the first dramatic "process payoff."** Cooling under a field then removing
  it leaves a near-single-domain magnet (`remanence ≈ 0.97`) versus `~0.1` for zero-field
  cooling — the neodymium preview. Honest caveat: with isotropic Ising this is *kinetic*
  remanence (low-T freezing), not *coercive* remanence; true hard-magnet behaviour needs the
  anisotropy + hysteresis block (a later, separately de-risked milestone).
- **Scope (honest).** At the spin level we get **magnetic-microstructure** path-dependence only
  (occupancy is frozen, so density/percolation don't move). The dramatic multi-property version
  (martensite, glass-vs-crystal, grain growth) arrives when **M5** makes `occupied` thermal —
  which plugs into this same executor. Also: a process is **not** part of a material's identity
  yet (id derives from lineage only), so registry-caching by process is a follow-up.

- [x] **Process layer (Step 1)** — synthesis as a *trajectory through conditions-space*
      (`engine/process.py`): the same ingredients, **annealed vs. quenched vs. field-cooled**,
      settle into measurably different structures. Generalises `relax()` (a single constant-T
      hold *is* a relax — `STANDARD_PROCESS` reproduces it byte-for-byte); `combine`/`from_element`
      take an optional `process=`. Properties still *measured* from the result (spec §1):
      "wrong process = different material" is a legible microstructure consequence, not a gate.
      See **Process layer findings** below.
- [ ] M5 — Thermal occupancy: make `occupied` thermal → melting, thermal expansion,
      density(T,P), pressure-tuned conductivity. **Plugs into the same process executor** — the
      moment atomic structure becomes dynamic, anneal/quench gain density/grain effects.
      Re-validate M2/M3 under the new dynamics.
- [ ] M6 — Transport + honest superconductivity (phase-coherence XY → real Tc; retire the
      static SC proxy). · M7 — Spectral (band gap). · M8 — Mechanical (strength/ductility).
- [ ] Machine layer (motor) + game shell follow (spec §8, §11).

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

# M4 keystone: sweep temperature for a material; M(T), C(T), and the Curie point
# (heat-capacity peak coincides with the order-parameter collapse). One id, or two to combine.
python -m tools.explorer temperature-sweep iron --plot
python -m tools.explorer temperature-sweep iron copper --hi 3.0

# Process layer: same ingredients, different synthesis (anneal/quench/field-cool) ->
# different microstructure. Field-cool builds remanence (the "process payoff").
python -m tools.explorer process-compare iron chromium --plot

# See the higher-order percolation transitions superconductivity rides (P(min-cut>=k) vs fill)
python -m tools.explorer connectivity-sweep --plot

# Population view: distributions over many random combinations (spec §7 checkpoint)
python -m tools.explorer distribution --n 500 --plot

# Tests
pytest
```
