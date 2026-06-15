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
  conditions.py  Conditions(T, P, H) — the structure/state separation (M4+); P live (M5)
  thermal.py     thermal-ensemble engine: spin + Curie (M4); occupancy + melting (M5);
                 XY phase coherence + superconducting Tc (M6)
  process.py     synthesis as a trajectory: anneal/quench/field-cool; optional occupancy evolution (M5)
  properties/    pure Lattice -> float extractors (M2+); microstructure.py = process readouts
                 (domain/positional order); cohesion field on the lattice drives melting (M5)
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

### M6 findings (honest superconductivity via phase coherence; the static proxy retired)

M6a replaced the static superconductivity proxy with real phase-coherence physics — de-risked
on the BKT keystone first, then pressure-tested.

- **Superconductivity as XY/BKT phase coherence — a real, parameter-free Tc.** A conducting
  backbone carries an XY phase field (`engine.lattice.xy_sweep`); the **helicity modulus** Υ(T)
  (phase/superconducting stiffness) is the order parameter. A fully-conducting lattice coheres at
  the textbook 2D-XY `T_BKT = 0.893·J` — located as the crossing of Υ(T) with the universal line
  `Υ = (2/π)·T`, no free parameter. Tc **emerges from structure**: a redundant solid backbone
  coheres up to ~0.89, a thin near-percolation filament barely coheres (Tc→0). So the
  k-edge-connectivity work becomes the *coupling input* (a redundant backbone is phase-stiff →
  higher Tc), not the label. (`tests/test_superconductivity.py`; `tools.explorer sc-sweep`.)
- **The BKT subtlety we respected (no-fudge).** Unlike Curie (M4) and melting (M5), the
  **heat-capacity peak does NOT mark Tc** — for 2D XY the C-peak sits *above* `T_BKT` (~1.04 vs
  0.893). The universal C-peak detector that nailed those transitions explicitly *fails* here; we
  prove it in a test and use the helicity-modulus crossing instead. Reporting where the old method
  breaks is the norm, not the exception.
- **Why SC is measured on-demand, not stored on every material (an honest cost call).** XY/BKT
  suffers critical slowing-down: the helicity modulus needs long equilibration (burn ≈ 300+) to
  converge, paid by *every conductor* (~65% of materials) — unlike the fast-equilibrating Ising
  Curie point, paid only by the ~10% ferromagnets. A leaned, cheap sweep gives an
  *under-equilibrated, burn-in-sensitive* number (we measured the swing — an apparent rate of 7.5%
  at burn 120 vs 28% at burn 70, i.e. not converged), which the no-fudge norm forbids storing. So
  the precise Tc is an on-demand instrument (explorer `sc-sweep`, where proper equilibration is
  affordable) validated by the keystone tests; materials store `edge_connectivity` as its
  structural input. A fast XY cluster update (Wolff) would make a stored Tc cheap — a later option.
- **Proxy retired.** The old `superconductor` flag (k-edge-connectivity ≥ k at fixed conditions,
  no real Tc) is gone from the measured properties; `conductance.py` keeps the genuinely-structural
  effective resistance + edge-connectivity (relabelled as backbone redundancy). Supersedes
  `superconductivity-status`.

### M5 findings (crystalline melting falls out, and its honest scope)

M5 made `occupied` thermal and asked melting to *emerge* — pressure-tested the same way as M4.

- **Crystalline melting as an order-disorder transition — proven, parameter-free.** Occupancy is
  a non-conserved **repulsive** lattice gas; at half-filling its low-T state is a checkerboard
  crystal (atoms on one sublattice), and the order parameter is the **staggered density** `ψ`.
  Three things coincide with no fit: the occupancy heat capacity `C(T)=Var(E)/(N·T²)` peaks at the
  textbook 2D point `T_m = 2.269·J0·⟨coh²⟩`, `ψ` collapses at that same `T`, **and the mean
  density stays pinned at ½ across it**. That last column is the whole point of the user's chosen
  model: *positional order is lost at fixed density* — crystalline melting, not the
  sublimation/condensation an *attractive* lattice gas would give (we considered and rejected that
  framing). `tests/test_melting.py`; live: `python -m tools.explorer melting-sweep iron --plot`.
- **Melting tracks bonding — recovers the real ordering.** `T_m ∝ cohesion²` and `cohesion` comes
  from `bond_energy`, so refractory elements (tungsten, carbon) melt *above* soft ones (zinc) —
  the structural analogue of iron>cobalt>nickel for Curie, **measured** not assigned.
- **Pressure is now live (M2 re-validated under the new dynamics).** `P` enters as a
  chemical-potential offset `μ = μ_sym + P`: raising it densifies the lattice gas and drives it
  across the percolation threshold `p_c` — *pressure-tuned percolation*, i.e. the validated M2
  transition reappearing under thermal occupancy. At `P=0` the particle-hole-symmetric `μ` pins
  half-filling (where melting is cleanest). Honest: a solid *below* `T_m` is ≈incompressible
  (density barely moves with `P`); the compressible response lives in the fluid phase / near `T_m`.
- **The process payoff here is structural density — not grain size (a reported negative).** Occupancy
  evolution is wired into the process executor (opt-in; `STANDARD_PROCESS` byte-identical). Cooling
  under a pressure schedule freezes in **different densities** (sinter-dense vs sinter-porous) — a
  structural, non-magnetic path-dependence (`process-compare`). But the cooling-rate → *grain-size*
  signal is **weak**, and we say so: non-conserved checkerboard order heals too fast to freeze
  anti-phase domains at this lattice scale, and the global `ψ` cancels opposing anti-phase domains
  (the same "net `|M|` at `H=0` is the wrong readout" lesson from the process layer). Density is the
  robust readout.
- **Scope (honest).** (1) This is a *continuous* order-disorder transition (the β-brass lattice
  analogue), **not** a first-order liquid-solid with latent heat. (2) `melting_temperature` is the
  order-disorder point of the material's **bond network at commensurate (half) filling** — an
  intrinsic property of the bonding, measured at the filling where crystalline order is defined, not
  at the material's standard-conditions fill. (3) The **stored** `melting_temperature` is gated by
  solidity (a dispersed/gas-like structure has no crystal to melt → `0.0`) and uses a lean sweep, so
  it is honestly *coarse*; the explorer's `melting-sweep` gives the accurate curve. (4) `density(T,P)`
  is `⟨n⟩ × mean atomic mass` (a uniform per-site-mass approximation — full per-site mass on
  thermally-filled sites is later work). `cohesion` is deliberately kept **out of**
  `structural_signature` (it carries no independent entropy — it's derived from `bond_energy`), so
  every M0–M4 seed and stored value stays byte-identical.

- [x] **Process layer (Step 1)** — synthesis as a *trajectory through conditions-space*
      (`engine/process.py`): the same ingredients, **annealed vs. quenched vs. field-cooled**,
      settle into measurably different structures. Generalises `relax()` (a single constant-T
      hold *is* a relax — `STANDARD_PROCESS` reproduces it byte-for-byte); `combine`/`from_element`
      take an optional `process=`. Properties still *measured* from the result (spec §1):
      "wrong process = different material" is a legible microstructure consequence, not a gate.
      See **Process layer findings** below.
- [x] **M5 — Thermal occupancy → crystalline melting.** `occupied` is now a thermal degree of
      freedom: a non-conserved **repulsive lattice gas** (`engine.lattice.occupancy_sweep`, the
      positional twin of the spin kernel) on a new per-site `cohesion` field derived from
      `bond_energy`. Its order-disorder transition *is* crystalline melting. `engine/thermal.py`
      gains the occupancy ensemble (staggered-density order parameter, occupancy heat capacity);
      `melting_temperature` is a stored property (gated by solidity, like Curie). `Conditions.pressure`
      is now **live** (chemical-potential offset → density(P), pressure-tuned percolation), and the
      process executor can evolve occupancy too (opt-in; `STANDARD_PROCESS` stays byte-identical).
      **Keystone proven, parameter-free:** the occupancy `C(T)` peaks at the textbook
      `T_m = 2.269·J0·⟨coh²⟩` exactly where the staggered order collapses, *at fixed density ½*
      (crystalline melting, not sublimation). See **M5 findings** below and
      `python -m tools.explorer melting-sweep iron --plot`.
- [x] **M6a — Honest superconductivity (phase coherence → real Tc).** Replaced the static
      k-edge-connectivity proxy with an **XY model** on the conducting backbone
      (`engine.lattice.xy_sweep`): the helicity modulus Υ(T) (phase stiffness) is the order
      parameter, and the superconducting `Tc` is *measured* as the **BKT universal-line crossing**
      `Υ = (2/π)·T` (`engine.thermal.superconducting_tc`). **Keystone proven, parameter-free:** a
      fully-conducting lattice coheres at the textbook 2D-XY `T_BKT = 0.893·J`, and Tc emerges
      from backbone structure (redundant → high Tc, thin filament → ~0). The static
      `superconductor` proxy is **retired**; SC is measured on demand (explorer `sc-sweep`) — see
      **M6 findings**. `python -m tools.explorer sc-sweep iron copper --plot`.
- [ ] M6b — Thermal conductivity (reuse the Laplacian; `atom_type`-gated phonon/electronic
      carriers → Wiedemann–Franz-like). · M7 — Spectral (band gap). · M8 — Mechanical.
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
# different microstructure. Field-cool builds remanence; the M5 structural section shows
# cooling under pressure freezing in different densities (sinter-dense vs sinter-porous).
python -m tools.explorer process-compare iron chromium --plot

# M5 keystone: crystalline melting. Occupancy order-disorder sweep -> psi(T), C(T), rho(T).
# The C-peak lands on the order collapse AT FIXED DENSITY (melting, not sublimation); a
# uniform-cohesion material melts at the textbook 2.269. Refractory elements melt hotter.
python -m tools.explorer melting-sweep iron --plot
python -m tools.explorer melting-sweep tungsten --pressure 0.0

# See the higher-order percolation transitions (backbone redundancy, the SC coupling input)
python -m tools.explorer connectivity-sweep --plot

# M6: honest superconductivity. Sweep T for a material's phase coherence -> helicity modulus
# Y(T); Tc is where Y crosses the BKT universal line (2/pi)T (NOT the C-peak, which is above
# it). A redundant backbone coheres hotter; a thin one barely coheres. Two ids combine first.
python -m tools.explorer sc-sweep iron copper --plot

# Population view: distributions over many random combinations (spec §7 checkpoint)
python -m tools.explorer distribution --n 500 --plot

# Tests
pytest
```
