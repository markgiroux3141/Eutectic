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
                 (domain/positional order); cohesion drives melting (M5); conductance.py adds
                 thermal conductivity, percolation.py splits charge (metallic) vs solid masks (M6b)
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
- **Thermal conductivity + the diamond divergence (M6b).** Heat has two carriers: *electronic*
  (the charge-carrying electrons also carry heat → Wiedemann–Franz `κ_e = L·T·σ`, which is the
  single-carrier WF content, not a fitted relation) and *phononic* (lattice vibrations through all
  occupied matter, bond conductance ∝ `cohesion²/√mass` — stiff, light lattices win). Charge is
  gated by a per-cell **metallicity** field (from `conduction_tendency`); heat is not. So the
  **diamond divergence falls out**: carbon (non-metallic, stiff, light) has electrical σ=0 yet the
  highest thermal conductivity in the element set — a heat conductor that carries no charge.
  Implementation honesty: this required splitting `percolation`'s **charge mask** (occupied ∧
  metallic — used by electrical conductivity, resistance, edge-connectivity, SC) from the **solid
  mask** (occupied matter — used by `spanning_fraction`/`largest_cluster_fraction`, melting
  solidity, and phonons). The solid measures stay byte-identical (melting untouched); only the
  electrical family is now metallicity-gated, and M2/M3 were re-validated (conductivity stays
  bimodal, magnetism transition intact, determinism green — 126 tests). `metallicity` is kept out
  of `structural_signature` (derived from `conduction_tendency`), so combination seeds are
  unchanged.

### M8 findings (strength + ductility emerge from the bond network; the tradeoff falls out)

M8 measures mechanics from a **central-force spring network** on the settled lattice: occupied
cells are nodes, bonds join nearest- and next-nearest (diagonal) occupied neighbours with spring
stiffness `k_ij = cohesion_i·cohesion_j` (the same `cohesion` that sets melting), and the
stiffness matrix `K` is the elastic analogue of M3's scalar Laplacian. Pressure-tested the same way.

- **Strength = shear modulus; ductility = coordination deficit — both measured, never assigned.**
  Strength is the relaxed strain energy under an imposed face shear (a sparse `spsolve`, the vector
  twin of the effective-resistance solve); ductility is `1 − z̄/z_max`, the density of
  under-coordinated, slip-enabling sites (spec §5.7's "slip planes"). Ductility is *geometric* —
  invariant to scaling every spring constant — so it is a genuine second axis, independent of the
  cohesion magnitude that drives strength.
- **The strength↔ductility anti-correlation FALLS OUT (the §5.7 headline) — not hardcoded.** Across
  the element set the two are anti-correlated (corr ≈ −0.81), and the real ordering is recovered:
  refractory/dense (tungsten, carbon, platinum) come out **strong + brittle**; soft/porous
  (aluminium, lead, mercury, hydrogen) **weak + ductile**. The tradeoff is a consequence of
  coordination (more constraints → higher modulus but fewer slip sites), not a rule we wrote.
  `python -m tools.explorer mechanical`.
- **Keystone — the rigidity transition, parameter-free.** A fully-occupied NN+diagonal lattice is
  over-constrained: rigid, with **exactly zero** floppy modes beyond rigid-body; diluting it drives
  the shear modulus to zero and opens floppy modes. Honest nuance (de-risked first): an NN-only
  central-force *square* lattice is a shear *mechanism* even when full, so the diagonals brace it;
  the resulting **generic shear-rigidity threshold sits at coordination z≈6–7**, above the
  mean-field Maxwell isostatic `z = 2d = 4` (the square lattice's bonds are partially redundant —
  Maxwell counting is the *wrong* cheap proxy here, and we say so). Our ~0.6-fill materials
  therefore live in the **marginally-rigid** regime: moduli are small but cleanly discriminating
  (best resolved at 64²; at smaller lattices many solids dip below the threshold and read 0).
- **The cheap stored ductility tracks the exact (expensive) mechanics — the "coarse stored,
  accurate instrument" split (as for Curie/melting).** The rigorous ductility is the floppy-mode
  fraction (nullity of `K` beyond rigid-body — a dense `eigvalsh`), but it costs 5–13 s/material and
  is near-zero for *every* true solid (it separates solids from liquids, not brittle from ductile).
  The cheap coordination deficit correlates with it (Pearson ≈ 0.93, Spearman ≈ 0.98 over the
  elements) **and** resolves solids better, so it is the stored value; `floppy_fraction` stays an
  explorer/validation instrument.
- **Scope (honest).** (1) Marginally-rigid regime → small absolute moduli; **bond-bending (angular)
  forces** would lower the threshold toward percolation and stiffen things — an optional future
  enrichment, not needed for the milestone. (2) `MODULUS_SCALE` is a fixed display constant (rescales
  every material identically, changes no ordering), not a tuned dial. (3) **Stress σ is the
  conjugate condition** (strain → nonlinear response / fracture); it rides inert for now, to be
  activated like pressure was in M5. (4) Cost: the modulus solve is ~M3-conductance scale (~65 ms),
  gated by solidity like melting; the dense floppy-mode `eigvalsh` is deliberately *not* stored.

### Machine layer findings (the payoff loop — engineering, not new physics)

The machine layer is an *engineering/design* milestone (spec §8), so there is no textbook value
to recover — but the project's spirit still holds: real-ish equations, a legible payoff loop, and
no faked physics. A machine is an **assembly of roles**; performance is **computed from the
assigned materials' measured `.properties`** (`machines/` consumes the engine's outputs only; the
engine never imports it — asserted in a test).

- **Requirements emerge from the equations; they are never hard gates** (a deliberate design fork).
  A non-conducting coil wire gives `R_wire → ∞ → current 0 → torque 0`; a non-ferromagnetic core
  has zero flux; a core run above its Curie point demagnetizes (`flux = magnetism·max(0,1−T/Tc)`).
  The physics punishes a wrong material directly — exactly the One-Principle stance ("measure, don't
  assign") carried up to the assembly. A soft `0..1` per-role **suitability** score (the weighted
  geometric mean of normalized requirement terms) is a *legibility readout only* — it explains
  *why* a slot fits poorly, and never overrides the equations.
- **The payoff loop is real and legible.** A better coil wire (higher `conductivity_continuous`)
  yields strictly more torque *and* higher efficiency; a stronger `shaft` (higher M8 `strength`)
  lifts the torque a weak shaft clips to its yield cap; a higher-`thermal_conductivity` wire sheds
  I²R heat and tolerates more current before burnout (M6b genuinely matters). Performance is a
  **curve over an `OperatingPoint`** (supply voltage, ambient T) — the machine-layer analogue of
  `Conditions`: raise the voltage and torque climbs until it flattens at the I²R **burnout
  ceiling** (`python -m tools.explorer motor iron copper tungsten --plot`).
- **A suite of five machines, one per property family — the framework is genuinely reusable.** The
  `roles.py` framework carries four more assemblies, each rewarding a *different* rare material (so
  the property space has multiple distinct payoffs, not one "best material"): **heat sink**
  (`heatsink.py`, thermal) — dissipation-per-mass `= thermal_conductivity/density`, where **carbon
  wins** (electrically dead, so a useless coil wire, but the M6b *diamond divergence* makes it the
  per-mass cooling champion); **power cable** (`cable.py`, electrical) — transmission efficiency +
  ampacity + sag over a *distance* (tungsten conducts best but is heavy; light+conductive titanium
  wins once weight counts; insulators deliver nothing); **electromagnet** (`electromagnet.py`,
  magnetic) — lift force ∝ I² (quadratic), Curie-gated; **composite armor** (`armor.py`, mechanical)
  — *solves* the M8 strength↔ductility dilemma: protection needs **both** a hard face and a ductile
  backing, so the best plate combines opposite ends of the anti-correlation (one material can't fill
  both — a tungsten backing is too brittle, an aluminium face too weak). The shared I²R-coil math
  lives once in `machines/_electrical.py` (motor, electromagnet, cable).
- **An honest tension we did not fake.** In *our* universe the best in-lattice electrical conductor
  is tungsten (its dense backbone), so tungsten makes a slightly higher-torque coil wire than
  copper even after the brittleness manufacturability penalty (`turns ∝ ductility`) — while the
  *suitability readout* still ranks copper higher (tungsten's low ductility). We report the engine's
  σ-ordering as-is rather than hand-tuning it to the real-world copper-wins story.
- **Calibration constants are fixed display scales, not dials.** `WIRE_R0`, `R_LOAD`, `BURNOUT_K`,
  `TORQUE_K`, `SHAFT_K`, `DUCT_REF` are the motor's design constants (supply circuit, geometry),
  chosen so element-built motors land in legible ranges. Like `mechanical.MODULUS_SCALE` they
  rescale every motor identically and change no ordering — they are not per-material knobs.
- **Scope (honest).** Superconductivity is *not* consumed: a real `Tc` is measured on demand
  (M6a), not stored on `.properties`, and this layer reads stored properties only. The framework is
  intentionally thin (one concrete machine); a second machine would justify generalizing the
  performance side (today only `motor.py` has equations). Roles take one material each (no
  multi-material composites yet).

### M7 (deferred) findings (band gap falsified on our substrate — a reported negative)

M7 (spectral / band gap via `eigh`, spec §5.6) was de-risked and **deferred** — the no-fudge norm
in action, same as the rejected Kawasaki SC fix. The keystone passed (a staggered ±Δ on-site
potential on a *full* periodic square lattice gives `gap = 2Δ` to ~1e-14, parameter-free), and the
honest detector is gap/level-spacing (a metal's raw HOMO-LUMO gap → 0 like 1/N). **But the killer is
dilution:** our materials are ~0.6 fill, and a staggered potential that opens a clean gap at fill
1.00 gives **gap = 0 already at fill 0.95** — 5% vacancies (dangling bonds → mid-gap states) close it
and push the density-of-states at the Fermi level *above* the clean-metal value. A hard band gap is a
property of a near-perfect crystal; a 37%-vacancy network is a defect-dominated mess (physically
correct — cf. amorphous silicon, and silicon metallizing on melting). Both candidate metallicity→
on-site mappings gave `gap = 0` for every element including silicon/carbon. M7 returns only on a
substrate a gap can live on (near-full-fill 2-sublattice crystalline order and/or 3D), ideally as a
*conditions* property (gap closing as T/P opens vacancies). M8 (mechanical) was chosen as the next
milestone instead — the property a diluted percolation network expresses *natively*.

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
- [x] **M6b — Thermal conductivity + the diamond divergence.** Two heat carriers on the
      structure: **electronic** (Wiedemann–Franz, `κ_e = L·T·σ`) and **phononic** (a weighted
      Laplacian over *all* occupied matter, bond conductance ∝ stiffness/mass from `cohesion`/
      `mass`). Charge is now gated by a per-cell **metallicity** field (from `conduction_tendency`):
      only metallic cells carry charge, so the electrical/SC backbone is the *metallic* subset
      while heat flows through all matter. **The diamond divergence falls out:** carbon (non-metallic
      but stiff, light) has electrical σ=0 yet the highest thermal conductivity. `conductance.py`
      gains `thermal_conductivity`; `percolation.py` splits the charge mask (metallic) from the
      solid mask (matter, byte-identical for melting). See **M6 findings**;
      `python -m tools.explorer transport --plot`.
- [x] **M8 — Mechanical (strength + ductility from the bond network).** A central-force spring
      network on the occupied NN+diagonal bonds (stiffness `k_ij = cohesion_i·cohesion_j`);
      **strength** = the shear modulus (a sparse elastic solve, the vector analogue of M3's
      Laplacian), **ductility** = the coordination deficit (the density of slip-enabling
      under-coordinated sites). The strength↔ductility anti-correlation *falls out* (spec §5.7);
      both are stored, gated by solidity. See **M8 findings**; `python -m tools.explorer mechanical`.
- [~] **M7 — Spectral (band gap)** — **deferred** (de-risked and falsified on our ~0.6-fill
      substrate: a hard band gap needs a near-perfect crystal, and 5% vacancies close it
      completely). Returns later on a crystalline/3D substrate or as a conditions-driven property.
      See **M7 (deferred) findings**.
- [x] **Machine layer — the electric-motor worked example (spec §8).** The payoff loop: a
      machine is an *assembly* of **roles** (`machines/roles.py`: a thin `Role`/`Requirement`/
      `Blueprint` framework), and **performance is computed from the assigned materials' measured
      properties** by real-ish equations (`machines/motor.py`) — never assigned. The motor's
      `core`/`coil_wire`/`shaft` consume `magnetism`+`curie_temperature` (M3/M4),
      `conductivity_continuous`+`melting_temperature`+`thermal_conductivity`+`ductility`
      (M5/M6b/M8), and `strength` (M8) respectively — so it exercises nearly every milestone.
      Requirements are **emergent, not gated** (a non-conducting wire → dead motor; an over-Curie
      core demagnetizes); a soft per-role suitability score is a legibility readout only. The
      engine never imports `machines` (asserted in `tests/test_motor.py`). See **Machine layer
      findings**; `python -m tools.explorer motor iron copper tungsten --plot`.
- [x] **Machine suite — four more machines, one per property family.** Proving the `roles.py`
      framework reuses: **heat sink** (thermal — carbon wins, the diamond divergence), **power
      cable** (electrical — loss/ampacity/sag over distance), **electromagnet** (magnetic — lift
      ∝ I²), **composite armor** (mechanical — *solves* the M8 strength↔ductility dilemma via a
      hard face + ductile backing). Shared coil math in `machines/_electrical.py`. See **Machine
      layer findings**; `tests/test_machines.py`.
- [ ] Game shell follows (spec §11 M6 — inventory, crafting UI, building, progression).

### Chemistry layer (chemistry-engine-spec — the layer *below* materials)

A separate ladder (C0..C5) building atoms → bonds → compounds → reactions, whose product
crystals feed the existing extractors unchanged ("chemistry generates structure; the materials
engine measures it"). See `chemistry-engine-spec.md`.

- [x] **C0 — Atom model** (`chemistry/atoms.py`). Root atoms carry distilled descriptors;
      everything else is **derived parameter-free from Z**. **Keystone proven:** valence/bonding
      capacity from the octet-deficit rule `min(valence_e, octet−valence_e)` over a Madelung-filled
      configuration reproduces H→1, C→4 (the promotion case, no special-casing), N→3, O→2, Na→+1,
      Mg→+2, Cl→−1, noble gases→0 — and lone pairs (O→2, N→1) for VSEPR next. **Bonus keystone:** a
      Slater Z_eff model makes `EN ∝ Z_eff/n²`, `radius ∝ n²/Z_eff` reproduce every periodic trend.
      Honest caveats (flagged in-module): the derived EN is *ordinal, not Pauling-calibrated* (so
      C1b reads authored Pauling values); transition-metal valence reads a single common 2 (no
      d-shell multivalence yet); plain Madelung misses the Cr/Cu half-shell anomalies. See
      `tests/test_valence.py`; `python -m tools.chem_explorer inspect-atom O`.
- [x] **C1 — Bonding & molecules** (`chemistry/bonding.py`, `chemistry/orbitals.py`,
      `chemistry/molecule.py`). **(a) geometry** from steric number + lone pairs; **(b) bond
      character** (ΔEN thresholds + a metallic ceiling), order, and a distilled energy; **(c)
      formation** = minimum-energy bonding over candidate ratios. **Keystones proven:** geometry —
      CH₄ 109.5° sp³, H₂O bent 104.5°, CO₂ linear 180°, NH₃ pyramidal 107° (one shared
      2.5°/lone-pair constant); bond type — NaCl ionic, Cl₂ covalent, Cu metallic, with
      triple>double>single energy ordering; stoichiometry — NaCl 1:1, MgCl₂ 1:2, H₂O 2:1, CO₂ 1:2,
      Al₂O₃ 2:3 *fall out* parameter-free, and the satisfied ratio is the genuine energy minimum
      (tested, not asserted); noble gases refuse. **Covalent energy = the Pauling model** (real
      kJ/mol): `√(E_AA·E_BB) + k·(Δχ)²` × a sublinear bond-order factor, recalibrated after C4 to
      fix the C3 enthalpy signs (see below) — it predicts held-out heteronuclear bonds at r≈0.97
      (the old `EN_avg/(r)` form managed r≈0.32). Honest caveats (in-module): O=O/N≡N are built on
      the anomalously-weak O–O/N–N singles so multiple-bond magnitudes on O/N are underestimated
      (signs hold); ionic energy is the single ion-pair Coulomb term and metallic its own — neither
      is cross-calibrated to the covalent kJ/mol scale (Madelung lattice energy is future work);
      covalent formation is single-central (multi-center molecules like C₂H₆ need the general
      search, spec §19.2). See `tests/test_{geometry,bond_type,stoichiometry}.py`; `python -m
      tools.chem_explorer build-molecule H O`.
- [x] **C2 — Compound → lattice + integration** (`chemistry/crystal.py`). The bridge: a compound's
      crystal is an `engine.lattice.Lattice` whose **per-cell fields are set by the bonding**, so
      the *existing* extractors measure it **unchanged**. Packing from bond character (ionic
      rock-salt / metallic close-packed / covalent network); `metallicity` from character (→ NaCl
      insulates, Cu conducts); `moment` from **localized (d/f) unpaired electrons** (→ Fe magnetic,
      Cu/diamond/NaCl not — s/p electrons are quenched by bonding); `cohesion`/`mass` drive
      melting/strength/phonon-κ. **Keystone proven:** NaCl insulates, Cu conducts, diamond is an
      electrical insulator that beats metals on phonon heat conduction (the M6b divergence), Fe
      magnetises from unpaired electrons. **Affinity-derivation finding (the make-or-break, reported
      not buried):** deriving the *existing elements'* authored affinities from descriptors
      reproduces conduction moderately (+0.66) but **magnetism does not derive** — ferromagnetism is
      the Stoner criterion, not unpaired count (Cr has Fe's unpaired count but is antiferromagnetic),
      and cohesion (many-body) anti-correlates. So `engine/elements.py` is **untouched** (authored
      affinities kept as reference data) → **M0–M8 stay byte-identical green**; chemistry-derived
      fields apply only to new compounds. See `tests/test_crystal_properties.py`; `python -m
      tools.chem_explorer measure-compound Na Cl`. (M7 band-gap revival deferred — needs a 3D
      network substrate.)
- [x] **C3 — Reaction thermodynamics** (`chemistry/reaction.py`, `chemistry/conditions.py`).
      Feasibility *measured* from energetics: **ΔH** by Hess's law over the C1 bond energies
      (`Σ E(broken) − Σ E(formed)`); **ΔS** from a distilled phase-entropy estimate (gas ≫ liquid >
      solid, so more gas moles → higher S); **ΔG = ΔH − T·ΔS** with the **mass-action / Le Chatelier**
      term folded in (`ΔG = ΔG° + R·T·ln Q`, gas activity ∝ P); **K = exp(−ΔG/RT)** (so K=1 at the
      ΔG=0 boundary). `ChemConditions(T, P, concentration, catalysts)` extends `engine.Conditions`.
      **Keystone proven:** exergonic reactions proceed at standard T (recombination `2A→A₂`, K>1),
      endergonic ones don't (`A₂→2A`, K<1) — **until a temperature threshold flips ΔG's sign** at
      `T*=ΔH/ΔS`, a *genuine* single sign-crossing (ΔG monotone in T, the boolean flips at T*, not a
      smooth dial); the thresholds **order by bond strength** (weaker bond dissociates cooler,
      emergent and parameter-free); Le Chatelier reproduces — raising P on a gas-producing reaction
      raises T* (suppresses dissociation), concentration likewise.
      **The C3↔C1 fix (the honest story, start to finish):** originally Hess's law over the C1
      `order·EN_avg/(r)` bond energy gave the **wrong ΔH sign** for every reaction that *breaks* a
      multiple bond (`2H₂+O₂→2H₂O` read +68 endothermic; Haber, `H₂+Cl₂` likewise). We refused to
      tune a bond-order knob to force a pass; a de-risk then showed the real culprit wasn't the bond
      order at all but the **functional form** (it tracked real single-bond energies at only r≈0.32).
      The honest fix was to replace it with the **Pauling model** (`chemistry/bonding.py`),
      *calibrated from independent single-bond data* (homonuclear energies + the published ionic
      constant + a σ/π ratio from clean C-series) — **never fitted to a reaction**. Held-out
      heteronuclear bonds then predict at r≈0.97, and the correct combustion/synthesis signs fall out
      as a **consequence** (`2H₂+O₂` now −453 kJ/mol vs real −482, spontaneous, K~1e20). Residual
      honest limit, pinned as a test: O=O/N≡N are still underestimated (anomalously-weak O–O/N–N
      singles), so the model now *under*-orders O₂ dissociation — magnitudes off on O/N multiple
      bonds, signs right. See `tests/test_thermo.py`; `python -m tools.chem_explorer react "O = 2 ~O" -T 13` and
      `condition-sweep "Cl = 2 ~Cl"`.
- [x] **C4 — Kinetics & catalysts** (`chemistry/kinetics.py`). Feasibility (ΔG) says a reaction
      *can* go; **rate** says whether it does. **Arrhenius rate** `A·exp(−Ea/RT)` (the same
      Boltzmann/exponential form as Metropolis acceptance); **activation energy Ea** from a
      transition-state estimate — a fixed fraction of the energy of the **reactant** bonds that must
      break (so radical recombination `2A→A₂`, with no bonds to break, is barrierless, as in
      reality); **catalysts** open a lower-barrier path (rate only, never ΔG/K, not consumed).
      **Keystone proven:** the favourable-but-trapped case — C+O₂→CO₂ is exergonic (ΔG<0) yet its
      rate is negligible cold (1e-8) and climbs ~8 orders of magnitude when heated (trapped → ignited);
      a catalyst halving Ea gives a ~500× rate boost and **halves** the temperature needed for a given
      rate, with **ΔG identical** with/without it. (H₂+O₂ can't serve as the trapped example here — its
      ΔG sign is wrong in our model, the C3 finding — so the keystone uses C+O₂→CO₂.) **Honest caveats
      (flagged):** the "ignition threshold" is **not a real transition** — rate(T) is a smooth
      exponential; real ignition is *thermal runaway* (feedback) we don't model, so we report rate
      ratios / temperature-for-a-rate and refuse to dress a rate>cutoff boolean as a phase transition
      (nudging the cutoff slides it smoothly — the disguised-dial tell). Absolute rates are
      uncalibrated; the T-sensitivity and the catalyst's fixed barrier-shift are what's emergent. See
      `tests/test_kinetics.py`; `python -m tools.chem_explorer kinetics "~C + O = C.O" --catalyst Pt`.
- [x] **C5 — Reaction network / tech tree** (`chemistry/network.py`). Species + reactions form a
      directed graph; `reachable(inventory, conditions)` is its **transitive closure** — fire every
      *live* reaction (ΔG<0 *and* fast enough at the available T given any catalysts), add products,
      repeat to a fixed point. Prerequisites, condition-gating, and rarity **emerge from the graph**,
      not authored progression. **Keystone proven (anchored on the genuine threshold):** from
      {H₂,O₂,N₂,Cl₂}, the free radicals are unreachable cold and **unlock as T crosses each diatomic's
      dissociation T\*, in emergent bond-strength order** (Cl 4.03 < O 4.62 < H 7.27 < N 7.79 — the C3
      ordering, parameter-free); the unlock temperature *equals* the C3 ΔG sign-crossing, and the
      reachable set grows monotonically (and **sheds** H₂O above its own crossover ~7.55 — correct
      entropy). **Multi-step prerequisite** (`O₂→2O` then `O+H₂→H₂O`): the target is unreachable below
      atomic O's gate and only forms after the intermediate is made (overlap window T∈[4.62, 6.09]).
      **Catalyst gate:** Fe flips NH₃ from unreachable to reachable inside its thermodynamic window
      (T<2.10), with ΔG identical either way. **Emergent rarity:** `NO` sits in the graph but is
      reachable at **no** temperature — direct `N₂+O₂→2NO` is endothermic with Δn_gas=0 (ΔG>0 always)
      and the radical window (<5.4) is disjoint from when atomic N exists (>7.8); a closed
      thermodynamic window, nothing tagged "rare". **Honest watch-outs (flagged, per no-fudge):**
      (1) the "fast enough" gate is a **soft rate dial** (`DEFAULT_RATE_CUTOFF`) — so the headline
      keystone gates on **ΔG only** (`require_rate=False`), and we *prove* the distinction with a
      disguised-dial test: the ΔG-gated radical onset is **pinned** at T\* as the cutoff is nudged
      over orders of magnitude, while the rate-gated unlock (HCl, Haber) **slides smoothly** — we
      report which gate is which rather than dressing the dial as a transition. (2) A *unique*
      radical-only synthesis to a stable compound generally **doesn't exist** in this substrate
      (recombination shrinks gas moles, turning endergonic below the dissociation T\* unless the
      product bond beats the diatomic) — a structural negative, reported. (3) Every reaction in the
      demo network is gas-phase covalent, so no live/dead decision touches the uncalibrated
      ionic/metallic scale. See `tests/test_network.py`; `python -m tools.chem_explorer tech-tree -T 5`
      and `tech-tree -T 1.8 --catalyst Fe`.
- [x] **Synthesis as a trajectory** (`chemistry/synthesis.py`, spec §13) — the chemistry analog of
      the materials **process layer** (`engine/process.py`). Where C5 asks "what's reachable at *one*
      fixed condition?", a `Route` is an ordered schedule of `ChemConditions` set-points, and
      `synthesize` threads the **inventory** through them — folding the C5 reachability closure and
      **accumulating** (a single-stage route reproduces a C5 `reachable` exactly, so it's a strict
      generalisation). **Keystone proven:** a target reachable at **no static temperature** falls out
      of a trajectory, because the step that *makes* an intermediate and the step that *consumes* it
      want different temperatures. `NO` is the case (C5's locked-out target): atomic N exists only
      above T\*≈7.8, but `N+O→NO` is exergonic only below T\*≈5.4 — disjoint windows. A
      **heat(8)→quench(1)** route makes it (heat to liberate radicals, quench to capture NO — real
      radical chemistry: NO forms in combustion/lightning and freezes in on cooling), and **order is
      load-bearing**: cool-then-heat ends hot and fails. Both gates are *genuine ΔG sign-crossings*
      (quenching just above the recombination T\* captures nothing; a hot stage below the dissociation
      T\* liberates no radicals — both pinned). **Honest scope:** this is *cumulative attainability*
      (what the route can yield, capture-at-the-favourable-stage), not a concentration/yield model —
      we don't model back-reaction if the product isn't isolated (spec §20: qualitative emergence, not
      yields); stages are discrete holds (a ramp = more stages). See `tests/test_synthesis.py`;
      `python -m tools.chem_explorer synthesize`.

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

# M6a: honest superconductivity. Sweep T for a material's phase coherence -> helicity modulus
# Y(T); Tc is where Y crosses the BKT universal line (2/pi)T (NOT the C-peak, which is above
# it). A redundant backbone coheres hotter; a thin one barely coheres. Two ids combine first.
python -m tools.explorer sc-sweep iron copper --plot

# M6b: electrical vs thermal conductivity across elements -> the diamond divergence (carbon
# conducts heat superbly but carries no charge; metals carry both via Wiedemann-Franz).
python -m tools.explorer transport --plot

# M8: strength vs ductility across elements -> the anti-correlation (spec 5.7). Refractory/dense
# come out strong+brittle, soft/porous weak+ductile; both measured from the spring network.
python -m tools.explorer mechanical --plot

# Machine layer: build an electric motor from 3 materials (core/coil_wire/shaft) and see its
# performance computed from their measured properties. --plot sweeps voltage -> torque climbs
# into the I^2R burnout ceiling. A better wire/core/shaft visibly builds a better motor.
python -m tools.explorer motor iron copper tungsten --plot
python -m tools.explorer motor iron copper copper --voltage 2.0   # weak shaft clips the torque

# More machines (one per property family). Heat sink: carbon wins cooling-per-mass (the diamond
# divergence). Cable: tungsten conducts best but is heavy; titanium is the light sweet spot.
# Electromagnet: lift ~ I^2, Curie-gated. Armor: a composite solves the strength<->ductility dilemma.
python -m tools.explorer heatsink              # all elements, sorted by cooling-per-mass
python -m tools.explorer cable                 # all elements, sorted by transmission efficiency
python -m tools.explorer electromagnet iron copper
python -m tools.explorer armor tungsten aluminium   # hard face + ductile backing

# Population view: distributions over many random combinations (spec §7 checkpoint)
python -m tools.explorer distribution --n 500 --plot

# Tests
pytest
```
