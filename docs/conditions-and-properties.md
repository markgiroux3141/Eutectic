# Conditions & Properties — the architecture for condition-dependent materials

**Status:** adopted design direction (post-M3). Revises and extends the milestone path in
`materials-engine-spec.md` §11. Read this *with* the spec (especially §1, §5) and the
README "M3 findings" section.

This document defines how the engine grows from "properties measured once on a frozen
lattice" to "properties that vary with **temperature, pressure, magnetic field, …** — and
where those conditions *fall out of the substrate* rather than being bolted on."

---

## 1. The principle, unchanged

> **Measure properties from a structure. Never assign them from a function.** (spec §1)

Everything below is in service of that. A *condition* is not a new dial we invent — it is a
term added to the lattice's energy function that couples to a degree of freedom we already
have. A property is not a stored number — it is a measurement of the lattice's behaviour
*at* a condition.

---

## 2. The layer down: separate **structure** from **state**

Today (`engine/material.py`) a material is created by `merge → relax → measure`, and the
measured numbers are frozen onto the `Material`. This **conflates two things**:

1. The material's **permanent structure** — geometry + coupling constants (the lattice
   fields `occupied`, `atom_type`, `spin`, `mass`, `moment`, and the couplings derived from
   them). *This is the material.*
2. The transient **thermodynamic state** — the actual spin/occupancy configuration *right
   now, at this temperature*. We currently relax once (one state at the fixed
   `RELAX_TEMPERATURE`) and treat that single snapshot as if it were the material.

**The architecture:** the material *is* its structure (a Hamiltonian). A property is an
**ensemble observable of that structure at given conditions**. This is just statistical
mechanics, and it is the same "measure from structure" principle — we measure the
structure's *behaviour under conditions*, not one frozen snapshot.

Consequences:
- Temperature *falls out*: it is the ensemble temperature (already a parameter of
  `relax()`).
- Each property becomes (conceptually) `measure(structure, conditions) -> value`.
- For legibility we still store a **reference snapshot** at *standard conditions*
  (`T0, P0, H0=0`) on the `Material`; full curves are measured on demand.

---

## 3. Conditions are thermodynamic conjugate pairs

Physics tells us exactly which conditions are "natural": each is conjugate to a response we
can already measure. This is the map of what falls out cheaply vs. what is real new work.

| Condition (dial)        | couples to →            | Response field            | Naturalness |
|-------------------------|-------------------------|---------------------------|-------------|
| **Temperature T**       | entropy / energy        | all thermal fields        | free — already in `relax()` |
| **Magnetic field H**    | magnetization           | `spin` (−H·Σspin term)    | trivial |
| **Pressure P** / μ      | amount of matter        | `occupied` (density)      | one real change (make `occupied` thermal) |
| **Voltage / current**   | charge transport        | conducting graph          | already implicit in effective resistance |
| **Mechanical stress σ** | strain                  | bond network              | real work (rigidity model) |
| **Species potential (pH)** | amount of a species  | `atom_type` chemistry     | hack-risk — needs a chemistry layer |

**Core dial set: `T`, `P`, `H`** (voltage stays implicit in resistance). Stress and pH are
explicitly *out of core* — they are later, separate decisions, flagged so we don't hack.

---

## 4. Property roadmap (tiered by how naturally it emerges)

Three measurement "engines" already exist or are cheap: **reductions** (counts/means),
**graph/transport** (percolation, Laplacian, min-cut), **thermal ensemble** (Metropolis at
T, + conjugate fields). A fourth, **spectral** (eigendecomposition), is an optional new
engine.

**Tier 0 — already have / free**
- density, atomic mass, composition (reductions)
- magnetism **+ Curie temperature Tc** (spin ensemble vs T — *validated feasible*, see §6)
- electrical conductivity / effective resistance (graph Laplacian)

**Tier 1 — cheap, reuse existing machinery**
- **thermal conductivity** — the *same* graph as electrical conductivity ("heat flows"
  instead of "charge"); `atom_type`-gated heat vs charge carriers gives Wiedemann–Franz-like
  relations for free.
- **heat capacity** — energy fluctuations `Var(E)/T²` of the thermal ensemble. Free once
  anything is thermal, **and it peaks at every transition** → a *universal transition
  detector* (we don't hand-define Curie/melting/SC points; the peaks *are* them).
- **magnetic susceptibility + hysteresis / coercivity** — add field H to the Ising engine.
- **melting point + thermal expansion + density(T,P)** — make `occupied` a thermal field
  (lattice gas + pressure/μ term). Positional order breaks above a critical T = melting.

**Tier 2 — principled, but a genuinely new engine**
- **superconductivity with a *real* Tc — ✅ DONE (M6a).** Modelled as a **phase-coherence (XY)
  ordering** on the conducting backbone (`engine.lattice.xy_sweep`): the helicity modulus Υ(T)
  is the order parameter, and Tc is the **BKT universal-line crossing** `Υ=(2/π)·T`
  (`engine.thermal.superconducting_tc`). A fully-conducting lattice recovers the textbook
  `T_BKT=0.893·J` parameter-free; Tc emerges from backbone rigidity (k-edge-connectivity is the
  *coupling input*, not the label). The static proxy is **retired**. Honest scope: the
  heat-capacity peak does NOT mark a BKT Tc (it sits above it — the universal C-detector fails
  here, and we use the stiffness crossing); and because XY/BKT equilibrates slowly for every
  conductor, the precise Tc is **measured on demand** (explorer `sc-sweep`), not stored per
  material. Supersedes `superconductivity-status` / README "M6 findings".
- **band gap → conductor / semiconductor / insulator** — eigenvalues of a lattice
  Hamiltonian (spec §5.6); gap shrinks with T/P. New engine (`numpy.linalg.eigh`).
- **strength / ductility / fracture** — rigidity of the bond graph; ductility = low-energy
  shear rearrangements; strength↔ductility anti-correlation should fall out (spec §5.7).
  Stress σ is its conjugate condition.

**Tier 3 — does NOT fall out cleanly (defer / separate layer; do not hack into core)**
- **radioactivity** — which `atom_type`s are unstable is *authored*, not measured (real
  radioactivity is nuclear, which we don't model). It can *drive* emergent dynamics (decay
  heat → raises T, transmutation → material changes over time), but the instability itself
  is an input. Honest framing: "authored cause, emergent consequences."
- **pH / corrosion / reactivity** — needs a chemistry layer on `atom_type` (reaction rules +
  species chemical potential). Real new machinery; highest hack-risk.

---

## 5. The single highest-leverage move ✅ DONE (M5)

Making **`occupied` a thermal degree of freedom** (non-conserved vacancy dynamics with a
pressure/chemical-potential term — *simpler* than the conserved Kawasaki dynamics we tested
and rejected for SC) unlocks, from one change: **melting, thermal expansion, density(T,P),
pressure-tuned conductivity/superconductivity, and heat capacity as a transition detector.**
It also touches the validated M2/M3 percolation/density, so it must re-validate them.

**Done in M5** (see §7 and README **M5 findings**): occupancy is a *repulsive* lattice gas
whose order-disorder transition is crystalline melting (parameter-free `T_m` at the textbook
2D point, at fixed density); pressure is live (density(P), pressure-tuned percolation
re-validates M2). The crystalline (fixed-density) framing was a deliberate choice over the
attractive-lattice-gas sublimation it would otherwise have been.

---

## 6. Validation discipline (non-negotiable — this is how we avoid fudging)

M3 taught us the failure mode: a property can *look* emergent while actually being a
disguised probability dial or a relabelled input. Every emergent claim must be
pressure-tested:

- **Recover known physics with no free parameters.** The magnetism transition matched the
  2D Ising critical coupling `m_c = √(0.4407·T)`; k=1 connectivity reproduced site
  `p_c ≈ 0.593`. New transitions must hit their textbook values where one exists.
- **Heat capacity peaks must coincide with the transitions** the order parameters show.
- **Hunt for disguised dials.** If a "transition" is really a percentile cut on a smooth
  tail, its rate moves smoothly as you nudge the threshold (the M3 SC red flag). A genuine
  transition is robust to threshold placement.
- **Distinguish the axis.** A sharp change vs *fill* is a density transition; vs
  *temperature* is a thermal transition. Don't claim a Tc from a fill sweep.
- **Report negative results.** The Kawasaki "fix" was falsified and we said so. That is the
  norm, not the exception.

**Feasibility already shown:** sweeping relaxation T on real materials produced clean
per-material Curie points with a physically sensible spread (ferromagnets Tc≈1.7–1.8,
diluted combos ≈1.2–1.5, non-magnetic / sub-threshold → no Tc), and Tc depended on
*structure* (connectivity), not just magnetic content. Temperature-as-measurement-axis
works.

---

## 7. Revised milestone sequencing (supersedes spec §11 M4+)

Do these in order; the conditions layer gates everything. Each is shippable, tested, and
demo'd in the explorer before moving on.

- **M4 — Conditions & thermodynamic state (the enabler). ✅ DONE.** `engine/conditions.py`
  introduces `Conditions(T, P, H)` (P inert until M5, flagged honestly); `engine/thermal.py`
  is the ensemble engine — it measures observables (`⟨|M|⟩`, energy, **heat capacity
  `C = Var(E)/(N·T²)`**) from a structure's spin Hamiltonian at conditions, reusing one
  shared deterministic Metropolis kernel (`engine.lattice.metropolis_sweep`) for both M3
  settling and M4 measurement. Field `H` is wired as the conjugate `-H·Σ(moment·spin)` term.
  `curie_temperature` is stored on every Material (gated by reference order so the
  non-magnetic majority cost nothing). Explorer gains `temperature-sweep`. Determinism
  preserved; M2/M3 stay green (91→92 tests).
  - **Keystone validation (rigorous, parameter-free):** a fully-occupied unit-moment lattice
    is plain 2D Ising → `C(T)` peaks at the textbook `Tc = 2/ln(1+√2) ≈ 2.269`, and the
    order parameter `⟨|M|⟩` collapses at that exact temperature. The C-peak ↔ M-collapse ↔
    textbook three-way agreement (`tests/test_thermal.py`) is the proof the
    measure-at-conditions architecture works. **We did not proceed past this.**
  - **Honest scope (the no-fudge findings):** (1) the *clean, sharp* transition is the full
    lattice; **real site-diluted materials show a physically broadened transition** (dilution
    near percolation smears it — iron/cobalt/nickel high-quality C-peaks land at ~1.8/1.6/1.5,
    a real ordering but a *broad* one). So a single per-material `Tc` is inherently an
    approximate midpoint. (2) The **stored** Tc uses a lean sweep (cost-gated) and is
    therefore *coarse* (~±0.3 vs the high-quality C-peak); the explorer's `temperature-sweep`
    with high sampling gives the accurate curve. (3) We checked an order-parameter
    steepest-descent locator as a cheaper Tc and **rejected it** — it is biased low (~1.3 for
    all three ferromagnets, catching the saturation roll-off, not the fluctuation peak), so
    the heat-capacity peak remains the canonical detector.
- **M5 — Thermal occupancy. ✅ DONE.** `occupied` is now a thermal degree of freedom — a
  non-conserved **repulsive** lattice gas (`engine.lattice.occupancy_sweep`) on a new per-site
  `cohesion` field (from `bond_energy`), with `Conditions.pressure` entering as the
  chemical-potential μ. `engine/thermal.py` gains the occupancy ensemble (staggered-density order
  parameter, occupancy heat capacity, `melting_point`); `melting_temperature` is a stored property
  gated by solidity (like Curie).
  - **Keystone validation (parameter-free, the user chose *crystalline* melting):** the occupancy
    order-disorder transition is melting — `C(T)` peaks at the textbook `T_m = 2.269·J0·⟨coh²⟩`
    exactly where the staggered order parameter `ψ` collapses, **at fixed density ½** (positional
    order lost without a density change → crystalline melting, not the sublimation an attractive
    lattice gas gives). `tests/test_melting.py`; `tools.explorer melting-sweep`.
  - **Falls out / re-validated:** `T_m` tracks `bond_energy` (tungsten > zinc — recovers the real
    ordering); pressure tunes density and drives the M2 percolation transition (pressure-tuned
    percolation re-validates M2 under thermal occupancy). **Honest scope (no-fudge):** it is a
    *continuous* order-disorder transition (β-brass analogue), not a first-order melt with latent
    heat; `melting_temperature` is the bond network's order-disorder point at *commensurate*
    filling (intrinsic to bonding, not the standard-conditions fill); stored value is coarse
    (lean sweep); the cooling-rate → grain-size process signal is weak (reported), so the process
    payoff is structural **density** under a pressure schedule. See README **M5 findings**.
- **M6 — Transport + honest superconductivity.** **M6a ✅ DONE:** superconductivity via
  phase-coherence (XY/BKT) → real measured Tc (helicity-modulus universal-line crossing); static
  proxy retired; on-demand (slow XY equilibration). **M6b (next):** thermal conductivity (reuse
  the Laplacian; `atom_type`-gated phonon/electronic carriers → Wiedemann–Franz-like).
- **M7 — Spectral.** Band gap → conductor/semiconductor/insulator (`eigh`).
- **M8 — Mechanical.** Strength / ductility / fracture; confirm the strength↔ductility
  anti-correlation emerges.

Then the spec's machine layer (motor) and game shell follow.

---

## 7b. The process layer — synthesis as a *trajectory* through conditions-space

A `Conditions` is a *point* in dial-space; a **process** is a *path* through it. Today
`combine` settles at one fixed point (`relax` at `T0`). The process layer
(`engine/process.py`, Step 1 done) generalises this: a `Process` is an ordered list of
`Stage`s (hold/ramp `T`, with field `H`, over N sweeps), and `run_process` carries the live
spin state along that schedule using the *same* kernel `relax` uses
(`engine.lattice.metropolis_sweep`). A single constant-`T` hold reproduces `relax`
byte-for-byte (`STANDARD_PROCESS`), so this is a strict generalisation; `combine`/`from_element`
take an optional `process=`.

This is where real-world synthesis complexity lives (anneal/quench/field-cool, sequence,
catalysts) — and it stays inside the One Principle: the process shapes the **structure**;
properties are still measured from the result. "Wrong process = different material" is a
*measured* microstructure consequence, not a hidden gate.

**Validated (Step 0 de-risk → Step 1 tests):** slower cooling reaches lower-energy,
larger-domain structures (monotonic); field-cooling builds remanence (`≈0.97` vs `≈0.1`). Key
finding: net `|M|` at `H=0` is a *poor* process readout (random domain sign cancels) — the
signal is in **microstructure** (`engine/properties/microstructure.py`) and field-remanence.

**Sequencing:** at the spin level this is *magnetic-microstructure* path-dependence only.
The dramatic multi-property version (melting/grain growth/amorphous-vs-crystalline,
martensite-style quench hardening) arrives when **M5** makes `occupied` thermal — and M5's
dynamics plug into this same executor. The hard-magnet payoff (coercivity) needs a separate,
de-risked **anisotropy + hysteresis** block. Open: a process is not yet part of material
identity (caching by process is a follow-up); pressure `P` rides on each `Stage` but is inert
until M5.

## 8. Related parked idea

`future-ml-surrogate-hierarchy.md` (ML surrogate of `combine()` that *generates the field*,
keeping the measurement extractors downstream) is compatible with this architecture: a
surrogate would replace the expensive `merge → relax` step, and the conditions layer's
ensemble measurements would run on the generated structure exactly as on a real one. Not on
this path; revisit later.
