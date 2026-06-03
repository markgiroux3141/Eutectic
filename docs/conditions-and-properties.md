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
- **superconductivity with a *real* Tc** — model as a **phase-coherence (XY) ordering**
  transition on the conducting network (which is what SC physically is). Below Tc phases
  lock → dissipationless; above → normal. Tc emerges from network structure; the existing
  k-edge-connectivity work becomes the *coupling input*, not the label. Reuses Ising
  machinery. **This is the honest replacement for the current static SC proxy** (see
  `superconductivity-status` / README M3 findings).
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

## 5. The single highest-leverage move

Making **`occupied` a thermal degree of freedom** (non-conserved vacancy dynamics with a
pressure/chemical-potential term — *simpler* than the conserved Kawasaki dynamics we tested
and rejected for SC) unlocks, from one change: **melting, thermal expansion, density(T,P),
pressure-tuned conductivity/superconductivity, and heat capacity as a transition detector.**
It also touches the validated M2/M3 percolation/density, so it must re-validate them.

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

- **M4 — Conditions & thermodynamic state (the enabler).** Introduce `Conditions(T, P, H)`.
  Make property extractors measure from an ensemble at conditions; store a reference
  snapshot at standard conditions for legibility. **Keystone validation:** first-class
  magnetism(T)/Curie point + **heat capacity as a universal transition detector** (confirm
  C(T) peaks at the Curie point). Determinism preserved; M2/M3 stay green.
- **M5 — Thermal occupancy.** Make `occupied` thermal (non-conserved + pressure/μ). Falls
  out: melting point, thermal expansion, density(T,P), pressure-tuned conductivity.
  Re-validate M2/M3 under the new dynamics.
- **M6 — Transport + honest superconductivity.** Thermal conductivity (reuse the Laplacian);
  superconductivity via phase-coherence (XY) → real Tc; retire/relabel the static SC proxy.
- **M7 — Spectral.** Band gap → conductor/semiconductor/insulator (`eigh`).
- **M8 — Mechanical.** Strength / ductility / fracture; confirm the strength↔ductility
  anti-correlation emerges.

Then the spec's machine layer (motor) and game shell follow.

---

## 8. Related parked idea

`future-ml-surrogate-hierarchy.md` (ML surrogate of `combine()` that *generates the field*,
keeping the measurement extractors downstream) is compatible with this architecture: a
surrogate would replace the expensive `merge → relax` step, and the conditions layer's
ensemble measurements would run on the generated structure exactly as on a real one. Not on
this path; revisit later.
