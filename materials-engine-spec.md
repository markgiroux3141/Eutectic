# Emergent Materials Engine — Design Spec

## 0. Purpose of this document

This is an implementation spec for the core mechanic of a crafting/engineering game: a
system that takes root **elements** (iron, gold, uranium, …), lets the player **combine**
them into new **materials**, and derives each new material's physical **properties**
(strength, conductivity, magnetism, ductility, superconductivity, …) deterministically.

The hard part — and the thing this spec exists to get right — is *how new material
properties are determined*. The answer this spec commits to is a single design principle,
stated next. Everything downstream follows from it.

---

## 1. The one design principle

> **Measure properties from a structure. Never assign them from a function.**

A material is not a vector of numbers. A material is a small **lattice** (a grid of cells).
Its properties are *measurements taken from that lattice* after it settles — never values
emitted by a black-box function of the parents.

Why this and not a hash / uninitialized neural net:

- A hash *assigns* properties, so neighboring inputs scatter to unrelated outputs. Players
  can't build intuition; the tech tree becomes a lookup table the community solves in a
  weekend. It gives determinism and rarity but kills learnability and emergence.
- Measuring properties from a structure gives determinism, rarity, **and** learnability at
  once, because properties correlate with structure. Players reason ("regular dense
  lattices come out strong but brittle") instead of memorizing.
- Emergent properties like superconductivity are, in real physics, **collective phase
  transitions** — they appear suddenly past a critical threshold and belong to the whole
  system, not any single piece. To make them "fall out," the substrate we measure must
  itself contain phase transitions. It does (percolation, Ising). See §5.

If a future change ever tempts us back toward "just have a function output the numbers,"
that is the moment to re-read this section. The roulette/rarity feel must emerge from
thresholds in the substrate, not from a probability dial bolted onto a hash.

---

## 2. Stack & scope (defaults — override if you prefer)

**Recommendation: build the materials engine as a standalone, headless, deterministic
library first, with an explorer tool — before any game/rendering work.** The central risk
is "do interesting properties actually emerge and feel good?", and that's answered fastest
in a headless harness.

- **Prototype engine language: Python + numpy + scipy.** Rationale: percolation labeling
  (`scipy.ndimage.label`), Laplacian/effective-resistance solves (`scipy.sparse.linalg`),
  eigenvalue gaps (`numpy.linalg.eigh`), and Ising relaxation are all near one-liners.
  The math is portable to TypeScript/Rust later if the shipping game needs it in-engine.
- **Explorer tool:** a CLI + simple plots (matplotlib) to browse combinations and view
  property *distributions* across many random combos. This is how we verify thresholds and
  rarity look right.
- **Game shell (inventory, crafting UI, building, machines):** a separate, later effort.
  Keep it out of the engine library. The engine must never import game/UI code.

Dimensionality: **make lattice dimension a config parameter.** Default to **2D 64×64 for
prototyping** (easy to visualize and debug; all the math is dimension-agnostic), with 3D
16×16×16 as the target once the mechanic feels right.

---

## 3. Data model

### 3.1 Element (root material)

```
Element:
  id: str                      # "iron", "gold", "uranium"
  display_name: str
  atomic_mass: float           # feeds density
  signature: int               # stable seed used to generate this element's base lattice
  base_affinities: dict        # small set of scalars that bias merging/relaxation
                               # e.g. {bond_energy, magnetic_tendency, conduction_tendency}
```

Root elements get their lattice generated deterministically from `signature` +
`base_affinities` (see §4.1). Author ~15–30 root elements by hand to start.

### 3.2 Lattice

```
Lattice:
  dim: int                     # 2 or 3
  shape: tuple                 # (64,64) or (16,16,16)
  cells: ndarray               # per-cell state; see below
```

Per-cell state (start minimal, expand as properties demand):

```
cell = {
  occupied: bool / 0|1         # is there matter here (fill fraction → density, percolation)
  atom_type: int               # which "kind" of site; drives bond rules
  spin: int8 in {-1,+1}        # for Ising magnetism
}
```

Represent the lattice as a few parallel numpy arrays (`occupied`, `atom_type`, `spin`)
rather than an array of structs — keeps the math vectorized.

### 3.3 Material

```
Material:
  id: str                      # deterministic, derived from lineage (see §6)
  lattice: Lattice             # the settled lattice
  properties: dict[str,float]  # QUANTIZED measured properties (see §6 on quantization)
  lineage: (parent_a_id, parent_b_id)   # or element id for roots
  discovered_at: <game event>  # optional, game layer only
```

---

## 4. The combination pipeline (deterministic)

`combine(A, B) -> Material`. Four stages: **hash → merge → relax → measure**, then cache.

### 4.1 Hash / seed

- Canonically order parents (sort by id) so `combine(A,B) == combine(B,A)`.
  **Decision: combination is commutative in v1.** (Open question §9.)
- `seed = mix(A.structural_signature, B.structural_signature, UNIVERSE_SEED)`
  where `structural_signature` is a stable hash of the parent's lattice contents.
- Use one explicit deterministic PRNG (e.g. a seeded SplitMix64 / PCG implementation we
  control). **Never** use Python's global `random`, time, or hash-map iteration order.

Root element lattices are generated by the same kind of seeded process from
`Element.signature`.

### 4.2 Merge

Produce a child lattice from the two parent lattices, driven by the seeded PRNG and the
parents' cell states. Start simple:

- For each cell, deterministically choose to inherit from A, from B, or blend, where the
  bias is a function of the parents' `base_affinities` and the PRNG stream.
- Interleaving / domain patterns are fine and desirable — they create the structure that
  later produces anisotropy and spanning clusters.

### 4.3 Relax

Run a short, fixed number of settling steps so the merged lattice reaches a stable
configuration. This is where emergence happens (ordered domains, spanning clusters form).

- Option A (recommended start): a few sweeps of local energy minimization on `spin` /
  `occupied` (Metropolis-style but **deterministic**: drive all randomness from the seeded
  PRNG, iterate cells in a fixed order).
- Option B (your CA instinct, can layer in later): let the parents parameterize a
  Lenia/SmoothLife-style continuous CA rule, run to an attractor. Interesting rules live at
  the rare "edge of chaos" (Wolfram class IV), which gives built-in rarity. Treat this as an
  alternative relaxation backend behind the same interface.

Fixed step count + fixed iteration order + seeded PRNG ⇒ identical settled lattice every
time.

### 4.4 Measure → 4.5 Cache

Run the property extractors of §5 on the settled lattice, quantize (§6), store, and cache
keyed by the canonical parent pair so a combination is only ever computed once.

---

## 5. Property extractors (the substrates)

Each extractor is a pure function `Lattice -> float`. Group them in
`engine/properties/`. Build in this order; each is independently testable.

### 5.1 Scalar blends (cheap, predictable — gives players an anchor)

- **density** = `mean(occupied) * weighted_mean(atomic_mass by atom_type)`
- **mass-ish quantities**: straightforward reductions.

These should be *legible*: players can roughly predict them. That's intentional contrast
to the threshold properties below.

### 5.2 Percolation → conductivity (threshold behavior, rarity for free)

- Label connected clusters of conducting cells (`scipy.ndimage.label`).
- **conductivity (boolean/continuous)**: does a cluster *span* the lattice from one
  face to the opposite face? Below a critical fill density there is no spanning cluster
  (insulator); right around the critical point a spanning cluster appears suddenly
  (conductor). The sharp threshold is the rarity mechanism — most materials land clearly on
  one side, the interesting ones sit near the critical point.

### 5.3 Effective conductance (Laplacian) → resistance, and the road to superconductivity

- Treat the conducting lattice as a resistor network. Solve the graph Laplacian for
  effective resistance between opposite faces (`scipy.sparse.linalg`).
- **resistance** = effective resistance; **conductivity (continuous)** = `1/resistance`.
- Normal conductors have finite resistance; once enough redundant parallel paths exist,
  effective resistance collapses toward zero.

### 5.4 Superconductivity (threshold on a threshold ⇒ exponentially rare)

Superconductivity is **not** a separate dial. It's a rare condition *conditional on* an
already-uncommon one:

- Requires: (a) a spanning cluster exists (§5.2), AND
- (b) the cluster is "loss-free" — encode as: minimum bottleneck width ≥ threshold
  (compute via min-cut / graph connectivity), AND/OR defect-free metric above threshold,
  AND/OR effective resistance below epsilon (§5.3).
- Output a **continuous "coherence" score** and a `Tc`-like value, plus a boolean flag when
  it crosses the bar.

Because you're requiring a rare property given an already-uncommon one, superconductors come
out exponentially rare **without tuning a probability**. Verify this empirically in the
explorer (§7): plot the distribution and confirm superconductors are a thin tail.

### 5.5 Ising → magnetism (the canonical emergent phenomenon)

- Derive neighbor coupling constants from lattice structure (atom_type adjacency,
  affinities).
- Relax spins (this can be folded into §4.3). Below a critical coupling, spins are
  disordered → no net magnetism; above it, spontaneous alignment emerges across the lattice.
- **magnetism** = `abs(net magnetization)` of the settled spin field. Emergent, with a clean
  critical point.

### 5.6 Spectral / band gap → conductor vs semiconductor vs insulator

- Build a matrix from the lattice (adjacency or a toy Hamiltonian), compute eigenvalues
  (`numpy.linalg.eigh`), measure the gap between two specific eigenvalues.
- gap ≈ 0 → conductor; small gap → semiconductor; large gap → insulator. Mirrors real band
  theory; deterministic.

### 5.7 Mechanical (strength / ductility / sharpness / melting)

- **strength (tensile/compressive/shear)**: bond density × connectivity; can reuse the
  Laplacian/rigidity of the occupied lattice.
- **ductility**: availability of low-energy shear rearrangements ("slip planes") — count
  configurations where the lattice can shear without the connected structure fracturing.
  Note: high strength + high ductility being hard to get *simultaneously* will fall out
  naturally — that's a real materials tradeoff we get for free, not a hardcoded rule.
- **sharpness**: anisotropy × hardness (can it hold a thin edge).
- **melting point**: sum of bond energies / coordination.

### 5.8 OPTIONAL post-transform (use sparingly, if at all)

The `1/(target - value)` resonance/pole idea from the design chat can be applied *on top of
a measured value* to sharpen a tail if a property's natural distribution is too flat. Mark
any such transform clearly and prefer fixing the substrate first. Do **not** use it as the
primary property source — that reintroduces the hash problem §1 warns against.

---

## 6. Determinism contract (read carefully — these are the classic bugs)

Same parents (and same UNIVERSE_SEED) must yield byte-identical materials, forever, across
runs. Enforce:

1. **One seeded PRNG we control** (SplitMix64/PCG). No global `random`, no `time`, no
   `os.urandom`, no Python `hash()` of strings (salted per process).
2. **Fixed iteration order everywhere.** No iterating over `set`/`dict` for anything that
   affects results; sort first.
3. **No parallel reductions that change float summation order** in the property math (or
   use a fixed reduction order).
4. **Quantize final property outputs** to fixed precision (e.g. round to 4 decimals or to
   integer "stat" units) before storing/comparing. This prevents floating-point dust from
   making two identical materials compare as different, and makes caching/equality robust.
5. **Material id is derived from lineage**, canonically ordered: e.g.
   `id = hash(sorted(parent_ids) + [UNIVERSE_SEED])`.
6. `UNIVERSE_SEED`: a single global seed mixed into every combination. Default `0` /
   fixed. Setting it per-save makes each playthrough's material space different while
   staying fully deterministic within a save (this is the Noita trick — delays the community
   from mapping the whole tech tree). Keep it a config knob.

A determinism test (run pipeline twice, assert identical lattices + properties) must be in
the suite from M1 onward and run in CI.

---

## 7. Explorer tool (verification harness)

Before trusting the mechanic, we need to *see* the property space. Build a small tool that:

- Generates N random combinations (and N random multi-step combination chains).
- Plots histograms / scatter of each property across the population.
- Confirms: conductivity is bimodal-ish around a threshold; superconductors are a thin
  rare tail; magnetism shows a critical transition; strength/ductility are anti-correlated.
- Lets you inspect a single material: render its lattice (2D heatmap), show its spanning
  cluster, show measured properties and how they were derived.

If the distributions look boring or uniform, the substrate needs tuning — fix it here, not
in the game.

---

## 8. Object / machine layer (later milestone)

Objects are assemblies: each object defines **roles**, each role has **property
requirements**, and performance is computed by real-ish equations from the assigned
materials' properties.

**Worked example — electric motor:**

- Role `coil_wire` requires: low electrical resistance, high melting point, high ductility
  (manufacturability into wire).
- Role `core` requires: high magnetism (permeability).
- Simplified performance model:
  - `torque ∝ flux(core.magnetism) × current`
  - `current` limited by `coil_wire.resistance` (Ohm) and by I²R heating vs
    `coil_wire.melting_point` (burnout limit)
  - `efficiency` penalized by resistance losses
- Surface these as the object's stats. The point: a player who discovered a rare
  low-resistance, high-melt, ductile material can build a markedly better motor — the
  payoff loop for the whole system.

Keep the equations in a `machines/` module separate from the engine; they consume
`Material.properties` only.

---

## 9. Open design decisions (resolve as you go)

1. **Commutativity & arity.** v1: commutative, exactly 2 parents. Consider order-dependent
   or 3+ ingredient combinations later (Noita's recipes are ternary).
2. **Lattice dimension.** Prototype 2D, ship 3D — confirm the feel survives the jump.
3. **Visible vs hidden lattice.** If players can see the structure, the game is a
   deduction/engineering toy; if hidden, it's a discovery/gambling loop. Different games —
   decide intentionally. (The explorer should show it regardless, for dev.)
4. **Universe seed on/off** for shipped game (replayability vs shareable global wiki).
5. **Shipping engine language** if the in-game runtime needs the math (TS/Rust port).

---

## 10. Suggested module layout

```
engine/
  rng.py            # seeded deterministic PRNG + hashing/mixing
  elements.py       # root element definitions + base lattice generation
  lattice.py        # Lattice type, generation, merge, relax (CA/energy backends)
  material.py       # Material type, combine() pipeline, quantization, caching
  registry.py       # discovered materials + lineage graph
  properties/
    scalar.py       # density, mass
    percolation.py  # spanning cluster, conductivity (boolean/continuous)
    conductance.py  # Laplacian effective resistance, superconductivity
    ising.py        # magnetism
    spectral.py     # band gap classification
    mechanical.py   # strength, ductility, sharpness, melting
machines/
  motor.py          # worked example; performance equations
tools/
  explorer.py       # distributions, single-material inspection, lattice render
tests/
  test_determinism.py   # combine twice → identical (run in CI)
  test_distributions.py # thresholds present, superconductors rare, etc.
```

---

## 11. Build milestones (sequence for Claude Code)

- **M0 — Foundations.** `rng.py`, hashing/mixing, `Lattice` type, root `elements.py`,
  ability to generate + render/inspect a single element's lattice.
- **M1 — Combination pipeline.** `combine()` doing hash → merge → relax → produce a child
  lattice. **`test_determinism.py` passing.** No properties yet beyond a placeholder.
- **M2 — Legible properties + first threshold.** `scalar.py` (density) and
  `percolation.py` (conductivity). Explorer shows histograms; confirm the conductivity
  threshold is visible.
- **M3 — Emergent properties + rarity.** `ising.py` (magnetism), `conductance.py`
  (resistance + superconductivity double-threshold). Confirm in explorer that
  superconductors are a thin rare tail and magnetism shows a transition.
- **M4 — Remaining properties.** `spectral.py` (band gap), `mechanical.py` (strength,
  ductility, sharpness, melting). Confirm strength/ductility anti-correlation emerges.
- **M5 — Machine layer.** `machines/motor.py` + role/requirement framework; build a motor
  from materials and compute its performance.
- **M6 — Game shell (separate effort).** Inventory, crafting UI, building, progression.

Each milestone is shippable and independently testable. Do not start M-game-shell until the
explorer shows the property space feels good — that's the whole bet.
