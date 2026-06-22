# Chemistry Engine — Design Spec

## 0. Purpose of this document

This specifies a **chemistry layer** for Eutectic: a deterministic system where root **atoms**
(defined by distilled quantum descriptors) **bond** into **molecules/compounds**, **react** under
**conditions** (temperature, pressure, concentration, catalysts) along a **reaction network**, and
whose products' **bulk material properties** are then *measured* by the existing materials engine.

It is the layer *below* the current materials engine. Read `materials-engine-spec.md` first — this
document assumes its one principle, its determinism contract, and its no-fudge validation culture,
and extends all three to a new level of the hierarchy.

The design fork this commits to (decided with the user): **explicit orbitals + hybridization** for
the atom model, **full integration** with the materials engine (compounds feed the existing
extractors directly), and a **full reaction network** (thermodynamics + kinetics + catalysts +
prerequisite graph) as the target. The milestone ladder (§18) de-risks that ambition in stages.

---

## 1. The one principle, at two levels

> **Measure properties from a structure. Never assign them from a function.** (materials spec §1)

The chemistry layer obeys this twice:

- **Molecular level:** a molecule/compound is a *bonding graph* of atoms, settled by energy
  minimization. Its molecular properties — geometry, stability, stoichiometry, bond character —
  are **measured from that graph**, never assigned. We do not write "salt is 1:1 ionic"; the 1:1
  ratio and the ionic character *fall out* of valence + electronegativity.
- **Bulk level:** a compound's crystal **is a lattice** (the existing `engine.lattice.Lattice`).
  Its bulk properties (conductivity, magnetism, melting point, strength, …) are measured by the
  **existing** extractors. Chemistry's job is to *generate the structure*; the materials engine
  measures it.

The unifying thesis:

> **Chemistry generates structure; the materials engine measures it.**

Emergence and rarity come from **thresholds in the substrate**, exactly as before: which compounds
are stable is a question of energy minima (deep, narrow wells = rare compounds), which reactions
proceed is a question of ΔG sign and activation barriers — never a probability dial bolted onto a
hash.

---

## 2. Relationship to the materials engine (the integration thesis)

Today an "element" is an atom and a "material" is a spatial blend (`combine` = domain merge).
Real materials are compounds. So:

- **Elements gain quantum descriptors** (§5). The current macro-affinities (`bond_energy`,
  `magnetic_tendency`, `conduction_tendency`) become **derived quantities** of those descriptors,
  not authored independently — so the existing M0–M8 keystones can be preserved (a derivation that
  reproduces today's affinities) or consciously re-validated. **This is the single riskiest
  integration point; treat it as a keystone in its own right** (§18, C2).
- **Two ways to make a new material, both producing a `Lattice`:**
  - `combine(A, B)` (existing) → **physical mixture / alloy**: a domain blend, no new bonds.
  - `react(reactants, conditions)` (new) → **chemical compound**: new bonding, new stoichiometry,
    a crystal structure.
- A `Material` can therefore be an **element**, a **mixture** (combine), or a **compound** (react).
  All three are lattices; all three are measured by the same extractors. The registry/lineage
  tracks which.

The dependency arrow: `chemistry → engine.lattice` (chemistry builds lattices) and
`engine.material → chemistry` (combine/react live at the material level). Chemistry must **not**
import game/UI. The materials extractors must **not** know whether a lattice came from an element,
a mixture, or a compound.

A bonus the integration unlocks: **M7 (band gap) becomes viable again.** It was deferred because
diluted blends had no real crystal structure (5% vacancies closed any gap). Real compounds have
real unit cells — a tight-binding/Hückel pass over a covalent-network crystal (diamond, quartz)
can give an honest semiconductor/insulator gap. M7 is a natural C2+ deliverable here.

---

## 3. How far down: the distilled-quantum stance

We do **not** simulate quantum mechanics (no DFT, no wavefunction solves — infeasible,
non-deterministic, and unnecessary). We **distill** QM's outputs into a handful of per-element
descriptors and let chemistry *emerge* from rule-based energy minimization over them. This is the
same move as modeling magnetism with Ising spins rather than real spin dynamics: the model is a
distillation, but the emergent phenomena (critical points, stoichiometry, compound stability) are
real.

The distilled descriptors are the compressed output of QM:

- **Electron configuration → valence shell** (which orbitals, how many electrons, how many
  unpaired) — distilled aufbau/Hund.
- **Electronegativity** — distilled orbital-energy/charge-pull.
- **Atomic / covalent / ionic radius** — distilled orbital extent.
- **Ionization energy & electron affinity** — distilled cost/gain of moving an electron.

From just these, an enormous amount of chemistry emerges: bonding capacity, bond type, molecular
geometry, stoichiometry, compound stability, reaction energetics. Going lower (real tight-binding
for *every* property) is reserved for the few properties that genuinely need it (band structure)
and is treated as an optional backend, never the default.

---

## 4. Data model

```
Atom (root):
  symbol: str                  # "H", "Na", "Cl", "C"
  atomic_number: int           # Z
  atomic_mass: float
  electron_config: tuple       # distilled to valence-shell occupancy (s, p, d counts)
  electronegativity: float     # Pauling-like scalar
  covalent_radius: float
  ionic_radius: dict[int,float]# radius by oxidation state (optional)
  ionization_energy: float
  electron_affinity: float
  # DERIVED (never authored): valence_electrons, bonding_capacity (unpaired/octet deficit),
  #   lone_pairs, available_orbitals (s/p/d for hybridization)

Bond:
  a_index, b_index: int        # atoms it joins (within a Molecule)
  order: float                 # 1 single, 2 double, 3 triple (fractional allowed for resonance)
  character: enum               # IONIC | POLAR_COVALENT | COVALENT | METALLIC  (from ΔEN)
  energy: float                # MEASURED from the bonding model (§7)

Molecule / Compound:
  atoms: list[Atom-instance]   # with oxidation state / formal charge
  bonds: list[Bond]
  geometry: per-atom hybridization + idealized bond angles (VSEPR, §6)
  formula: derived (canonical, e.g. "H2O")  # stoichiometry FALLS OUT
  formation_energy: float      # MEASURED (sum of bond energies, ± lattice/solvation)
  phase: enum                  # GAS | LIQUID | SOLID  (from conditions + cohesion)

Crystal (Compound -> Lattice):
  packing: enum                # ROCK_SALT | DIAMOND_NET | CLOSE_PACKED | MOLECULAR | ...
  lattice: engine.lattice.Lattice   # the bridge to the materials engine (§9)

Reaction:
  reactants: dict[species,int] # stoichiometric coefficients
  products:  dict[species,int]
  delta_H, delta_S: float      # MEASURED (Hess's law over bond energies; entropy estimate)
  activation_energy: float     # MEASURED (transition-state estimate)
  catalysts: set[species]      # species that lower Ea without being consumed

ReactionNetwork:
  species: set
  reactions: list[Reaction]
  # reachable(inventory, conditions) -> set of attainable species (the tech tree)

ChemConditions(T, P, concentration, catalysts):  # extends engine.Conditions
```

---

## 5. The atom model (C0)

Author ~20–40 root atoms by hand with the §4 descriptors (the existing element set is the seed;
extend toward a usable slice of the periodic table — at minimum H, C, N, O, Na, Cl, Mg, Fe, Cu, Si,
plus the current metals). Periodic data is from distilled real values (this is *reference data*,
not a fudge — like atomic_mass already is).

**Derived, never authored** (the emergence starts here):

- `valence_electrons` from the electron configuration.
- `bonding_capacity` from unpaired electrons / octet (or duet for H/He) deficit, with **expanded
  octet** allowed when d-orbitals are available (this is *why* we chose the orbital model: it lets
  PCl₅/SF₆-style hypervalency emerge instead of being special-cased).
- `lone_pairs` from (valence − bonding electrons) / 2.
- `available_orbitals` (s, p, d) for hybridization (§6).

**Keystone (C0):** valences reproduce parameter-free — H→1, O→2, N→3, C→4, Na→1(+), Mg→2(+),
Cl→1(−) — straight from electron configuration. Electronegativity/radius show the right **periodic
trends** (EN rises across a period and falls down a group; radius the reverse). If we *derive* EN
from Z/shell structure, the trend is a keystone; if authored, it must at least be consistent.

---

## 6. Orbitals, hybridization, VSEPR geometry (C1a)

Molecular **geometry emerges from electron counting**, not assignment:

1. **Steric number** = σ-bonds + lone pairs on the central atom.
2. **Hybridization** from steric number: 2→sp (linear), 3→sp² (trigonal planar), 4→sp³
   (tetrahedral), 5→sp³d (trigonal bipyramidal), 6→sp³d² (octahedral). (d-orbitals gate 5/6 —
   hence the orbital model.)
3. **Idealized bond angles** from VSEPR, with lone-pair repulsion compressing them (e.g. H₂O's
   104.5° vs the tetrahedral 109.5° because two lone pairs squeeze the bonds).

Output: per-atom hybridization + a geometry (topology + idealized angles; full 3D coordinates
optional, useful for the explorer and for crystal packing in §9).

**Keystone (C1a):** CH₄ tetrahedral (109.5°), H₂O bent (~104.5°), CO₂ linear (180°), NH₃ trigonal
pyramidal — all from steric number + lone-pair repulsion, no per-molecule rules.

---

## 7. Bonding model (C1b)

A bond's **character** emerges from the electronegativity difference ΔEN (Pauling-style thresholds,
to be confirmed in de-risk): ΔEN ≳ 1.7 → **ionic**; 0.4–1.7 → **polar covalent**; < 0.4 →
**covalent**; two electropositive (low-EN) atoms → **metallic** (delocalized electron sea).

Bond **order** from shared electron pairs needed to satisfy both atoms' shells (single/double/
triple; fractional for resonance).

Bond **energy** is *measured* from a distilled model, by character:

- **Covalent:** orbital-overlap model — `E ∝ order × f(EN_avg) / (r_a + r_b)` (closer, higher-order,
  more-electronegative pairs bond harder). Calibrate the proportionality once against a textbook
  bond-energy or two; it is a fixed scale, not a per-bond dial.
- **Ionic:** Coulombic / Born-Landé lattice energy — `E ∝ (z⁺·z⁻) / (r⁺ + r⁻) × Madelung` (charges
  from electron transfer; Madelung from the crystal packing in §9).
- **Metallic:** electron-sea cohesion `∝ valence-electron density × overlap`.

**Keystone (C1b):** NaCl comes out ionic, Cl₂ covalent, Cu metallic; bond energies order sensibly
(triple > double > single; C–C strong, Na–Cl ionic-strong). One or two calibrated constants, the
rest emergent ordering.

---

## 8. Molecule formation = constrained energy minimization (C1c)

The "combine" analog at the molecular level. Given a set of atoms and conditions, find the bonding
configuration that **minimizes total energy** subject to:

- **Valence/shell satisfaction** (octet/duet, expanded octet where allowed) — every atom wants a
  full shell;
- **Charge balance** (for ionic — total transferred electrons balance);
- **Geometry/steric feasibility** (VSEPR angles, no impossible crowding).

For small molecules this is a **deterministic combinatorial search** over bonding topologies; for
larger/condensed systems, a **deterministic simulated anneal** reusing the existing Metropolis +
`SplitMix64` machinery (the same kernel that settles spins/occupancy settles bonds). Output: the
molecular graph + `formation_energy`.

**Stable compound ⇔ energy minimum.** Stoichiometry is *not* searched-for and assigned — it is the
ratio at which valence/charge balance is satisfied at minimum energy.

**Keystone (C1c):** stoichiometry falls out parameter-free — Na+Cl→**NaCl (1:1)**, Mg+Cl→**MgCl₂
(1:2)**, H+O→**H₂O (2:1)**, C+O→**CO₂ (1:2)**; noble gases refuse to bond (full shell → no energy
gain). A nonsense pairing (e.g. He–Na) has no stabilizing minimum and does not form.

---

## 9. Compound → crystal lattice (C2 — the integration)

The bridge to the materials engine. A compound's bulk structure is a crystal; the chemistry engine
emits an `engine.lattice.Lattice` whose **per-cell fields are set by the bonding**, so the existing
extractors measure it unchanged:

- **packing** chosen by bond character: ionic → alternating-charge sublattice (rock-salt);
  covalent-network → coordination tiling (diamond/tetrahedral, quartz); metallic → close-packed;
  molecular → weakly-bound molecular units on a lattice.
- **`occupied`** from the packing; **`atom_type`** from the species arrangement (sublattices!);
- **`cohesion`** from bond energies (drives melting, M5; strength, M8);
- **`metallicity`** from bond character (metallic/delocalized → conductive; ionic/covalent → not —
  this is what makes NaCl an insulator and Cu a conductor, M6b);
- **`moment`** from **unpaired electrons** (→ magnetism, M3/M4 — now sourced from real electron
  structure, not an authored tendency);
- **`mass`** from atomic masses.

The materials engine then measures conductivity, magnetism, melting, superconductivity, strength,
(and band gap, M7-revived) from this lattice with **no changes to the extractors**.

**Keystone (C2):** measured bulk properties match the compound's chemistry — **NaCl insulates**
(ionic, no free carriers), **Cu conducts** (metallic), **diamond has a band gap** and tops thermal
conductivity (covalent network, the M6b divergence re-derived from real bonding), an **unpaired-
electron compound is magnetic**. Also the **affinity-derivation keystone** (§2): the current
elements' macro-affinities, when *derived* from their quantum descriptors, reproduce the M0–M8
stored values closely enough to keep those milestones green (or the deltas are understood and
re-validated).

---

## 10. Reaction thermodynamics (C3)

A reaction transforms reactants → products; feasibility is **measured** from energetics:

- **ΔH** by Hess's law over the bonding model: `ΔH = Σ E(bonds broken) − Σ E(bonds formed)`
  (exothermic when products are more strongly bound).
- **ΔS** from a distilled entropy estimate: change in number of gas-phase species, phase changes,
  and symmetry (coarse but directional — more gas molecules / more disorder → higher S).
- **ΔG = ΔH − T·ΔS.** Spontaneous when ΔG < 0. Equilibrium constant `K = exp(−ΔG / RT)`.
- **Conditions bite:** T scales the −T·ΔS term (entropy-driven reactions switch on when hot);
  P shifts gas-phase equilibria (Le Chatelier — fewer-gas-mole side favored under pressure);
  concentration shifts K toward/away from products.

**Keystone (C3):** known exergonic reactions proceed (2H₂+O₂→2H₂O, ΔG<0), endergonic ones don't —
**until a temperature threshold flips ΔG's sign** for an entropy-favored reaction (the C-peak/
threshold discipline: a real transition where ΔG crosses zero, not a smooth dial). Le Chatelier
shifts reproduce qualitatively under P and concentration.

---

## 11. Kinetics & catalysts (C4)

Feasibility (ΔG) is necessary but not sufficient — **rate** gates what actually happens:

- **Arrhenius rate:** `rate = A · exp(−Ea / RT)` — the same Boltzmann/exponential machinery as
  Metropolis acceptance.
- **Activation energy Ea** from a transition-state estimate (e.g. a fraction of the bonds-broken
  energy along the lowest-barrier path).
- **Catalysts** provide an alternative pathway with **lower Ea** (model: the catalyst forms a
  transient intermediate complex, lowering the barrier), affecting *rate only* — it appears in the
  kinetics but **not** in the net stoichiometry and is not consumed.

**Keystone (C4):** the classic "thermodynamically favorable but kinetically trapped" case —
H₂+O₂ is strongly exergonic yet does not react at low T (high Ea) until **ignited (raise T) or
catalyzed (lower Ea)**. The rate jumps sharply across the activation threshold; a catalyst measurably
moves the threshold without changing ΔG.

---

## 12. Reaction network / tech tree (C5)

Species and reactions form a **directed graph**; what you can make emerges from it:

- **`reachable(inventory, conditions)`** — the set of species attainable from a starting inventory
  under accessible conditions (ΔG<0 *and* rate fast enough at the available T, given any catalysts
  present). This is the **tech tree**, and it is *emergent*: prerequisites (you must synthesize
  intermediate Y before X) and condition-gating (X only forms above some T/P, or only with a
  catalyst you must first obtain) fall out of the graph, not authored progression.
- **Rarity** re-appears here as it does everywhere: exotic compounds sit behind narrow condition
  windows and long prerequisite chains.

**Keystone (C5):** a multi-step synthesis path emerges for a target compound; the target is
*unreachable* without an intermediate or without unlocking a condition/catalyst — and the gate is a
real threshold (ΔG sign / activation), not a tuned probability.

---

## 13. Conditions (T, P, concentration, catalyst)

Extend `engine.Conditions(T, P, H)` to `ChemConditions(T, P, concentration, catalysts, [H])`.
`T` and `P` already exist and are live; chemistry adds **concentration** (shifts equilibria/rates)
and **catalysts** (a set of present species that lower specific Ea's). The `measure(structure,
conditions)` pattern generalizes to `react(reactants, conditions) → products` and
`reachable(inventory, conditions)`. Reuse the quantize-conditions-before-seeding discipline.

A reaction *pathway* (heat, add catalyst, cool, pressurize) is a **trajectory through
conditions-space** — i.e. it is a `process.Process` (the existing executor already threads live
state through condition schedules). Synthesis routes are processes; "wrong route → different
product/yield" is a legible consequence, not a gate.

---

## 14. Determinism contract (inherits materials spec §6)

Same rules, extended:

1. One seeded PRNG (`engine.rng.SplitMix64`); no global `random`/time/`hash()`.
2. Fixed iteration order over atoms/bonds/species — sort canonically (by symbol/index) before any
   reduction or search.
3. **Deterministic search/anneal** for molecule formation: fixed step count, fixed order, seeded.
4. **Quantize** energies, ΔG, rates, and geometries before storing/comparing.
5. Canonical molecule identity: a graph-canonical form (e.g. a Morgan-style canonical atom ranking)
   so the same compound always hashes/compares identically regardless of construction order.
6. Reaction identity from canonical (sorted) reactant/product multisets + conditions quantum.

A determinism test (form a molecule / run a reaction twice → byte-identical) is in the suite from
C1 onward.

---

## 15. Emergence & validation keystones (the no-fudge discipline, per level)

Every claim is pressure-tested against a textbook value **parameter-free** where one exists, and we
hunt for disguised dials (a rate that slides smoothly as a threshold is nudged = fake transition).
The keystone ladder, one per level (collected from §5–§12):

- **Valence** (C0): H/C/N/O/Na/Mg/Cl valences from electron config.
- **Geometry** (C1a): CH₄/H₂O/CO₂/NH₃ angles from steric number + lone pairs.
- **Bond type** (C1b): NaCl ionic, Cl₂ covalent, Cu metallic from ΔEN.
- **Stoichiometry** (C1c): NaCl 1:1, MgCl₂ 1:2, H₂O 2:1, CO₂ 1:2 from valence/charge; noble gases
  inert.
- **Bulk match** (C2): NaCl insulates, Cu conducts, diamond gaps + tops thermal κ, unpaired-electron
  compound is magnetic; **affinity-derivation reproduces M0–M8**.
- **Thermo** (C3): exergonic reactions go, endergonic don't until a T threshold flips ΔG; Le
  Chatelier under P/concentration.
- **Kinetics** (C4): favorable-but-trapped reaction (H₂+O₂) needs ignition/catalyst; catalyst moves
  the activation threshold, not ΔG.
- **Network** (C5): an emergent multi-step synthesis with real prerequisite/condition gates.

The recurring keystone pattern (from the materials engine): a transition is real when an order
parameter, a detector, and a textbook value coincide. Here the detectors are ΔG sign-crossings and
activation-barrier rate jumps; resist reusing one detector blindly across levels (the materials
engine's hard-won lesson that the heat-capacity peak marks Curie/melting but *not* the BKT Tc).

---

## 16. Explorer / verification harness

A `tools/chem_explorer.py` mirroring `tools/explorer.py` (ASCII stdout; unicode only in matplotlib):

- **inspect-atom** — descriptors + derived valence/orbitals/geometry capacity.
- **build-molecule** — given atoms, show the settled bonding graph, geometry, formula
  (stoichiometry), bond characters/energies, formation energy.
- **react** — reactants + conditions → products, with ΔH/ΔS/ΔG, K, Ea, rate, and which
  condition/catalyst gates it.
- **phase/condition sweep** — ΔG(T), rate(T), equilibrium vs P/concentration; show the threshold
  crossings.
- **tech-tree** — reachable species from an inventory under conditions; the synthesis graph.
- **measure-compound** — emit the compound's crystal lattice and run the **materials** extractors
  (the integration view: chemistry in, bulk properties out).

If the emergent space looks boring/uniform, fix the substrate here — not in the game.

---

## 17. Module layout

```
chemistry/
  atoms.py        atom descriptors + periodic data; derive valence/unpaired/lone-pairs/orbitals
  orbitals.py     hybridization (steric number) + VSEPR geometry / bond angles
  bonding.py      bond character (ΔEN), order, energy (covalent / ionic / metallic)
  molecule.py     Molecule graph; formation = constrained deterministic energy minimization; canonical id
  crystal.py      Compound -> engine.lattice.Lattice (packing + per-cell fields)   # the bridge
  reaction.py     ΔH (Hess) / ΔS / ΔG / K; feasibility; T,P,concentration dependence
  kinetics.py     Arrhenius rate; activation energy; catalysts (alternative low-Ea pathway)
  network.py      species + reaction graph; reachable()/tech-tree
  conditions.py   ChemConditions(T,P,concentration,catalysts) extending engine.Conditions
tools/
  chem_explorer.py
tests/
  test_valence.py test_geometry.py test_bond_type.py test_stoichiometry.py
  test_crystal_properties.py   # integration: bulk props match chemistry
  test_thermo.py test_kinetics.py test_network.py
  test_chem_determinism.py
```

Integration touch-points in the existing tree: `engine/elements.py` gains quantum descriptors (old
affinities derived from them); `engine/material.py` gains `react()` alongside `combine()`; the
extractors are untouched. The engine must not import `chemistry` for its *own* measurements — the
arrow is `chemistry → engine.lattice`, and `material.react` is the composition point.

---

## 18. Build milestones (the de-risked ladder)

Each milestone is shippable, independently testable, and gated by its keystone (§15). Build a
throwaway de-risk prototype and report the numbers **before** committing architecture, per the
no-fudge norm.

- **C0 — Atom model.** Descriptors + derived valence/orbitals. Keystone: valences + periodic trends.
- **C1 — Bonding & molecules.** (a) hybridization/VSEPR geometry; (b) bond character/order/energy;
  (c) molecule formation by energy minimization. Keystones: geometry, bond type, **stoichiometry**.
- **C2 — Compound → lattice + integration.** Crystal packing + per-cell fields; materials extractors
  measure compounds; derive old affinities from descriptors. Keystone: bulk-match + **M0–M8 stay
  green**. (M7 band gap revival is a stretch goal here.)
- **C3 — Reaction thermodynamics.** ΔG feasibility, equilibria, T/P/concentration. Keystone: thermo.
- **C4 — Kinetics & catalysts.** Arrhenius rate, Ea, catalysts. Keystone: trapped-reaction + catalyst.
- **C5 — Reaction network / tech tree.** Reachability graph, prerequisites, condition gates.
  Keystone: emergent multi-step synthesis.

**Sequencing note:** although the chosen target is "full integration / full network now," C2 is the
load-bearing risk (does the affinity derivation keep M0–M8 green, and do compound lattices measure
sensibly?). De-risk C2 *before* building deep reaction machinery — a reaction network producing
compounds whose bulk properties are wrong is worthless. The ambition is the destination; this order
is how we get there without breaking what works.

---

## 19. Open design decisions (resolve as you go)

1. **Where do electronegativity/radius come from** — authored reference data (safe) vs derived from
   Z/shell (a bonus keystone if the trend emerges). Recommend: author first, attempt derivation as a
   keystone later.
2. **Molecule formation search** — exact combinatorial (small molecules, exact minima) vs
   deterministic anneal (scales, approximate). Likely both: exact for ≤ N atoms, anneal beyond.
3. **Crystal packing selection** — rule-based from bond character (start) vs energy-minimized over
   candidate packings (richer, later).
4. **Lattice dimension for crystals** — 2D prototype (consistent with the materials engine) vs 3D
   (real coordination numbers, better band-gap revival). Start 2D, plan 3D.
5. **Identity & caching** — graph-canonical molecule ids; whether a compound's identity includes its
   synthesis route (cf. the materials engine's parked "process-as-identity" question).
6. **How much of the periodic table** — start with a curated ~20–40 atoms (enough for the keystones)
   and grow.

---

## 20. Risks & honest caveats (state these up front)

- **The C2 affinity-derivation is the make-or-break.** If deriving the current affinities from
  quantum descriptors can't reproduce M0–M8, integration means *re-validating* those milestones, not
  silently shifting them. Budget for that; treat any divergence as a finding to report, not bury.
- **Distilled ≠ first-principles.** This models the *category* of chemical behavior (stoichiometry,
  bond type, feasibility) from distilled parameters. It will not reproduce real reaction
  enthalpies/rates quantitatively, and we do not claim it does — the keystones are *qualitative
  correctness + parameter-free emergence*, the same bar as the materials engine.
- **Entropy is the weakest link.** ΔS from a coarse species/phase count is the least principled
  piece; flag every ΔG conclusion that hinges on it, and prefer reactions whose sign is robust to the
  ΔS estimate.
- **Combinatorial blow-up.** Molecule formation and reaction enumeration can explode; keep atom
  counts small, cache canonical forms, and gate expensive searches (the materials engine's standing
  lever: coarse stored values, the explorer as the accurate instrument).
- **Determinism across a graph search** is harder than across a lattice sweep — canonical ordering
  and a single seeded stream are non-negotiable (§14).

---

*This spec is implementable in a fresh context. Start at C0, de-risk each keystone with a throwaway
prototype, report the numbers (including negatives) before committing architecture, and keep the
determinism test green from C1 onward — exactly as M0–M8 were built.*
