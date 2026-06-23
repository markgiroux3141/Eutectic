# M7 — Spectral / Band Gap (revival spec)

## 0. Purpose & why now

M7 classifies a material as **conductor / semiconductor / insulator** from a **band gap** measured
off its lattice (materials spec §5.6): build a tight-binding Hamiltonian, take eigenvalues, read the
gap at the Fermi level. It was de-risked on 2026-06-15 and **falsified** before any code landed —
not by a tuning failure but by the substrate (memory `m7-spectral-deferred`):

- *The keystone itself passed:* `eigvalsh` of a tight-binding `H` with a staggered ±Δ on-site
  potential on a **full** periodic square lattice recovers `gap = 2Δ` to ~1e-14, **independent of
  the hopping `t`** — the textbook ionic gap, parameter-free.
- *The killer was dilution:* the materials engine's `combine` lattices are ~0.63 fill. A staggered
  potential giving a clean `gap = 2.0` at fill 1.00 gives **`gap = 0.0` already at fill 0.95** — 5%
  vacancies are dangling bonds = mid-gap states that fill the gap and push DOS(E_F) above the
  clean-metal value. Both candidate gap mechanisms (on-site stagger *and* dimerized hopping) died to
  the same defect physics. This is physically correct (amorphous Si is useless without H
  passivation; Si metallizes on melting), so it was accepted as a real negative, not fought.

**What changed:** the deferral note said revival needs *"a substrate a real gap can live on —
near-full-fill 2-sublattice (ionic/covalent) crystalline order, and/or the 3D target lattice."* The
chemistry layer now **produces exactly that.** `chemistry/crystal.py` emits **dense (fill = 1.0)**
crystals with `atom_type` sublattices — `_ionic_crystal` is a rock-salt checkerboard of alternating
cation/anion sublattices; element/metallic crystals are uniform. The chemistry engine spec
anticipated this precisely (§2): *"Real compounds have real unit cells — a tight-binding/Hückel pass
over a covalent-network crystal can give an honest semiconductor/insulator gap. M7 is a natural C2+
deliverable."* The substrate the gap needs now exists. This spec revives M7 on it.

## 1. The one principle (unchanged)

> **Measure properties from a structure. Never assign them from a function.**

The gap is **measured** from the lattice's Hamiltonian, never assigned. The thing that makes one
crystal an insulator and another a metal is the **structure chemistry built** (a charge-staggered
rock-salt lattice vs. a uniform metallic one), read out by one generic spectral extractor.

## 2. The model — tight-binding band gap from a `Lattice`

For a settled `Lattice` with `N` occupied sites, build the `N×N` real-symmetric Hamiltonian:

- **Hopping** `H[i,j] = -t` for every nearest-neighbour pair of occupied cells (the same periodic
  nearest-neighbour adjacency `engine/properties/percolation.py:conducting_mask`/`label_clusters`
  already walk — reuse it; periodic wrap, as the kernels do). `t` is a single fixed scale.
- **On-site energy** `H[i,i] = ε_i`, the per-site potential (see §3). Uniform `ε` ⇒ no ionic gap;
  staggered `ε` on a bipartite sublattice ⇒ a gap.
- Eigenvalues by `numpy.linalg.eigvalsh` (real-symmetric; deterministic).

**Filling / Fermi level.** Take the natural tight-binding reference of **half-filling** (one spinless
carrier per two sites — lower half of the spectrum filled). The gap is then the HOMO–LUMO gap
between eigenvalue `N/2` and `N/2 + 1`. (Filling is a *reference choice*, stated, not a dial; the
classification is robust to it for the staggered-bipartite case because the gap sits at mid-band.)

**Two gap mechanisms, both real:**

- **Ionic (on-site stagger).** A bipartite lattice with `ε = +Δ` on one sublattice and `−Δ` on the
  other has dispersion `E(k) = ±√(Δ² + t²|f(k)|²)`; the gap between the two bands is exactly `2Δ`,
  independent of `t`. This is the de-risk keystone — and the chemistry rock-salt crystal is exactly
  this bipartite ±Δ structure. **Achievable on the existing 2D substrate now.**
- **Covalent (hopping dimerization / SSH).** A uniform-site covalent network (Si, diamond) has no
  ionic stagger; its gap is the bonding–antibonding splitting from **bond alternation** — strong/weak
  hopping `t₁/t₂` opens `gap = 2|t₁ − t₂|`. On a 2D square lattice there is no natural sp³
  dimerization to encode this faithfully; this mechanism wants either an explicit per-bond hopping
  pattern or the **3D coordination substrate**. **This is the harder tier (§8).**

## 3. Where Δ comes from (parameter-free, from chemistry)

The on-site stagger must be **emergent**, not a dial. The ionic gap is set by the
**electronegativity difference** between the two sublattice species: a large ΔEN (Na vs Cl) localizes
electrons on the anion → large `Δ` → wide gap → insulator; ΔEN ≈ 0 (a metal, or a homonuclear
network) → `Δ = 0` → no gap → conductor. So:

> `Δ_site ∝ (χ_site − χ̄)` — each site's on-site energy is its species' electronegativity relative
> to the cell mean. The proportionality is **one fixed scale constant** (calibrate once against a
> textbook gap if desired; it does not change the *ordering* or the conductor/insulator
> *classification*, only the absolute eV scale — the same "one calibrated constant" stance as the
> C1 bond energy and the C3 entropy units).

The cleanest plumbing (mirrors how `cohesion`/`metallicity` were added, see §9): `chemistry/crystal.py`
sets a new **optional per-cell `site_potential` field** from species electronegativity; the spectral
extractor reads it. Non-chemistry lattices (elements, `combine` blends) default `site_potential = 0`
→ no gap → conductor, which is the correct and already-validated behaviour for those.

## 4. The detector & finite-size honesty

A raw HOMO–LUMO gap is **not** an honest detector: a metal's discrete gap → 0 like `1/N` (level
spacing), so on a finite lattice every material shows *some* gap. The de-risk established the honest
detectors — use one (or both, cross-checked):

- **`gap / mean-level-spacing`** near E_F: a true gap is `N`-independent while spacing `~ bandwidth/N`,
  so the ratio **grows with `N`** for an insulator and stays `O(1)` for a metal. Confirm by sweeping
  lattice size.
- **`DOS(E_F)`** (density of states at the Fermi level, broadened): zero in a gap, finite in a metal.
  Watch the overlap with the M2/M3 percolation axis (the deferral note flagged this) — DOS(E_F)
  conflates "gapped" with "disconnected"; on a dense crystal that's fine, but keep them distinct.

## 5. Classification

`gap ≈ 0` → **conductor**; small gap → **semiconductor**; large gap → **insulator** (materials
§5.6). Thresholds are on the *normalized* gap, not the raw one. **Keystone ordering target:** `Cu`
(metallic, Δ=0) conductor; `Si`/covalent semiconductor (small gap — tier-2, §8); `NaCl` (large ΔEN)
insulator. The classification must fall out of chemistry (ΔEN + bond character), not authored labels.

## 6. Conditions coupling (the gap as a phase property — handle with no-fudge care)

The docs want a *conditions* property: **the gap closes as T/P opens vacancies / melts the crystal**
(real Si metallizes on melting). This is the valuable, harder half — and exactly where the no-fudge
norm bites, because **"gap closes with T" is not automatically a transition.** Two candidate framings;
the de-risk must decide which is honest, and **if the closing is a smooth defect crossover, say so —
do not dress a `gap < cutoff` boolean as a phase transition** (this is the C4-ignition / M3-SC lesson):

- **(a) Thermal vacancies.** Run the occupancy lattice-gas (`engine/process.py` with
  `evolve_occupancy=True`, the M5 kernel) on the dense crystal under a heating schedule; vacancy
  fraction rises with T (activated `~exp(−cohesion/T)`), and — per the de-risk — even a few percent of
  vacancies collapse the gap. Likely a **steep but smooth crossover**, anchored on the vacancy density,
  *not* a sharp transition. Report it as such.
- **(b) Sublattice-order collapse (a genuine transition, if reframed).** If the ionic crystal is set
  up so the ±Δ stagger rides the M5 **staggered-density order parameter**, then melting (the M5
  order-disorder transition) averages the stagger to zero → the gap collapses **at the M5 melting
  point** — a real transition, with two coincident detectors (gap collapse + M5 staggered-density
  collapse + the textbook melting value), the canonical keystone pattern. This requires reconciling
  the dense (fill = 1.0) crystal with M5's half-fill commensurate order — a design choice to make in
  the de-risk.

**Anchor the milestone on the static keystone (§7.1), which is rock-solid; treat the conditions
coupling as the extension and characterize it honestly (transition vs. crossover) rather than
claiming a transition by default.**

## 7. Keystones (the no-fudge ladder)

1. **Static, parameter-free (the anchor).** On the chemistry `NaCl` rock-salt crystal, the
   tight-binding gap `= 2Δ` with `Δ` derived from ΔEN; recovered to numerical precision on the full
   (fill = 1.0) lattice, **independent of `t`**. `Cu` (uniform, Δ=0) shows no gap. The
   conductor/insulator split is emergent from chemistry.
2. **Finite-size honesty.** `gap / level-spacing` grows with `N` for `NaCl`, stays `O(1)` for `Cu`
   (the only honest gap detector; raw gap is rejected).
3. **Classification ordering.** Insulator (NaCl) > semiconductor (covalent, tier-2) > conductor (Cu),
   ordered by ΔEN / bond character, parameter-free.
4. **Conditions (characterized honestly).** The gap closes on heating — reported as either the M5
   melting transition (framing b, a real coincident-detector transition) or an activated vacancy
   crossover (framing a, explicitly **not** a transition). Whichever the de-risk supports.

## 8. The 2D-vs-3D fork (tiering)

- **M7a — ionic gap on the current 2D substrate (de-riskable & shippable now).** The rock-salt
  ±Δ keystone (§7.1–§7.3) transfers directly to `chemistry/crystal.py`'s 2D dense crystals. No new
  substrate. This is the recommended first deliverable.
- **M7b — covalent semiconductors / 3D (the stretch).** Faithful sp³ bonding-antibonding gaps for
  Si/diamond need bond-alternation (SSH) hopping or real tetrahedral coordination — i.e. the
  `DEFAULT_SHAPE_3D = (16,16,16)` substrate (already a named constant in `engine/lattice.py`). Larger
  lift; the materials spec's "3D target lattice." **Decide with the user whether to attempt M7b or
  ship M7a and defer covalent gaps**, surfacing it as an `AskUserQuestion` fork.

## 9. Module layout / integration touch-points

Mirror the established extractor shape (HANDOFF.md "Process for a new milestone"):

- **`engine/properties/spectral.py`** (new) — `band_gap(lattice) -> float` (normalized gap),
  `dos_at_fermi(lattice)`, `classify(lattice) -> str`. Builds `H` from periodic NN adjacency + a
  per-site potential; `eigvalsh`; honest detector. Reuses `percolation` adjacency helpers. Generic:
  reads lattice fields only, knows nothing about chemistry.
- **`engine/lattice.py`** — add an **optional** `site_potential: np.ndarray | None` field, defaulting
  (in `__post_init__`) to zeros, **excluded from `structural_signature`** exactly as `cohesion`/
  `metallicity` are (it is a derived field; keeping it out of the signature is what preserves M0–M8
  byte-identical). The spectral measurement folds the `site_potential` hash into *its own* seed.
- **`chemistry/crystal.py`** — set `site_potential` from species electronegativity (relative to the
  cell mean) in `_ionic_crystal` / `element_crystal` / `compound_crystal`. This is the only place
  chemistry feeds the gap; the extractor stays generic.
- **`engine/material.py:measure_properties`** — add `band_gap` (and a `material_class` string if
  wanted) to the stored dict, quantized. Gate by cost if `eigvalsh` on the full lattice is slow
  (it's `O(N³)`; a 64×64 lattice is N=4096 → ~seconds — consider a smaller spectral sub-shape or gate
  like Curie/melting are gated; **report the suite-time cost in the de-risk**, suite time is a
  standing concern).
- **`tools/explorer.py`** (and/or `tools/chem_explorer.py:measure-compound`) — a `spectral` /
  `band-gap` view: the spectrum, the gap, the normalized detector vs `N`, and (if §6) gap(T).
- **Tests** — `tests/test_spectral.py`: the parameter-free `2Δ` keystone, the finite-size detector,
  the classification ordering, determinism, **and re-validation that M0–M8 stay byte-identical**
  (the new optional field must not perturb any existing measurement or signature).

## 10. Risks & honest caveats (state up front)

- **The original falsification is the headline risk.** The whole revival bet is that chemistry's
  **fill = 1.0** crystals dodge the dangling-bond death. The **first de-risk number** must be: does the
  gap survive on the *actual* `chemistry/crystal.py` NaCl lattice (after `relax`), not just an
  idealized full lattice? If the relaxed crystal or the 2D square geometry reintroduces gap-killing
  disorder, report it and stop — same discipline as last time.
- **Conditions coupling may be a crossover, not a transition** (§6). Do not claim a transition unless
  a coincident-detector keystone (framing b) actually holds. A smooth `gap(T)` decline is an honest
  result to report, not to dress up.
- **Covalent gaps likely need 3D** (§8). The 2D substrate cleanly delivers the *ionic* gap; Si/diamond
  semiconductors are the part that genuinely waited for the 3D target.
- **DOS(E_F) overlaps the percolation axis** — keep "gapped" distinct from "disconnected" (dense
  crystals make this clean, but the detector choice must be deliberate).
- **Absolute eV scale is uncalibrated** (one fixed `Δ`-scale constant), like every prior layer; the
  emergent claims are *existence of the gap*, the *t-independence*, the *N-scaling*, and the
  *ΔEN-ordered classification* — qualitative correctness + parameter-free emergence, the standing bar.

## 11. De-risk plan (run BEFORE any architecture — report numbers incl. negatives)

A throwaway `derisk_m7.py`, mirroring the prior de-risk, reporting:

1. **Substrate survival (the make-or-break):** build the real `chemistry/crystal.py` NaCl crystal,
   relax it, map `atom_type` → ±Δ via ΔEN, build `H`, `eigvalsh`. Does `gap = 2Δ` survive on the
   actual relaxed crystal? Compare to `Cu` (Δ=0 → gap 0).
2. **t-independence:** vary `t`, confirm the ionic gap is unchanged (parameter-free).
3. **Finite-size detector:** sweep lattice `N`; confirm `gap/level-spacing` grows for NaCl, flat for
   Cu; confirm raw gap is misleading.
4. **ΔEN ordering:** a few binaries spanning ΔEN; confirm gap rises with ΔEN (classification ordering).
5. **Conditions (if attempting §6):** introduce vacancies / run M5 occupancy on the crystal vs T;
   plot `gap(T)`; **decide and report**: transition (framing b, coincident with M5 Tc) or smooth
   vacancy crossover (framing a). Nudge any cutoff and check whether the "closing T" slides (the
   disguised-dial test).
6. **Cost:** time `eigvalsh` at the production shape; report whether it needs a sub-shape or gating.

Only after the numbers are in (and the substrate-survival negative is *not* triggered) build the
extractor.

## 12. Determinism (inherits materials spec §6 / chemistry §14)

`eigvalsh` is deterministic. Fixed site ordering (row-major over occupied cells). Quantize the stored
gap before storing/comparing. The new `site_potential` field is excluded from `structural_signature`
(so combination/measurement seeds and all M0–M8 stored values stay byte-identical) and folded only
into the spectral measurement's own seed. A "measure the gap twice → identical" test from the start.
