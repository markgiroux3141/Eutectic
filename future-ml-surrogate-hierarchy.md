# Future Exploration — ML Surrogate Hierarchy

**Status:** speculative / parked idea. Not on the current milestone path (M0–M6 in
`materials-engine-spec.md`). Captured here so it isn't lost; revisit after the explorer
shows the deterministic property space "feels good."

---

## 1. The idea

Simulating physics at the lowest quantum level is compute-prohibitive. Proposed alternative:
a **hierarchical surrogate** approach.

1. Simulate the lowest level we care about and generate a large dataset of interactions.
2. Train a neural net to predict that level.
3. Use that net as the **generator** for the next level up.
4. Repeat — each layer's surrogate feeds the next.

Because the goal is *not* to replicate known physics, some drift is acceptable — and might
even introduce interesting new behavior.

This is a real, named field: **ML surrogate modeling / learned coarse-graining / multiscale
ML**. Prior art to mine: neural operators (FNO, DeepONet) for PDEs; ML interatomic
potentials (MACE, NequIP) that learn DFT and run MD ~1000× faster; neural cellular automata;
diffusion/GAN field generators.

---

## 2. The two big risks

### 2.1 Phase transitions — and why this is really §1 in disguise

The engine's founding principle (`materials-engine-spec.md` §1) is:

> **Measure properties from a structure. Never assign them from a function.**

Emergence (percolation, Ising, the superconductivity double-threshold) lives in **sharp
thresholds in the substrate**. A smooth function approximator smears those thresholds into a
learnable lookup table — exactly the "hash problem" §1 was written to avoid.

A naive surrogate `(parent_A, parent_B) → property_scalar` **is the function §1 forbids.** An
MLP regressor will:

- smooth the percolation knee,
- fatten the band gap,
- and worst of all, **regress the rare tails toward the mean.**

Superconductors-as-a-thin-tail is the entire payoff loop (§5.4). Tail collapse is the single
most reliable failure mode of regression surrogates. So a careless version of this idea
quietly deletes the thing that makes it a game.

### 2.2 Compounding drift (the risk the original framing missed)

"Some drift is fine" is true for **novelty** but dangerous as stated. In a stacked
hierarchy, each layer trains on the *previous layer's predictions*, not ground truth. Errors
compound autoregressively. The usual endpoint is not "interesting new behavior" — it's
**distribution collapse**: variance shrinks, modes merge, tails vanish.

We must distinguish:

- **Good drift** — new structure, novel correlations, behavior the deterministic engine
  can't produce.
- **Bad drift** — washed-out thresholds, collapsed rarity, regression to the mean.

They look identical until the explorer's (§7) property histograms go flat. Treat drift as a
**tunable knob**, not an accident we tolerate.

---

## 3. The fix that keeps §1 intact: predict the *field*, not the *property*

Don't have the net output `conductivity = 0.7`. Have it output a **settled lattice** (the
field), then run the existing cheap measurement extractors (§5) on the generated structure.

Why this is the key move:

- The **threshold stays in the measurement, not the net.** Percolation / min-cut / Laplacian
  on a generated field is still a hard, discontinuous test — sharpness for free even from a
  smooth generator.
- It is faithful to §1: still *measuring a structure*; the net only replaces the expensive
  `merge → relax` step, never the physics-bearing `measure` step.
- This is what neural CA, GANs, and **diffusion models** already do. Diffusion in particular
  captures multimodal, sharp-boundary distributions far better than MLP regression.
- If we ever do go scalar anyway: classification head for the *phase* + regression for the
  *value*, or a mixture-density output — never a single smooth regressor.

---

## 4. Reframe — what is our "lowest level," really?

We **already escaped** the quantum-compute problem: the substrate is a deliberately cheap toy
Ising/percolation lattice (§2). So a surrogate is not buying escape from QM. The honest,
valuable uses are:

1. **Shipping.** Open question §9.5 / §2 asks whether we port scipy to TS/Rust for the
   in-game runtime. A learned surrogate of `combine()` answers it: train on the Python
   engine, ship a small net that runs `combine()` in real-time in the browser with no scipy
   port. **Probably the strongest practical reason to build this.**
2. **Scale.** Bigger lattices / 3D / instant combinations where real relax+measure is too
   slow for interactive play.
3. **Active-learning hybrid (best of both).** Use the net as a cheap *proposer* for the
   boring bulk of material space; fall back to the **real** deterministic pipeline near
   critical points and in the rare-tail region. Spend real compute only where fidelity
   matters and where the net is least trustworthy. Bonus: where net and ground-truth
   disagree becomes a built-in drift monitor and a retraining signal.

Data generator already exists: the explorer (§7) emits N deterministic, labeled combos.

---

## 5. Recommended first prototype (if/when we pick this up)

Build a **field-generating surrogate of `combine()`'s merge+relax**, keeping the real
measurement extractors downstream:

- Input: parent lattices / signatures. Output: a settled child lattice.
- Architecture candidates: neural CA, diffusion model, or FNO-style operator. Avoid plain
  MLP scalar regression.
- Run the existing §5 extractors on the generated lattice for properties.
- Gate every iteration with the explorer's distribution checks:
  - Is the rare tail (superconductors) preserved?
  - Is the percolation/Ising threshold knee still sharp?
  - Are strength/ductility still anti-correlated?
- Expose drift as a tunable, not a side effect.

---

## 6. Open question to resolve before building

Does the surrogate **replace** the deterministic pipeline at runtime (shipping / scale play),
or run **alongside** it as a parallel "what-if" universe that is *allowed* to drift into novel
physics the deterministic engine can't produce?

- **Replace** → demands fidelity; the active-learning hybrid (§4.3) and field-generation
  (§3) are mandatory.
- **Alongside** → actively *wants* the drift; the bet flips from "stay faithful" to "explore
  novel emergent behavior the toy substrate can't reach."

Different bets → different first prototypes. Decide this before writing code.
