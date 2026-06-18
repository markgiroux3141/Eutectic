# Eutectic — Handoff / pick-up guide

A deterministic emergent-materials engine. Read this first in a new context, then the files it
points to. It captures orientation, conventions, the non-obvious learnings, and what's next — it
does **not** duplicate the per-milestone physics writeups (those live in `README.md` "findings"
sections and `docs/conditions-and-properties.md`).

## Current status (as of this handoff)

- **Branch:** `main`. **HEAD:** `232e4c7` (machine layer — motor); **machine *suite* (4 more machines) built on top, uncommitted** (awaiting the go-ahead).
- **Tests:** 162 passing (`.\.venv\Scripts\python.exe -m pytest`), ~75s. Determinism test must always stay green.
- **Done:** M0 foundations → M1 combine pipeline → M2 density/percolation → M3 magnetism + (retired) SC proxy → M4 conditions + Curie point → M5 thermal occupancy/melting → M6a honest superconductivity (XY/BKT) → M6b thermal conductivity → M8 mechanical (strength = shear modulus + ductility = coordination deficit; the anti-correlation falls out) → **machine layer (spec §8): a thin role/requirement framework (`machines/roles.py`) + a suite of FIVE machines consuming `Material.properties`, one per property family — `motor.py` (torque), `electromagnet.py` (lift ∝ I²), `cable.py` (transmission), `heatsink.py` (carbon wins — the diamond divergence), `armor.py` (a 2-role composite that SOLVES the M8 strength↔ductility dilemma). Shared coil math in `_electrical.py`. Requirements emerge from the equations, not gates; the engine never imports `machines`**. Plus the process layer (synthesis as a trajectory). **M7 (spectral / band gap) was de-risked and DEFERRED** — falsified on the ~0.6-fill substrate (5% vacancies close any gap); see README "M7 (deferred) findings".
- **Read in this order to get oriented:** `materials-engine-spec.md` (§1 = the ONE principle), `README.md` (status + every "M_ findings" section — these are the honest-scope record), `docs/conditions-and-properties.md` (the conditions/transport plan; §6 = validation discipline, §7 = milestone sequence).

## The two cultural non-negotiables

1. **The ONE principle (spec §1):** *measure properties from a structure, never assign them from a function.* A material **is** a lattice (a Hamiltonian); properties are ensemble/graph **measurements** of it. Every milestone obeys this.
2. **The no-fudge norm** (memory `no-fudge-norm`, docs §6): de-risk every emergent claim with a throwaway prototype and **report the numbers (including negatives) before committing architecture**. Pressure-test against known physics: recover a textbook value with **no free parameter** where one exists; hunt for "disguised dials" (a rate that slides smoothly as you nudge a threshold = a fake transition, not a real one — this is why the M3 SC percentile and the M6 leaned-SC snapshot were rejected). Honesty over green tests.

The recurring **keystone pattern**: a transition is real when an order parameter collapse, a detector, and a textbook value all coincide. For Curie (M4) and melting (M5) the detector is the **heat-capacity peak** `C(T)=Var(E)/(N T²)`. **Important exception:** for the BKT superconducting transition (M6a) the C-peak sits *above* Tc — the detector there is the **helicity-modulus universal-line crossing** `Υ=(2/π)T`. Don't reuse the C-peak detector blindly.

## Architecture in one screen

- **Structure vs state (docs §2):** a `Material` (engine/material.py) is a settled `Lattice` + measured `properties` + lineage. The lattice is the *permanent structure*; a property is `measure(structure, conditions)` — an ensemble observable, not a frozen number. Reference snapshot stored at standard conditions; full curves measured on demand (the explorer).
- **`Conditions(T, P, H)`** (engine/conditions.py): the dial-space. T drives all ensembles; H is the magnetic field; **P is live as of M5** (chemical-potential offset for the occupancy lattice gas).
- **Three deterministic checkerboard Metropolis kernels in engine/lattice.py** (all share the parity-split trick so vectorized == sequential):
  - `metropolis_sweep` — Ising **spin** (magnetism, M3/M4).
  - `occupancy_sweep` — non-conserved repulsive **lattice gas** (melting, M5).
  - `xy_sweep` — continuous-angle **XY phase** (superconductivity, M6a).
- **engine/thermal.py** is THE ensemble-measurement engine: spin ensemble + `curie_temperature` (M4); occupancy ensemble + `melting_point` (M5); XY ensemble + `superconducting_tc` (M6a). All seed one `SplitMix64` from structure signature + quantized conditions.
- **Per-cell lattice fields** (parallel numpy arrays): `occupied`, `atom_type`, `spin`, `mass`, `moment` (spin coupling, M3), `cohesion` (bond stiffness → melting, M5), `metallicity` (charge-carrier quality → electrical/SC gating, M6b). **`cohesion` and `metallicity` are deliberately excluded from `structural_signature`** (they're deterministic functions of fields already hashed, so combination seeds — and all M0–M4 stored values — stay byte-identical; measurements that *use* them fold the field hash into their own seed).
- **engine/process.py:** synthesis as a *trajectory* through conditions-space (anneal/quench/field-cool). `run_process` threads live state through `Stage`s using the same kernels. `evolve_occupancy=True` (M5) also evolves occupancy. `STANDARD_PROCESS` reproduces `relax` byte-for-byte (a pinned guard).
- **engine/properties/percolation.py:** the mask split is load-bearing (M6b) — `conducting_mask` = occupied ∧ metallic (**charge**: electrical, resistance, edge-connectivity, SC); `solid_mask` = occupied (**matter**: `spanning_fraction`/`largest_cluster_fraction`, melting solidity, phonons). Keeping solid measures on `occupied` is what made melting byte-identical through the M6b change.
- **tools/explorer.py:** the §7 verification harness. Views: `inspect`, `combine`, `distribution`, `percolation-sweep`, `magnetism-sweep`, `connectivity-sweep`, `temperature-sweep` (Curie), `melting-sweep` (M5), `sc-sweep` (M6a), `transport` (M6b electrical-vs-thermal), `process-compare`.

## Environment & workflow (Windows / PowerShell)

- venv: `.\.venv\Scripts\python.exe -m pytest` and `.\.venv\Scripts\python.exe -m tools.explorer <cmd>`. Deps: numpy/scipy/matplotlib/pytest.
- **Use the Bash tool for pytest/explorer** — the `.\.venv\...` path mangles under bash if you're not careful; in practice `./.venv/Scripts/python.exe ...` from the repo root works in the Bash tool. (Avoid the PowerShell tool for these.)
- **Explorer stdout must be ASCII** — the Windows console is cp1252 and will crash on `≈ · ρ ψ Υ ⟨⟩`. Use unicode only in matplotlib labels, never in `print()`.
- **Determinism:** only `engine/rng.py` SplitMix64; fixed iteration order; quantize before storing; `engine/` never imports game/UI.
- **Commit only when the user says so.** On Windows, write the message to a file and `git commit -F <file>` (inline here-strings mangle the subject). Footer:
  ```
  Co-Authored-By: Mark Giroux <mgmarkgiroux@gmail.com>
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
  (Project convention: commit straight to `main`; that's how M1–M6 landed.) `out/` is gitignored (explorer plots). De-risk prototypes are throwaway — delete before committing.
- **Suite time** is a standing concern (the user has asked to keep it down). Ensemble sweeps are the cost. Lever used repeatedly: gate expensive measurements (Curie by magnetism, melting by solidity) and keep stored values deliberately *coarse* (lean sweeps), with the explorer as the accurate instrument.

## Honest-scope cheatsheet (what's genuinely emergent vs. caveated)

Full detail in README "findings". The caveats you must not forget:
- **Curie (M4):** keystone exact on a full lattice (C-peak at textbook 2.269); real diluted materials show *broadened* transitions; stored Tc is coarse.
- **Melting (M5):** chosen model is **crystalline order-disorder** (repulsive lattice gas at half-filling, staggered-density order parameter) — order lost at *fixed density*, the signature the user explicitly chose over sublimation. It's a *continuous* β-brass-style transition, **not** first-order with latent heat. Measured at commensurate (half) filling. The cooling-rate→grain-size process signal is weak (a reported negative); the robust process payoff is **density under a pressure schedule**.
- **Superconductivity (M6a):** real BKT Tc, parameter-free keystone (0.893·J on a clean lattice). **Measured ON DEMAND, not stored per material** — XY/BKT critical slowing-down needs long equilibration (burn≈300+) for every conductor; a leaned stored sweep was under-equilibrated/burn-in-sensitive (rejected by the no-fudge norm). The static k-edge-connectivity proxy is **retired** (`edge_connectivity` kept as the structural coupling input). A Wolff XY cluster update would make a stored Tc cheap.
- **Thermal conductivity (M6b):** Wiedemann-Franz κ_e=L·T·σ is the single-carrier content (imposed, not an emergent prediction — said so). The emergent results are the **diamond divergence** (carbon: σ=0, top κ) and phonon κ tracking stiffness/mass. Charge gating is binary metallic/not (real semiconductors await M7).

## What's next (roadmap)

- **Machine layer** (spec §8) — **DONE**, now a *suite* of five machines (`machines/{roles,_electrical,motor,electromagnet,cable,heatsink,armor}.py`; explorer `motor`/`heatsink`/`cable`/`electromagnet`/`armor` views; `tests/test_motor.py` + `tests/test_machines.py`). See README "Machine layer findings". Natural follow-ups if revisited: **multi-material composites per role**, a **conditions-coupled machine** (performance over `Conditions`, not just a bespoke operating point), or wiring SC `Tc` in (needs a cheap stored Tc first — the Wolff-update option). Otherwise → the **game shell**.
- **Game shell** (spec §11 M6): the interactive interface (inventory, crafting UI, building, progression). The engine + explorer already generate-and-view today; this is the separate front-end effort.
- **M8 follow-ups (optional enrichments, not blockers):** activate **stress σ as a condition** (strain → nonlinear response / fracture), the conjugate that currently rides inert like P did pre-M5; add **bond-bending (angular) forces** to lower the shear-rigidity threshold toward percolation (our materials are currently *marginally* rigid, so absolute moduli are small).
- **M7 — Spectral / band gap: DEFERRED** (de-risked, falsified on the diluted substrate). Revive only on a crystalline/3D substrate or as a conditions-driven property (gap closing with T/P). See `memory/m7-spectral-deferred.md` and README "M7 (deferred) findings".
- **Parked (don't start without the user):** the **anisotropy + hysteresis** block (real coercivity / the neodymium payoff — true hard-magnet behaviour, currently only kinetic remanence); making **`process` part of material identity** + registry caching-by-process (currently id derives from lineage only).
- **Tier-3 "do NOT hack into core"** (docs §4): radioactivity (authored cause, emergent consequences), pH/corrosion (needs a chemistry layer). Flagged as separate, high-hack-risk.

## Process for a new milestone (what's worked 3× now)

1. Propose the plan + the de-risk design **before** writing code; surface genuine forks to the user (AskUserQuestion) — physics-model choices and anything that disrupts earlier milestones.
2. Write a throwaway `derisk_*.py`, run it, **report the numbers** (recover the textbook value parameter-free; confirm the detector; show the emergent dependence; report negatives). Only then build.
3. Mirror the established shape: per-cell field on the lattice (out of signature if derived) → kernel in lattice.py → ensemble/measurement in thermal.py → gated stored property in material.py → explorer view → tests (keystone + determinism + re-validate touched milestones) → README/docs findings → commit when told → delete throwaway.

## Memory pointers (auto-loaded; verify before trusting)

`MEMORY.md` indexes them. Key ones: `no-fudge-norm`, `conditions-architecture`, `superconductivity-status` (now: real XY/BKT Tc, proxy retired), `m4-curie-status`, `process-layer-architecture`, `commit-conventions`. Memories are point-in-time — verify file/line claims against current code.
