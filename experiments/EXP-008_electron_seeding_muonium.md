# EXP-008: Electron seeding — does seeding an up-electron on the muon bind silicon T-muonium, or wash out?

- **Date:** 2026-07-08   **Status:** PLANNED (design + rationale; not yet run)
- **System:** silicon 2×2×2 supercell, muon at the tetrahedral interstitial.
  **Primary vehicle: CLASSICAL muon** — fixed `'H'` nucleus at the T-site,
  65 electrons (33,32), PP on Si, a=10.26 bohr (clone of `t_relaxed_classical.py`
  geometry). Quantum muon (33,32,1) as the physically-complete follow-up.

## Why the CLASSICAL muon is the smarter vehicle
Use a fixed-H muon, not a quantum one, for this test — it isolates the physics and
gives a cleaner readout:
- **Removes the muon-localization half of the chicken-and-egg.** The muon is
  pinned at the exact T-site, so the *only* question left is the one we care about:
  can the electron bind to a +1 at the T-site? One variable, not two.
- **The classical muon is already an envelope centre.** It is an `'H'` atom in
  `cfg.system.molecule`, so a decaying orbital is anchored on it — the [[EXP-007]]
  representability gap is *partly closed for free*. (It's a *tight* hydrogenic
  envelope, so this cleanly tests envelope **shape** — silicon may need a diffuse
  one — rather than mere presence.)
- **Fixed origin ⇒ sharpest readout + a matched-geometry energy test.** The SRPD
  uses a fixed origin (the cleanest measurement, per [[EXP-006]]); and because the
  nuclei are fixed, **E(seeded/localized) vs E(diamagnetic) is apples-to-apples**
  at identical geometry — a direct test of whether muonium is even a competitive
  minimum, with no ZPM confound.
- **Baselines already exist.** Classical diamond-T binds *unseeded* (#7, +0.523);
  classical silicon-T is diamagnetic *unseeded* (#11) — same setup, muon fixed +
  envelope centre present in both. So the framework demonstrably *can* bind;
  electron seeding adds exactly one variable to the known silicon #11 baseline.

## Question / Motivation (why)
Silicon T-site is diamagnetic in every run we have, yet µSR sees a real *normal
muonium* Mu⁰_T there ([[EXP-007]] context). Two candidate causes for why our sims
miss it: **basin-selection** (the ansatz *could* represent muonium, but the joint
optimize-while-sample loop never discovers that basin — the muonium chicken-and-
egg: no up-density at the muon early ⇒ gradients never learn to bind it) vs
**representability** (the plane-wave PBC envelope has no muon-localized component,
so a bound orbital can't be built at all — [[EXP-007]]).

**Electron seeding is the cheapest possible probe and a clean discriminator
between the two** — it needs **zero ansatz changes**, only initial-walker
placement, reusing the existing muon-seeding machinery. Seed one up-spin electron
on the muon in a fresh-net run and watch what happens:
- **If the seeded bound state HOLDS** (a stable T-contact net-spin excess that
  survives past burn-in and lowers/matches the energy) ⇒ the blocker was
  **basin-selection** — a cheap win, no ansatz surgery needed.
- **If it WASHES OUT** (excess decays to ~0 within the burn-in) ⇒ the blocker is
  **representability** ⇒ the [[EXP-007]] muon-anchored envelope is required.

Either outcome is decisive and cheap.

## Why electron seeding is a *weaker* lever than muon seeding (and why that's the point)
Muon seeding (EXP-002) works to *hold* BC because the muon is a **distinguishable,
heavy** particle: once the net localizes it, its walkers stay put. An up-electron
is **indistinguishable** — seeding a slot is identity-free (antisymmetry washes out
which slot), and the seed only sets *initial* positions, not the stationary |ψ|².
So a seeded up-electron **relaxes back to |ψ|² on the first MCMC sweep unless the
wavefunction itself develops density at the muon.** Hence: electron seeding can
only *hold* if the ansatz can already (at least marginally) represent the bound
state and merely needed the optimizer nudged into that basin. That dependency is
exactly what makes it a clean test of "basin vs representability."

## Hypothesis (H1)
Seeding an up-spin walker coordinate on the **fixed** muon lets the optimizer lock
into a muonium basin that **holds after burn-in**: classical silicon-T develops a
stable T-contact net-spin excess (g↑>g↓, `localized_excess` plateau ≳+0.3) at
**energy ≤ the diamagnetic #11 run at the identical geometry** — i.e. the earlier
diamagnetic result was a basin the optimizer missed.

## Null hypothesis (H0)
The electron seed **washes out** and/or **E(localized) > E(diamagnetic)** at the
fixed geometry: the localized muonium orbital is not a competitive minimum in this
cell (the electron prefers the conduction band), so basin-selection is *not* the
blocker — the tight-H envelope and/or the 2×2×2 cell are ⇒ EXP-007 (diffuse
envelope, larger cell). *(Honest prior: given classical-Si #11 is already
diamagnetic with the H envelope centre present, H0 is the more likely outcome —
but the matched-geometry energy test makes it definitive either way.)*

## Alternative explanations (confounds) & controls
- **"Washed out" ≠ "ansatz can't represent it"** — could instead be an optimizer
  schedule / too-short basin-formation window. *Controls:* (a) **positive control**
  — classical diamond-T already binds *unseeded* (#7, +0.523), confirming the
  classical framework can hold muonium; electron-seeded diamond-T must also hold.
  If it doesn't, the seeding machinery/schedule is broken, not the silicon physics.
  (b) **held-seed variant** — re-apply the electron seed for the first ~2–5k steps
  (not just t=0) to give the basin longer to form, distinguishing "washes out
  instantly" from "optimizer just needed longer." (c) **matched-geometry energy** —
  the decisive control unique to the classical vehicle: if E(localized) >
  E(diamagnetic #11) at identical fixed nuclei, muonium simply isn't a competitive
  minimum here — independent of whether any seed "holds."
- **Indistinguishability** — seeding slot 0 (an up electron) is identity-free by
  antisymmetry ([[reference-vmc-basin-trapping-seeding]]); not a confound, only
  noted.
- **Spurious imprinted state** — a held bound state must be **energetically
  favourable** (E ≤ the unseeded diamagnetic run at matched steps) and **stable
  after the seed is no longer applied**, not an artefact of initial conditions.

## Method (experiment detail)

**Implementation (mirror the muon-seed path exactly — small, gated, stock runs unchanged):**
- `base_config.py` (near the existing `muon_init_coord`/`muon_init_width` at
  ~L255-257): add `mcmc.electron_init_coord = None`, `mcmc.electron_init_width = 0.5`.
- `init.py:init_electrons` (mirror the muon block at L102-108): after the
  atom-centred init, if `electron_init_coord is not None`, re-seed **slot 0** (an
  up-spin electron — `_assign_spin_configuration` makes slots `0..particles[0]-1`
  spin-up) to `electron_init_coord + N(0, electron_init_width)`:
  ```python
  if electron_init_coord is not None:
      ep = electron_positions.reshape(batch_size, sum(electrons), ndim)
      key, subkey = jax.random.split(key)
      ep = ep.at[:, 0, :].set(jnp.asarray(electron_init_coord)
              + jax.random.normal(subkey, (batch_size, ndim)) * electron_init_width)
      electron_positions = ep.reshape(batch_size, sum(electrons) * ndim)
  ```
- `init.py:init_mcmc_data` (L144-154): thread `cfg.mcmc.electron_init_coord`
  /`electron_init_width` into the `init_electrons` call.
- (held-seed variant only) a small hook in `mcmc.py` to re-apply the slot-0 seed
  for the first N steps.

**Run (classical, primary):** fresh-net **classical** silicon-T — fixed `'H'`
muon at the T-site `(0.75,0.75,0.75)·a`, 65 electrons — with `electron_init_coord`
= the T-site (an initial *muonium* config: up-electron sitting on the fixed muon),
width ~0.3–0.5 b. New config `configs/silicon/t_classical_electron_seeded.py`
(clone of `t_relaxed_classical.py` + the electron seed) + SLURM launcher. Read the
**fixed-origin T-contact SRPD + `localized_excess(r)` vs step from checkpoints**
([[project-bound-state-formation-dynamics]] tooling — no inference dump; verdict
readable from ~50k per [[EXP-006]]) **and the block-averaged energy** vs #11.

**Controls / comparators:** electron-seeded classical diamond-T (positive control;
classical diamond-T binds even unseeded, #7); **unseeded classical silicon-T #11**
as the matched-geometry baseline for excess **and energy** (the key comparison).

**Discriminator logic:**
- seed holds **and** E ≤ #11 ⇒ **basin-selection** was the blocker (cheap win) —
  then repeat with the quantum muon for the physically-complete result.
- seed washes out **or** E(localized) > E(#11) ⇒ muonium isn't a competitive
  minimum in this cell ⇒ **envelope-shape / finite-size** ⇒ [[EXP-007]] (diffuse
  muon envelope, 3×3×3 cell), with electron seeding as its companion.

**Follow-up (quantum):** if classical binds, port the electron seed to the quantum
muon (`t_seeded.py` + `electron_init_coord`, both muon and electron seeded at T) to
recover the ZPM-correct contact density / hyperfine.

## Result
_(pending — not yet run)_

**Success criteria:** fixed-origin T-contact net-spin excess with g↑/g↓>1 and a
`localized_excess` plateau ≳+0.3 that **survives past burn-in / seed release**, at
**E ≤ the diamagnetic #11 run at the identical geometry**.

## Verdict
_(pending)_

## Next / current steps
- [ ] Add `mcmc.electron_init_coord`/`_width` (base_config) + slot-0 re-seed
      (init.py) + threading (init_mcmc_data); gate so `None` ⇒ stock behaviour.
- [ ] `configs/silicon/t_classical_electron_seeded.py` (fixed-H + electron seed at
      T) + launcher.
- [ ] Positive control: classical diamond-T + electron seed (machinery holds).
- [ ] Checkpoint-sweep fixed-origin T-contact SRPD + **matched-geometry energy vs
      #11**.
- [ ] Verdict: holds & E≤#11 ⇒ basin (then port seed to the quantum muon);
      washes out or E>#11 ⇒ EXP-007 (diffuse envelope / 3×3×3).
- [ ] (Optional) held-seed variant to rule out a too-short basin window.

## Relation to prior experiments
The cheapest stage-1 lever of [[EXP-007]] (bind silicon T-muonium), separated out
because it is conceptually distinct (basin-selection, *not* representability) and
needs **no ansatz change**. Builds on [[EXP-002]] (muon seeding HOLDS BC — but the
muon is distinguishable, so this electron analogue is a weaker, more diagnostic
lever), [[EXP-006]] (verdict readable from ~50k), and
[[reference-vmc-basin-trapping-seeding]] (seeding = initialization, not cheating;
identity-free by antisymmetry).
