# EXP-004: Does a wider muon proposal + longer burn-in un-trap the muon (off-T → BC) without seeding?

- **Date:** 2026-07-06   **Status:** implemented, ready to run (compute-node pre-launch check pending)
- **System:** diamond 2×2 supercell, BC-relaxed cage (`bc_relaxed/pp` geometry),
  16 C + 1 muon, doublet. Identical to `bc_relaxed/pp` in every respect **except the
  muon MCMC proposal width and burn-in length** — the muon init stays the default
  (symmetric random-carbon Gaussian; *not* seeded at BC, unlike EXP-002).

## Question / Motivation (why)
The EXP-001 follow-up (2026-07-06 checkpoint audit; `tools/muon_width_check.py`,
`tools/muon_diffusion_check.py`) localised the off-T trap to a **transient early-time
sampling failure**, not the steady-state width collapse originally suspected:

- Steady-state muon proposal width is *healthy*: ~0.24 bohr, ~2.85× the electron
  width, ~54% acceptance. (Hypothesis "muon width collapses below electrons" —
  falsified.)
- But the muon walker ensemble is diffuse only at the random init (~5.5 bohr) and
  **collapses to a single ~0.15 bohr blob within the first ~2000 optimiser steps**,
  then never re-diffuses.
- Cause: **width-adaptation lag.** Defaults are `move_width=0.02`,
  `adapt_frequency=100`, `burn_in=100`, and the width does not adapt during burn-in.
  So through the first ~2000 steps the muon proposal is stuck at ~0.02–0.08 bohr —
  far too small to keep walkers spread — exactly while KFAC localises ψ fastest.

**Refined mechanism (from the design discussion, 2026-07-06).** Working through what a
fresh net actually produces sharpened the picture and lowered our prior on success:

- **ψ_fresh is set by the envelope, not "flat".** At init the network factor is a
  smooth O(1) modulation; the shape of ψ_fresh is the isotropic envelope
  `Σ_carbons π·exp(−σ·r)` with `π=σ=1` (`envelopes.py:103-122`). So |ψ_fresh|² **peaks
  on the carbons** and decays ~1 bohr into the interstitial. The muon is enveloped by
  the carbons (it is not itself an envelope centre).
- **Neither the init nor ψ_fresh favours off-T.** The default init
  (`init.py:79-96`) is a carbon-centred Gaussian; and in the interstitial tails
  |ψ_fresh|² is, if anything, mildly *larger* at BC (~2 carbons at ~1.5 bohr) than at
  off-T (~4 carbons at ~2.9 bohr). off-T is favoured by neither amplitude nor energy.
- **off-T wins by KINETIC accessibility.** As optimisation begins, the muon sits where
  E_L is huge (on the carbons) and ψ evacuates it into the interstitial. With
  0.02–0.08 bohr local moves the only reachable escape from a carbon is the **open
  tetrahedral cage → off-T**; reaching BC means moving toward/through the C1 carbon
  watershed (EXP-001), where proposals are rejected. Basin selection happens *during*
  this evacuation, and small steps channel it into off-T regardless of BC being deeper.
- **The intervention's job** is therefore to keep the muon proposal wide **through the
  selection window** so walkers can reach and hold BC (which ψ_fresh already tails
  into), giving BC a chance to accumulate gradient signal before ψ commits.

This is the complement to EXP-002: EXP-002 biases the init toward BC; EXP-004 leaves the
init symmetric and only widens the muon's sampling so the optimiser can choose freely.

## Hypothesis (H1)
The off-T collapse is caused by the early muon proposal being too narrow to reach BC
during the evacuation-off-carbons window. With the muon starting at its healthy width
(~0.3 bohr) and a long burn-in that equilibrates walkers to |ψ_fresh|² before ψ is
sharpened, the muon ensemble stays able to visit BC through the first several thousand
optimiser steps; the variational energy then drives ψ toward the deeper basin, and the
run localises the muon at **BC** (peak at cubic-frac 0.125, not 0.474) and converges
**below** the off-T run's −90.669 Ha. ⇒ off-T was an early-time sampling artifact,
fixable without seeding.

## Null hypothesis (H0)
A wide muon proposal makes no lasting difference: the ensemble re-collapses to a single
basin during early training and the run reproduces the off-T site and ≈−90.669 Ha. ⇒
single-peak selection is intrinsic to the variational optimiser — most plausibly because
BC's depth is a **muon–electron (muonium) correlation the fresh net has not learned**, so
BC never registers a low local energy for the gradient to reward no matter how well the
chain mixes (chicken-and-egg). Then proposal width is *not* the lever and only explicit
seeding (EXP-002), or an envelope-level bias on the muon, can place it at BC.

**Prior:** after the mechanistic analysis we consider **H0 the more likely outcome**, and
treat this experiment primarily as a **clean diagnostic** that separates "off-T = an
early-time sampling artifact" from "off-T = intrinsic single-peak selection." Either
result is informative.

## Alternative explanations (confounds to control)
- **Warm-start contamination.** Restoring any `bc_relaxed/pp` (off-T) checkpoint
  reintroduces the off-T-peaked ψ. → Fresh net, brand-new empty `save_path`,
  `restore_path = save_path`; verify the "Training new model" log line; re-check after
  every afterany restart.
- **Big initial width destabilises the electrons.** *Eliminated by design:* the wide
  proposal is applied to the **muon only** (`muon_move_width`), so the electrons keep
  their usual `move_width=0.02` and never over-move. (The original uniform-0.3 plan is
  abandoned precisely to remove this confound.)
- **Diffuse-but-committed illusion.** The spread metric can stay high while ψ has already
  picked a basin. → Corroborate with a mid-training inference + `muon_site_analysis.py`,
  never the spread alone.
- **Spontaneous symmetry breaking / seed dependence.** A single run landing on either
  basin could be luck. → Interpret with EXP-001's open multi-seed question; run 2 seeds
  if resources allow.
- **PBC wrapping inflates the spread reading.** `muon_diffusion_check.py` uses circular
  statistics, but a boundary-straddling single basin can still read "large". → Use the
  per-site occupancy columns and inference, not just `spread_bohr`.

## Method (experiment detail)
Fresh-network FermiNet training, carbons frozen at the BC-relaxed positions, muon init
left at the **default** symmetric carbon-centred Gaussian (`muon_init_coord` stays
`None`). All settings copied verbatim from `bc_relaxed/pp` **except** the two MCMC knobs.

### Interventions (vs `bc_relaxed/pp`)
1. **Wide muon-only proposal** — `cfg.mcmc.muon_move_width = 0.3` (≈ the muon's healthy
   adapted width; electrons stay at the default `move_width=0.02`). The muon keeps a
   diffuse, BC-reaching ensemble from step 0; each species still adapts its own width.
2. **Long burn-in** — `cfg.mcmc.burn_in = 2000` (×`steps=10` = 20 000 moves). ψ (the
   fresh net) is *frozen* during burn-in (`train.py` builds `burn_in_step` with
   `null_update`), so this is a pure equilibration phase: walkers relax to the true
   |ψ_fresh|² (including its BC-leaning interstitial tails) before the optimiser starts
   sharpening ψ.

### Code change (implemented)
New config field **`cfg.mcmc.muon_move_width`** (default `None`, `base_config.py`).
When set and `sample_all` is `False`, `train.py` starts the muon (last particle) at that
width while the electrons keep `move_width`; each species adapts independently thereafter
(`mcmc.mh_update` applies a per-species width; `mcmc.update_mcmc_width` adapts each
species separately). Default `None` ⇒ the width-init branch is byte-identical to before,
so **all existing configs are unaffected**.

*Abandoned from the original draft:* uniform `move_width=0.3`, an `adapt_during_burnin`
flag, adapting the width inside the burn-in loop, and the associated `max_width` hoist.
Per-species init achieves the same "muon wide early" goal with a smaller, lower-risk diff
and removes the electron-instability confound entirely.

### New config + job
- `configs/diamond/bc_relaxed/wide_burnin.py` — copy of the `bc_relaxed/pp` training
  config with `muon_move_width = 0.3`, `burn_in = 2000`, a fresh net, and a new save dir
  `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_wide_burnin`
  (`restore_path = save_path`, empty on first launch → fresh net).
- `jobs/diamond_2x2/bc_relaxed/wide_burnin.sh` — copy of the diamond pp launcher pointed
  at the new config.

### Verification (done, 2026-07-06)
Verified on CPU (`ferminet-piku` env) — see scratchpad `test_per_species_width.py`:
- Width-init logic: `muon_move_width=0.3` widens the muon only (`[0.02, 0.02, 0.3]`);
  `None` reproduces the uniform vector exactly.
- `mcmc.mh_update` applies the width **per species** — muon RMS displacement is 25× the
  electrons' at a 25× width ratio (matches `width·√nsteps`).
- `mcmc.update_mcmc_width` grows/shrinks **each species' width independently**.
- `get_config()` builds cleanly (no ConfigDict type error), correct values,
  `muon_init_coord=None` (unseeded); `from ferminet import train` imports.
- **Not yet run:** the full `train.train()` in situ — an actual multi-particle run blows
  the login node's `ulimit -u` (XLA `pthread_create` at compile). Deferred to the
  compute-node pre-launch check below (it is where the job runs anyway).

### Pre-launch check (on the compute node, before trusting a full run)
- Log says **"No checkpoint found. Training new model."** (fresh net — the #1 confound).
- Dump the initial `mcmc_width`: **muon = 0.3, electrons = 0.02**.
- Dump initial muon walkers: default carbon-centred spread (**not** BC).

### Instrumentation (pass/fail signatures)
- **`tools/muon_diffusion_check.py`** on the new `save_path`
  (`--a 6.74 --particles 33,32,1 --site offT=0.474,0.474,0.474 --site BC=0.125,0.125,0.125`):
  **H1** — spread stays large past step 2000 and settles on BC; **H0** — collapses to
  ~0.15 bohr as in `bc_relaxed/pp`.
- **`tools/muon_width_check.py --last`** — muon stayed wide early, electrons stayed ~0.02
  → ~0.08 (rules out any electron-instability effect).
- **`muon_site_analysis.py`** on a mid-training inference — peak at BC (0.125) vs off-T
  (0.474); track VMC energy vs −90.669 Ha.

## Result
_(pending first run)_

## Verdict
_(pending)_

## Next / current steps
- Compute-node pre-launch check (fresh-net line + initial muon width 0.3), then launch.
- Monitor with `tools/muon_diffusion_check.py` / `tools/muon_width_check.py` from the
  first checkpoints — the early-spread trajectory is the fast read on H1 vs H0.
- **If H1 (un-traps to BC):** the off-T trap is an early-time sampling artifact; replicate
  on a second seed and on silicon.
- **If H0 (re-collapses to off-T):** single-peak selection is intrinsic → proposal width
  is not the lever. The stronger levers are then an **envelope-level bias on the muon**
  (shape ψ_fresh toward BC directly, not just the walkers) or seeding (EXP-002); this also
  settles EXP-001's open "why off-T" question in favour of representability, not sampling.
