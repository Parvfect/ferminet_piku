---
name: project-muon-mcmc-width-diffusion
description: "Checkpoint audit of the off-T trap: steady-state muon width is HEALTHY (not collapsed); real cause is early-time width-adaptation lag — muon diffuse only at init, collapses to one basin by step 2000 and never re-diffuses"
metadata: 
  node_type: memory
  type: project
  originSessionId: 45574b8e-1bf1-4bf0-addd-c9df4bc0eb43
---

Audit (2026-07-06) of the diamond `bc_relaxed/pp` off-T trap
(`/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp`) via `tools/muon_width_check.py`
and `tools/muon_diffusion_check.py` (reads walker positions from checkpoints;
`JAX_PLATFORMS=cpu` needed to unpickle `data.positions`; muon = last particle,
particles=(33,32,1)). Refines [[project-bc-relaxed-pp-relax-2-muon-result]] and
EXP-001's open "why does the optimiser select off-T" question.

**Hypothesis A — steady-state muon proposal width collapses below electrons →
FALSIFIED.** Per-species adapted muon width settles at **0.24 bohr, ~2.85× the
electron width (0.084)**, muon acceptance a healthy **54%**. Same in every run
(diamond bc_seeded/unrelaxed/t_relaxed, silicon bc_seeded/unrelaxed): muon/e ratio
2.4–3.1, acceptance 51–54%. Naive 1/√m≈0.07 is wrong — width tracks *acceptance*,
not mass; one muon in a smooth interstitial tolerates a bigger step than cusped
electrons. Adapter is NOT pinning the muon. (Handoff's original mechanism is dead.)

**Hypothesis B — muon never reaches a diffuse multi-basin state → CONFIRMED, this
is the real mechanism.** Muon walker-ensemble spread over training: step 0 (random-C
init) ~5.5 bohr (spans the cell) → **step 2000 = 0.15 bohr (collapsed ~37×)** →
0.20 (10k) / 0.33 (40k) / ~0.46 (120k→end, intra-basin only). Diffuse ONLY at init;
commits to one basin within the first ~2000 optimiser steps and NEVER re-diffuses.

**Cause = width-adaptation LAG.** Defaults: `move_width=0.02`, `adapt_frequency=100`,
`burn_in=100`×`steps=10`, and width does NOT adapt during burn-in
(`train.py:~1172` `update_mcmc_width` is training-loop only). So the muon proposal is
~0.02–0.08 bohr through the first ~2000 steps — too small to hop basins — exactly
while KFAC localises ψ fastest. Width reaches its healthy 0.24 only near ~100k, long
after ψ zeroed amplitude outside one basin; endpoint-only Metropolis then rejects any
hop into an empty basin, so a bigger late width can't help.

**Bottom line:** off-T IS a sampling/init artifact (consistent with EXP-001 H0-reject:
BC is true site), but a *transient early-time* one (proposal lags ψ), not steady-state
width collapse. Explains why seeding ([[project-bc-seeded-muon-result]], EXP-002)
works.

**Refined mechanism + fix implemented (2026-07-06, EXP-004).** Deeper analysis: ψ_fresh
is the carbon-centred envelope (σ=π=1) × smooth net → peaks on carbons; interstitial
tails mildly favour BC, so neither init nor ψ_fresh favours off-T. off-T wins by KINETIC
accessibility — during the muon's evacuation off the carbons, tiny local moves reach only
the open T-cage; BC sits behind the C1 watershed. Fix now IMPLEMENTED as a new
`base_config.mcmc.muon_move_width` knob (default None; when set + sample_all=False,
train.py starts the MUON ONLY wide, electrons keep move_width=0.02, each species adapts
its own width). Chosen over the original uniform-0.3 + adapt-during-burnin plan — smaller
diff, no electron-instability confound, no burn-in-loop surgery. Config
`configs/diamond/bc_relaxed/wide_burnin.py` (muon_move_width=0.3, burn_in=2000, unseeded,
fresh net) + `jobs/diamond_2x2/bc_relaxed/wide_burnin.sh`. Verified on CPU (per-species
width application + adaptation + config build); full train.train run blocked on login
node (ulimit) → pre-launch check owed on compute node (fresh-net line + initial muon
width 0.3). **Updated prior: H0 (re-collapse to off-T) is the LIKELY outcome** — BC's
depth is a muonium correlation the fresh net hasn't learned, so wide sampling may not
install a BC-preferring force; run EXP-004 as a diagnostic. If H0, stronger lever is an
envelope-level muon bias or seeding. See experiments/EXP-004 for full detail.

**DEPLOYMENT BUG — first three EXP-004 runs are INVALID (2026-07-06).** All three wide-burnin
runs (`pp_wide_burnin`, `pp_wide_burnin_dense`, `pp_wide_burnin_v2`) have ckpt_000000 muon
width = **0.02, not 0.3** (checked with `tools/muon_width_check.py --particles 33,32,1`) — the
muon_move_width intervention never actually ran, so they just reproduce the off-T conditions.
Cause was NOT `adapt_during_burnin` (no such flag exists; burn-in uses null_update and never
adapts width) and NOT the config (all print muon_move_width=0.3). It was a **stale-code /
timing** problem: the `muon_move_width` block in `train.py` (currently UNCOMMITTED, working-tree
only) either post-dated the launch (v1 06:00, dense 07:33 both predate the 07:50 train.py edit)
or was masked by stale `__pycache__/train.cpython-312.pyc` on the compute node (v2 launched
07:50:59, 53s after the edit, yet its stdout has no "Initial MCMC width per species" guardrail
line → ran old code). Git commit status is irrelevant to SLURM (python imports train.py from
disk); the lesson is to clear bytecode and pre-check the width log on a compute node before
trusting a run. Fix: cleared train pyc, verified current code on CPU with the REAL wide_burnin
config (optimizer='none', taskset -c 0-1 + `--xla_cpu_multi_thread_eigen=false` to dodge the
login-node pthread/ulimit abort) → logs **`Initial MCMC width per species: [0.02 0.02 0.3]`** ✓.
Production config now points at fresh **`pp_wide_burnin_v3`**; a single-node verify job
(`configs/.../wide_burnin_verify.py` + `jobs/.../wide_burnin_verify.sh`, burn_in=50, throwaway
`pp_wide_burnin_verify`, SLURM 5518646) is queued to confirm width on a real node before the
8-node v3 launch.

**EXP-004 RESULT — H1 FALSIFIED (2026-07-07, at step 10 200 of `pp_wide_burnin_v3`).** The valid
v3 run reproduced the trap: muon width 0.30 (step 0) collapsed to **0.072 by step 1800** (adapter
÷1.1 every 100 steps unwinds the 0.3 init in ~1500 steps — briefly *below* the electron width at
steps 1600–3400) then slow re-growth to 0.116 (10 200). Ensemble diffuse only at init (spread
2.6–3.1 b, steps 0–400) → **collapsed to ~0.15 b by step ~700**, flat after; BC occupancy 5.5%→0%
by step 300. Site (checkpoint-snapshot `muon_site_analysis` on ckpt_010200): a **tetrahedral
interstitial**, cubic-frac ≈(0.95,0.55,0.95), 4 C at ~2.6–3.0 b — **3.2 b from BC, 0% walkers <2 b
of BC**; also 4.6 b from the `bc_relaxed/pp` off-T (0.474)³ pocket (a *different* T-cage, not BC and
not the old off-T). Tight blob RMS 0.38 b. Energy −89.41±0.12 (block-avg step≥8000), still
descending, above off-T ref −90.669. **Verdict: proposal width alone is NOT the lever** — a
one-shot wide init is unwound by the adapter before basin selection, and wide sampling installs no
BC-preferring force (consistent with BC being an unlearned muonium correlation). Next levers:
**EXP-005** = blunt the width adapter (keep muon wide through the whole selection window, not just
init — the ×/÷1.1-every-100-steps cadence is the culprit); failing that, an envelope-level muon bias
or seeding (EXP-002).

**EXP-005 built (2026-07-07, not yet launched) — FROZEN adapter / max-mobility test.** Decisive
complement to EXP-004: instead of slowing the adapter (goalpost-moving — always a slower setting),
FREEZE the width wide for the whole selection window and see what the optimiser does with a
*permanently* mobile muon. Logic: still off-BC ⇒ OPTIMISATION/representability artifact (not
sampling) ⇒ next lever is envelope-level muon bias, NOT a slower adapter; reaches BC ⇒ sampling was
the lever (flips EXP-004). Config `configs/diamond/bc_relaxed/wide_burnin_frozen.py` + job
`jobs/diamond_2x2/bc_relaxed/wide_burnin_frozen.sh`, fresh net → `pp_wide_burnin_frozen`. Knobs:
`muon_move_width=0.3`, `move_width=0.07` (electrons near their 0.084 eq), `burn_in=2000`, freeze =
**`adapt_frequency=100000`** (ZERO code: adapter guard `t>0 and t%freq==0` never fires before step
100000, 20× past the ~5k selection window). **Gotcha:** do NOT use ~1e9 — `pmoves` is
`np.zeros((nspecies, adapt_frequency))` at train.py:997, so 1e9 = ~24 GB OOM at startup; 100000 =
2.4 MB and still a full practical freeze. ~25% muon acceptance at width 0.3 is FINE (near RWM
optimum 0.234; Metropolis unbiased for any width; 0.3 ≈ 1–2× the localised per-axis density so
still an efficient in-site sampler). CPU-verified: get_config builds, zero adapter firings <5000
steps. Pre-launch check owed on compute node: fresh-net line + `Initial MCMC width per species:
[0.07 0.07 0.3]` + width STAYS frozen at ckpt 100/2000/5000. Prior unchanged: H0 (still off-BC)
more likely. See experiments/EXP-005.

**EXP-005 RESULT — H0 CONFIRMED (2026-07-07). Off-BC trap is an OPTIMISATION/representability
artifact, NOT sampling.** Job `5534792` (fresh net, `pp_wide_burnin_frozen`) died at step ~3272
(node abort) at E=−84.0 Ha (ψ far from converged) but the diagnostic is unambiguous; not
resubmitted. **Freeze verified:** width flat `[0.07,0.07,0.3]` for all 34 ckpts (adapter never
fired). **Result:** despite the PERMANENTLY wide muon proposal, the ensemble collapsed 2.78 b
(step 0) → ~0.14–0.30 b by step ~500 and localised in ONE non-BC cage; **0.0% walkers within 2 b
of BC at EVERY checkpoint incl. step 0**. The collapse is complete by step ~500 while acceptance
was still healthy (~0.15) and the cloud 2–3 b wide ⇒ ψ-driven, not a low-acceptance freeze. So
freezing the proposal wide does NOT change the outcome: proposal mobility is not the lever. Next
lever = **envelope-level muon bias** (shape ψ_fresh toward BC), NOT a slower/wider adapter — that
axis is now exhausted. Site is a single stable **T-type open interstitial** (prim-frac ≈
(0.845,0.577,0.330), drifts ≤0.10 b step 300→3300; nearest C ~2.3/2.5/2.65 b; ~5 b from ideal BC;
NOT a bond-centre — 1.9 b off the nearest C–C midpoint), reproducing EXP-004 v3's T-cage from an
independent fresh net. **Per-species pmoves** (frozen `[0.07,0.07,0.3]`): electrons healthy
~0.59–0.65; **muon acceptance craters to ~0.08** by step 1500 (design predicted ~0.25) — CAVEAT:
a fixed 0.3-b proposal overshoots the ~0.3-b well, so the freeze gave a fixed *proposal* not
sustained realised *mobility*; doesn't weaken H0 (collapse done early while acceptance healthy) but
an airtight mobility test would need a site-tracking/hop proposal. See [[reference-vmc-basin-trapping-seeding]].

**GOTCHA — muon site from checkpoints: FOLD to a common periodic image first.** The naive Cartesian
mean of muon walker positions lands in a meaningless spot when walkers occupy symmetry-equivalent
images (it averaged to ~1.38 b from a C, wrongly suggesting BC; the true folded site is 2.36 b).
Fold each walker's fractional coord near the circular mean (`fr - round(fr - circ_mean)`) THEN
average → true single site. `muon_diffusion_check.py`'s circ_std is fold-safe; a raw `.mean(0)` is not.
