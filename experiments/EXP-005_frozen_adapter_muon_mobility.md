# EXP-005: With the MCMC width FROZEN wide the whole run, does the muon still localise off-BC (⇒ optimisation artifact) or reach BC (⇒ sampling artifact)?

- **Date:** 2026-07-07   **Status:** DONE (verdict H0) — job `5534792`, 8 nodes, HOME env,
  fresh net into `pp_wide_burnin_frozen`; died at step ~3272 (node abort) but the site
  diagnostic is already conclusive; not resubmitted. Predecessor v3 wide-burnin
  run (`5529283`) cancelled; train/mcmc/base_config bytecode cleared before launch.
- **System:** diamond 2×2 supercell, BC-relaxed cage (`bc_relaxed/pp` geometry),
  16 C + 1 muon, doublet. Identical to EXP-004 `pp_wide_burnin_v3` **except the MCMC
  width is frozen for the whole run** (no adaptation) and the electrons start at
  `move_width=0.07` instead of 0.02. Muon init stays the default symmetric
  carbon-centred Gaussian (unseeded; `muon_init_coord = None`).

## Question / Motivation (why)
EXP-004 (`pp_wide_burnin_v3`) falsified H1: a **one-shot** wide muon init does not
un-trap the muon to BC. The muon-specific acceptance data (per-species `pmoves` from
the checkpoints) showed exactly why — at width 0.3 the muon accepts only **~25%**, which
is *below* the adapter's [0.50, 0.55] target, so the adapter did its job and divided the
width by 1.1 at nearly every one of the first ~14 events, collapsing 0.30 → 0.072 by step
~1800 (≈0.08 is the width that gives 50% muon acceptance in the sharpening ψ). The
ensemble committed to a single basin by step ~700 and the muon localised at a T-type
interstitial **3.2 bohr from BC** (0% of walkers within 2 bohr of BC ever).

So the wide init is unwound *before* basin selection completes. EXP-004's leftover
question: is that collapse the reason it misses BC (a **sampling** failure — fixable by
keeping the muon mobile longer), or would it miss BC even with unlimited mobility (an
**optimisation / representability** failure — ψ simply is not driven toward BC)?

**Slowing the adapter cannot answer this** — there is always a slower setting, so a
still-off-BC result at any finite slow-down is inconclusive (goalpost-moving). The only
informative single run on this axis is the **limit**: freeze the width wide for the whole
selection window and see what the optimiser does with a *permanently* maximally-mobile
muon. This experiment is the complement to EXP-002 (which biased the *init* toward BC and
showed the muon HOLDS BC when seeded, so BC is a real, holdable basin) — here the init
stays symmetric and only the *sampling* is held maximally mobile.

## Hypothesis (H1 — sampling)
With the muon proposal frozen at 0.3 bohr (a mobile, 2.6-bohr-equilibrated,
basin-hopping cloud — as measured at v3 step 0) for the entire selection window, the muon
visits BC repeatedly while ψ is still plastic; the variational energy then drives ψ toward
the deeper BC basin and the run localises the muon at **BC** (cubic-frac 0.125), converging
**below** the off-T/T reference (−90.669 Ha). ⇒ the off-BC trap was an early-time
sampling artifact after all; a *sustained* (not one-shot) wide proposal is the fix.

## Null hypothesis (H0 — optimisation / representability)
Even with a permanently-wide, maximally-mobile muon, ψ evacuates the muon to a
non-BC interstitial and holds it there, reproducing an off-BC site and ≈−90.669 Ha. ⇒
mobility is **not** the lever: the optimiser *chooses* non-BC on energetics because BC's
depth is a muon–electron (muonium) correlation the fresh net does not represent, so
visiting BC yields no gradient reward (chicken-and-egg). Then the remaining levers are an
**envelope-level bias on the muon** (shape ψ_fresh toward BC directly) or explicit seeding
(EXP-002) — **not** a slower adapter.

**Prior:** H0 remains the more likely outcome (unchanged from EXP-004), and this run is
primarily a **clean diagnostic** that closes the sampling-vs-optimisation question. A
frozen-wide failure is a *strong* (near-definitive) statement that mobility is not the
lever; a success flips EXP-004.

## Why freezing is safe (design notes)
- **Unbiasedness.** Metropolis samples the true |ψ|² for *any* proposal width; width
  affects variance, never correctness. So freezing at 0.3 cannot bias the energy or the
  wavefunction it builds — even within a localised site.
- **~25% acceptance is fine, not dangerous.** It is near the RWM efficiency optimum
  (~0.234, Roberts–Gelman–Gilks); the adapter's 0.50–0.55 target is the conservative side.
  And 0.3 bohr ≈ 1–2× the localised muon's per-axis density width (v3: per-axis σ ≈
  0.16–0.30), so it stays an efficient in-site sampler if/when ψ localises. Only acceptance
  so low the walker never moves (~<5–10%) would be a problem; 0.3 → ~25% is well clear.
- **Muon is 1 of 66 particles**, so any residual muon-sampling inefficiency barely touches
  the total energy/gradient variance.
- **Electrons frozen at 0.07** (near their v3-adapted 0.084 equilibrium) so all species
  sit in a healthy regime for the whole run; freezing them too is the benign direction
  (slightly-too-small ⇒ high acceptance ⇒ over-correlated but stable, never over-moving).

## Method / interventions (vs EXP-004 `wide_burnin.py`)
Copy of the v3 config; only the MCMC schedule differs.
1. **Freeze the width (zero code).** `cfg.mcmc.adapt_frequency = 100000`. The adapter's
   guard is `if t > 0 and t % adapt_frequency == 0`, so it **never fires** in the relevant
   window (first possible firing at step 100000, 20× past the ~5k selection window and past
   where these runs realistically die). Width stays fixed at its init for the whole run.
   - **Why 100000 and NOT ~1e9:** `pmoves` is allocated `np.zeros((nspecies,
     adapt_frequency))` (`train.py:997`); `1e9` ⇒ a (3, 1e9) array ≈ **24 GB** ⇒ OOM at
     startup. `100000` ⇒ a (3, 100000) ≈ **2.4 MB** buffer and still a full practical freeze.
2. **Wide muon, near-equilibrium electrons.** `muon_move_width = 0.3`, `move_width = 0.07`.
3. `burn_in = 2000` (unchanged) — equilibrate walkers to |ψ_fresh|² before optimisation.

### New config + job
- `configs/diamond/bc_relaxed/wide_burnin_frozen.py` — save dir
  `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_wide_burnin_frozen`
  (`restore_path = save_path`, empty on first launch ⇒ fresh net).
- `jobs/diamond_2x2/bc_relaxed/wide_burnin_frozen.sh` — 8-node launcher.

### Verification (done on CPU, 2026-07-07)
`get_config()` builds cleanly on the `ferminet-piku` env: `muon_move_width=0.3`,
`move_width=0.07`, `adapt_frequency=100000`, `burn_in=2000`, `muon_init_coord=None`,
`particles=(33,32,1)`. pmoves buffer (3, 100000) = 2.4 MB. **Zero adapter firings in the
first 5000 steps; first firing at step 100000.** (Full `train.train` deferred to the
compute node — a multi-particle run blows the login node's `ulimit -u` at XLA compile.)

### Pre-launch check (compute node, before trusting the run — same protocol as v3)
- Log: **"No checkpoint found. Training new model."** (fresh net — confound #1).
- Log: **"Initial MCMC width per species: [0.07 0.07 0.3]"**.
- `ckpt_000000`: muon width 0.3, electrons 0.07 (`tools/muon_width_check.py --particles 33,32,1`);
  muon walkers a broad diffuse cloud, not pinned.
- **Width STAYS [0.07, 0.07, 0.3] at ckpt_000100 … 002000 … 005000** (the freeze — the key
  new guardrail vs v3, where the muon width had already fallen to ~0.14 by step 800).

### Instrumentation (pass/fail signatures)
- `tools/muon_width_check.py` — width flat at 0.3 (muon) / 0.07 (e) throughout.
- `tools/muon_diffusion_check.py` — does the ensemble stay diffuse / reach BC, or collapse
  to a single non-BC basin *despite* the frozen-wide proposal?
- `tools/muon_site_analysis.py` on a mid-training checkpoint/inference — **H1** peak at BC
  (0.125); **H0** an off-BC interstitial (as in v3). Track VMC energy vs −90.669 Ha.

## Result
Ran as job `5534792`, fresh net, 8 nodes; **died at step ~3272** (node abort, the
usual pattern) after 84 min — E only at **−84.0 Ha** (far from the −90.67 reference,
ψ nowhere near converged), but the site/mobility diagnostic is already unambiguous.

**Freeze verified (the key guardrail held).** `tools/muon_width_check.py`: the MCMC
width is flat at **`[0.07, 0.07, 0.3]` for all 34 checkpoints** (step 0 → 3300) — the
adapter never fired, exactly as designed. Log confirms fresh net ("No checkpoint
found. Training new model.") and `Initial MCMC width per species: [0.07 0.07 0.3]`.

**→ H0 confirmed. The muon localises off-BC despite the permanently-wide proposal.**
`tools/muon_diffusion_check.py --a 6.74 --site BC=0.125,0.125,0.125`:
- Cloud collapses **2.78 bohr (step 0) → ~0.14–0.30 bohr by step ~500**, then flat.
- **0.0 % of walkers within 2 bohr of BC at *every* checkpoint**, including step 0.
- The collapse happens **early (step 200–500) while acceptance was still healthy
  (~0.15–0.24) and the cloud was 2.0–2.8 bohr** — i.e. it is ψ-driven, not a
  low-acceptance freeze.

**Single site, stable, and NOT a bond-centre.** Folding the walkers to a common
periodic image (the naïve Cartesian mean is a multi-image artifact — do not use it)
gives **one** tight crystallographic cluster at primitive-frac **≈ (0.845, 0.577,
0.330)**, rock-stable from step 300 on (drifts **≤ 0.10 bohr** vs its step-500
position through step 3300). It is an **open (T-type) interstitial cage**, not BC:

| step | folded prim-frac | spread (b) | nearest 3 C (b) | dist to ideal BC (b) | muon pmove |
|------|------------------|-----------:|-----------------|---------------------:|-----------:|
| 300  | (0.842,0.577,0.348) | 0.66 | 2.23 / 2.50 / 2.55 | 4.97 | 0.244 |
| 500  | (0.855,0.576,0.324) | 0.35 | 2.33 / 2.41 / 2.70 | 4.93 | 0.146 |
| 1500 | (0.852,0.576,0.334) | 0.27 | 2.26 / 2.45 / 2.62 | 4.93 | 0.069 |
| 3300 | (0.843,0.577,0.330) | 0.30 | 2.36 / 2.50 / 2.64 | 4.98 | 0.080 |

- Nearest carbons **~2.3–2.65 bohr** (a real BC sits **~1.46 bohr** from *two* bonded
  carbons); the muon is **1.9 bohr from the nearest C–C bond midpoint** ⇒ not in a
  bond. Caged by 3–4 C at 2.36–2.99 bohr; **~5 bohr from ideal BC**. Matches
  EXP-004 v3's "T-type interstitial", reproduced here from an independent fresh net.

**Per-species pmoves** (frozen widths `[0.07, 0.07, 0.3]`): electrons healthy
(~0.59–0.65 throughout); **muon acceptance falls to ~0.08** by step 1500 (design note
predicted ~0.25). See caveat below — the freeze delivered a fixed *proposal*, not
sustained realised *mobility*.

## Verdict
**H0 (optimisation / representability), not H1 (sampling).** With the muon proposal
frozen wide (0.3 bohr) for the whole selection window, ψ still evacuates the muon into
a single non-BC T-type interstitial cage (~2.4 bohr from its nearest carbons, ~5 bohr
from BC) by step ~500 and holds it there — **0 % BC occupancy at all times**, including
the diffuse initial cloud. So the off-BC trap is **not** an adapter/width-collapse
sampling artifact: keeping the proposal permanently wide does not change the outcome.
The remaining lever is an **envelope-level muon bias** (shape ψ_fresh toward BC
directly), **not** a slower/wider adapter — that axis is now exhausted. Consistent with
EXP-002 (BC *is* a real, holdable basin when the muon is seeded there), the fresh net
simply is not driven toward BC on energetics alone.

**Caveat (why this is strong but not the last word on mobility).** The muon acceptance
craters to ~8 % once ψ sharpens the cage to ~0.3 bohr (a 0.3-bohr proposal overshoots
a 0.3-bohr well), so *late* in the run the muon is not actually mobile — the frozen
width gave a fixed proposal, not sustained realised mobility. This does **not** weaken
H0 (the collapse is complete by step ~500, while acceptance was still ~0.15 and the
cloud 2–3 bohr wide, and even that wide cloud had 0 % BC weight), but a fully airtight
"sustained mobility" test would need a proposal that *tracks* the shrinking site (or a
site-hop proposal), not a fixed 0.3 bohr. That refinement is only worth doing if the
envelope-bias route also fails.

## Next / current steps
- **DONE:** verdict is H0 (see above). Sampling-mobility axis is now exhausted — do **not**
  slow/widen the adapter further, and do not resubmit this run.
- **Next lever (chosen): envelope-level muon bias.** Shape ψ_fresh toward BC directly (bias
  the muon envelope so visiting BC yields gradient reward from step 0), the test that
  separates "can't sample BC" from "can't represent BC". Complement to EXP-002 (seeding the
  *init*); this seeds the *representation*.
- Optional only-if-envelope-fails: a site-tracking / hop MCMC proposal to close the ~8 %
  late-run acceptance caveat above (a fixed 0.3-bohr proposal is not realised mobility once
  ψ localises).
