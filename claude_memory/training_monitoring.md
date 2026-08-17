---
name: training-monitoring
description: "Daily routine to summarize energy/variance of the 6 production FermiNet jobs, check squeue, and offer to (re)queue any not running"
metadata: 
  node_type: memory
  type: project
  originSessionId: 70601531-6df8-419a-80eb-fefbe0cf7f7a
---

## Daily job tracking & queuing routine

The user runs a daily check on 10 production SLURM training jobs (1-day time
limit, `#SBATCH --time=1-00:00:00`, so they get `CANCELLED DUE TO TIME LIMIT`
roughly once per day and need resubmission).

**Why:** Automating "check yesterday's results, queue today's runs" so the
user doesn't have to manually tail logs / run squeue / resubmit each day.

**How to apply:** When the user asks for a daily status check (phrasing like
"check yesterday's jobs", "daily update", "status check"), without
re-asking for paths:

1. Run `squeue -u $USER -o "%.10i %.30j %.8T %.10M %.6D %R"` and match job
   NAMEs against the table below to see which configs are PENDING/RUNNING.
2. For each config, tail the output log (last ~50-100 lines) and find lines
   matching `INFO:absl:Step (\d+): (\S+) E_h, exp. variance=(\S+) E_h\^2`.
   Energy and variance can independently go `nan` on some steps — report
   the most recent non-`nan` value for each (with its step number), and
   flag explicitly if variance has been `nan` for a sustained run (possible
   numerical issue, seen previously in the silicon unrelaxed-q run).
3. Present a summary table: config, last step, energy (E_h), variance
   (E_h^2), squeue status.
3a. **For every ACTIVE run, ALSO report a per-run table of the CLEAN energy
   change since the last check** (user confirmed 2026-06-29 after I misreported
   #14 as "+0.035 increase" — see below). Columns: run, last-check step, current
   step, Δsteps, clean E at last check, clean E now, ΔE (clean). "Last check" step
   = the step recorded for that run in [[current-status]] from the previous
   session.

   ⛔ **NEVER compute an energy CHANGE from the per-step `Step N: … E_h` log
   prints** — those are INSTANTANEOUS, single-sample, ±0.03 E_h noisy; comparing
   the first vs last printed step gives garbage (this is exactly the #14 mistake:
   raw endpoints −90.6714→−90.6363 looked like +0.035 UP, while the block-averaged
   run was flat at ≈−90.649). Use the `.out` `Step N:` prints ONLY to read the
   step RANGE (span): `grep -oE 'Step [0-9]+:' <log> | sed -n '1p;$p'`.

   ✅ **For every energy NUMBER, use block-averaged values from the run's
   `train_stats.csv`** (cols `step,energy,ewmean,ewvar,pmove`): block-average the
   raw `energy` col over a window (≥1000 steps) at the start-of-span and at the
   current step, report mean ± SEM; cross-check against the `ewmean` (smoothed)
   col and the [[tool-energy-convergence]] plateau. If a chunked block-avg
   trajectory shows a noisy excursion only in the last ~1–2k steps (doubled SEM),
   say so explicitly and quote the full-span mean/median, not the blip.

   Caveats to STILL state: (a) a FRESH net (restore==save, first line
   `Step 00000:`) starts at pre-training warm-up garbage (e.g. −38/−23 E_h), so
   its ΔE just reflects climbing out of init, not a physical gain; (b) a
   just-resubmitted run with only ~hundreds of steps so far has a ΔE dominated by
   post-restart settling noise — flag low-confidence.
3b. **ALWAYS also include the diamond AND silicon GSE energy comparison tables**
   ([[project-diamond-pp-energy-comparison]], [[project-silicon-pp-energy-comparison]])
   as part of the status report — the user confirmed (2026-06-25) they want both
   comparisons EVERY time they ask for today's training status, without having to
   ask separately. Refresh only the ACTIVE runs (reuse frozen/converged rows);
   use the robust-recent estimate for runs the convergence tool mishandles
   (#9 silicon, warm-up t_relaxed runs) per those memories. **Both comparison
   tables MUST include a Step (iteration) column for every run** (user confirmed
   2026-06-25) — current step for active runs, final/frozen step for stopped ones.
4. For any config NOT found in squeue, ask the user whether to submit it
   (`sbatch <script>` run from the "Submit from" directory below — the
   `#SBATCH --output=` path is relative to the submission directory, and the
   script itself `cd`s into the repo for the actual run).

### Queuing same-day follow-ups while jobs are still RUNNING

If the user wants today's continuation queued *before* a job hits its
time limit (so there's no gap), do NOT submit a plain `sbatch` for a config
that's still RUNNING — `cfg.log.save_path` and `--output=` are fixed,
non-timestamped paths, so two concurrent instances of the same config would
race on the same checkpoint files and `train_stats.csv`, and the new job
would truncate the log the running job is actively writing to.

**Why:** User confirmed this approach (2026-06-14) after asking whether
queuing today's jobs alongside still-running ones was risky.

**How to apply:** Submit a dependent job instead, from the config's "Submit
from" directory:
```
sbatch --dependency=afterany:<currently_running_jobid> <script>
```
Use `afterany` (not `afterok`) — these jobs always end via `TIMEOUT` or
`CANCELLED`, never `COMPLETED`, so `afterok` would never fire. The dependent
job sits `PENDING` until the running one terminates (any reason), then
starts cleanly with no overlap. This covers exactly one extra run — once
both finish, repeat the process for the next day.

Caveat: `afterany` also fires on a manual `scancel` or an early crash, so if
the user cancels a running job to debug it, its queued follow-up will start
immediately.

When submitting multiple of these via the Bash tool in one batch, use
absolute paths in subshells (`(cd /abs/path && sbatch ...)`) — a bare `cd
relative/path && sbatch` changes the persistent shell cwd, breaking
sibling parallel calls that assume the original cwd.

### Active vs closed (as of 2026-06-29)

The others are **closed** (do NOT auto-resubmit closed ones during the daily
check — only resubmit the active set, and mention closed ones only if the user
asks to reopen them).

2026-07-01: **ALL 6 active jobs DEAD, queue EMPTY.** The last follow-ups FAILED with
exit `6:0` (task/node abort), NOT clean TIMEOUT, all within ~08:23–09:20 UTC ⇒ a
cluster/node event, not per-run bugs. Ckpts + train_stats intact. **User said don't
submit new jobs → NOT resubmitted.** To relaunch later: plain `sbatch <script>` from each
config's submit dir (resumes from latest ckpt). Latest energies (all NOT converged) in the
two comparison memories; highlights: **★ #14 diamond bc_seeded @269.6k −90.67340 CROSSED #8
off-T (−90.669), now ~4.5 mHa below** (BC lower-E than off-T trap); **★ #15 Si bc_seeded @82k
−62.85292 FULLY RECOVERED** (back in normal Si range). Also: admin cancelled two inference
jobs (5451338/5451339) 06:56 — account-migration housekeeping.

2026-06-29: 6 active jobs RUNNING, queue had NO pending follow-ups (06-28's
follow-ups had taken over as the running jobs). Requeued one afterany each:
**current running job → its new follow-up:** #10 5402683→5415382, #11
5402686→5415383, #12 5404792→5415384, #13 5402701→5415385, #14 5404793→5415386,
#15 5402724→5415387. ⚠️ #15 Si bc_seeded energy REGRESSING upward (−62.75→−62.16
over the job span) — watch next check. Energies in the two comparison memories.

2026-06-27 eve: did a full requeue. Submitted #10/#11/#13 (were out of queue;
#13 launched for the first time), and queued one `afterany` follow-up for ALL
six active jobs. **Current job IDs + the current-job start→current span** (per
step 3a; instantaneous noisy energies):

- **ACTIVE:**
  **#10 Silicon T-relaxed PP q** (job **5402651**, follow-up 5402683):
    resumed @91135 (−62.851) → 91246 (−62.872), +111 steps (just restarted).
  **#11 Silicon T-relaxed classical PP** (job **5402661**, follow-up 5402686):
    resumed @92001 (−62.880) → 92092 (−62.903), +91 steps (just restarted).
  **#12 Diamond T-relaxed PP quantum** (job **5391699**, follow-up 5402704):
    @112001 (−90.511) → 160459 (−90.592), +48458 steps, ΔE −0.081.
  **#13 Diamond T-relaxed classical PP** (job **5402669**, follow-up 5402701,
    FIRST launch 2026-06-27): fresh net @0 (−38.0 warm-up) → 117 (−40.4), brand
    new. [[project-t-relaxed-runs]].
  **#14 Diamond BC-seeded PP q (EXP-002)** (job **5391700**, follow-up 5402711):
    @56001 (−90.377) → 104251 (−90.620), +48250 steps, ΔE −0.243 (descending).
  **#15 Silicon BC-seeded PP q (EXP-002 silicon analogue)** (job **5395609**,
    follow-up 5402724, width 0.45): fresh net @0 (−23.0 warm-up) → 14950
    (−62.637), +14950 steps, still early/spiky.
- **NOT STARTED:**
  - (#13 Diamond T-relaxed classical — now SUBMITTED 2026-06-27, moved to ACTIVE.)
  - **#16 Silicon BC-relaxed CLASSICAL PP (fixed-H at BC) — RUN NOT STARTED, and
    config/script DO NOT EXIST YET (added to monitoring 2026-06-25).** This is the
    silicon analogue of the diamond classical-BC run (#4): a fixed point-charge
    muon at the Si BC site in the bc_relaxed geometry. To create it, mirror
    `silicon/t_relaxed_classical.py`/`jobs/silicon/t_relaxed_classical.sh` but use
    the bc_relaxed geometry + fixed-H at the Si BC site (planned config
    `silicon/bc_relaxed_classical.py`, script `jobs/silicon/bc_relaxed_classical.sh`,
    SLURM name `muon_silicon_c_bc_relaxed`, save `silicon_unpaired/classical/bc_relaxed`).
    No artifacts created yet — must be written before it can be submitted.
- **CLOSED for now:**
  - **#10 Silicon T-relaxed PP q (`muon_silicon_q_t_relaxed`) — STOPPED 2026-06-29.**
    Cancelled training job 5402683 + its pending follow-up 5415382 + the (redundant,
    already-complete) position inference 5415683. Reason (EXP-003): position inference
    showed the quantum muon does NOT localise in the DFT-RELAXED cage — it sits at a
    different, essentially-UNRELAXED T-site (coordinating Si moved 0.0002 bohr), 8.47
    bohr min-image from the relaxed T-site (0.75,0.75,0.75)·a, 0% of samples within 3.5
    bohr. Diamagnetic (H0 confirmed) but the intended relaxed-cage binding test was not
    performed (classical-H ≠ quantum-muon site). Superseded by T-SEEDED run #17 (EXP-003b).
    NOTE the apparent ~10 mHa "below #1 unrelaxed" was convergence-inflated; the clean
    classical pair (#11 vs #2, proton pinned at design site) shows the true relaxation
    energy is only ~5–6 mHa at matched steps. RESTART training if needed: `sbatch
    t_relaxed.sh` from `jobs/silicon/`. See `experiments/EXP-003_silicon_t_relaxed_bound_state.md`.
  - **#9 Silicon BC-relaxed PP q (`muon_silicon_q_bc_relaxed`) — STOPPED
    2026-06-27.** Final E −62.86179 (ewmean −62.86343, var 0.0001) @step 138000.
    Cancelled job 5380709. Reason: position inference (job 5385972) showed the
    quantum muon did NOT hold BC — it drifted ~4.4 bohr to a near-symmetric T cage
    (diamagnetic, loose 0.87-bohr spread); the silicon analogue of #8's off-BC
    result. Question answered → superseded by the BC-seeded run #15 (EXP-002).
    NOTE its inference job 5385972 (`muon_silicon_q_bc_relaxed_inf`) may still be
    running (already dumped all 1000 positions; cancel to free 8 nodes if wanted).
    RESTART training if needed: `sbatch bc_relaxed.sh` from `jobs/silicon/`. See
    [[project-silicon-bc-relaxed-muon-result]].
  - **#7 Diamond classical T-site PP (`muon_diamond_tsite_classical`) — STOPPED
    2026-06-26, CONVERGED.** Final E −90.69624 ± 0.00015 @472k (slope flat,
    −2.6e-8/step). Cancelled job 5380708. SRPD+positions inference launched
    (job 5391680, `inf_muon_diamond_tsite`, `jobs/diamond_2x2/classical_muon/
    inference_pp.sh`); restores the 472k ckpt (stale Jun-21 ckpt_000000 moved to
    `inference_prior_jun21_256k/`). RESTART training if needed: `sbatch
    t_site_pp.sh` from `jobs/diamond_2x2/classical_muon/`.
  - **#8 Diamond BC-relaxed-2 (charged-state) PP q (`muon_d_qpp_bc_rel_2`) —
    STOPPED 2026-06-24, marked COMPLETED.** Final E −90.66904 ± 0.00029 @step
    280196 (was still drifting −1.5e-7/step, but stopped deliberately). Reason:
    the quantum muon localised to the OFF-T site, NOT the seeded BC — this run
    answered its question (a relaxed-BC cage does not pin the quantum muon at BC;
    EXP-001 classical PES shows off-T is a +1.90 eV trap). Cancelled pending
    continuation job 5365663. Superseded by the BC-seeded run #14 (EXP-002).
    RESTART if needed: `sbatch pp_relaxed_2.sh` from
    `ferminet/jobs/diamond_2x2/bc_relaxed/` (restores latest ckpt). See
    [[project-bc-relaxed-pp-relax-2-muon-result]].
  - #6 Diamond unrelaxed PP q (`muon_qpp_diamond_unrelaxed`) — **STOPPED
    2026-06-21, CONVERGED.** Not resubmitted; do not chain afterany follow-ups.
  - #2 Silicon classical T-site — **FINISHED 2026-06-20** (cancelled job 5301536
    at ~step 223695, E≈−62.92, var 0.0001, flat). SRPD inference submitted
    (job 5308476, `inf_t_site.sh`) on the 65-e⁻ `(33,32)` trained net; see
    config-fix notes below. [[project-silicon-quantum-muon-result]]
  - #1 Silicon unrelaxed quantum — cancelled 2026-06-18. Rationale: silicon
    muon is diamagnetic (no muonium) and classical SRPD matches quantum, so the
    classical T-site run (#2) settles the muonium question; a fixed muon is the
    upper bound on contact density, so if classical doesn't bind an e⁻ the
    quantum (delocalized) won't either. See [[project-silicon-quantum-muon-result]].
  - #3 Diamond BC-relaxed no-PP quantum — dropped from queue (~step 874k), not
    resubmitted.
  - #4 Diamond classical BC (pp_T) — CONVERGED (−90.73010±0.0003, slope flat),
    cancelled 2026-06-18.
  - #5 Diamond BC-relaxed PP quantum — noise-limited/plateaued
    (−90.59794±0.0015, slope flat at SEM), dropped from queue (~step 336k).

### Config table

| Config | Job script | Output log | SLURM job name | Submit from |
|---|---|---|---|---|
| Silicon, quantum muon, unrelaxed [CLOSED] | `ferminet/jobs/silicon/muon.sh` | `ferminet/jobs/silicon/muon_silicon_q_unrelaxed.out` | `muon_silicon_q_unrelaxed` | `ferminet/jobs/silicon/` |
| Silicon, classical muon, T-site [ACTIVE] | `ferminet/jobs/silicon/t_site.sh` | `ferminet/jobs/silicon/t_site.out` | `muon_silicon_c_tsite` | `ferminet/jobs/silicon/` |
| Diamond, BC-relaxed, no-PP quantum muon [CLOSED] | `ferminet/jobs/diamond_2x2/bc_relaxed/nopp.sh` | `ferminet/jobs/diamond_2x2/bc_relaxed/muon_nopp.out` | `muon_d_qnopp_bc_rel` | `ferminet/jobs/diamond_2x2/bc_relaxed/` |
| Diamond, BC-relaxed, classical muon (T-site, PP) [CLOSED — converged] | `ferminet/jobs/diamond_2x2/bc_relaxed/pp_classic.sh` | `ferminet/jobs/diamond_2x2/bc_relaxed/pp_T.out` | `bc_diamond_pp_T` | `ferminet/jobs/diamond_2x2/bc_relaxed/` |
| Diamond, BC-relaxed, PP quantum muon [CLOSED] | `ferminet/jobs/diamond_2x2/bc_relaxed/pp.sh` | `ferminet/jobs/diamond_2x2/bc_relaxed/muon.out` | `muon_d_qpp_bc_rel` | `ferminet/jobs/diamond_2x2/bc_relaxed/` |
| Diamond, unrelaxed, PP quantum muon [ACTIVE] | `ferminet/jobs/diamond_2x2/muon.sh` | `ferminet/jobs/diamond_2x2/muon.out` | `muon_qpp_diamond_unrelaxed` | `ferminet/jobs/diamond_2x2/` |
| Diamond, classical muon, T-site, PP [CLOSED — converged −90.69624 @472k, stopped 2026-06-26] | `ferminet/jobs/diamond_2x2/classical_muon/t_site_pp.sh` | `ferminet/jobs/diamond_2x2/classical_muon/t_site.out` (NOT `t_site_pp.out` — that one is STALE/cancelled Jun-19; the live job's StdOut is `t_site.out`) | `muon_diamond_tsite_classical` | `ferminet/jobs/diamond_2x2/classical_muon/` |
| Diamond, BC-relaxed (charged-state DFT, expanded BC), PP quantum muon [ACTIVE] | `ferminet/jobs/diamond_2x2/bc_relaxed/pp_relaxed_2.sh` | `ferminet/jobs/diamond_2x2/bc_relaxed/pp_relax_2.out` | `muon_d_qpp_bc_rel_2` | `ferminet/jobs/diamond_2x2/bc_relaxed/` |
| Silicon, BC-relaxed, PP quantum muon [CLOSED — stopped 2026-06-27 @138k, muon drifted to T cage] | `ferminet/jobs/silicon/bc_relaxed.sh` | `ferminet/jobs/silicon/muon_silicon_q_bc_relaxed.out` | `muon_silicon_q_bc_relaxed` | `ferminet/jobs/silicon/` |
| #10 Silicon, T-relaxed, PP quantum muon [CLOSED — stopped 2026-06-29, muon avoided relaxed cage] | `ferminet/jobs/silicon/t_relaxed.sh` | `ferminet/jobs/silicon/t_relaxed.out` | `muon_silicon_q_t_relaxed` | `ferminet/jobs/silicon/` |
| #11 Silicon, T-relaxed, classical muon (fixed-H, PP) [CLOSED — BLEW UP @step 306603 2026-07-06, job 5501849 FAILED; last clean ckpt 306000; user left stopped. Recover: move 308k–330k ckpts aside + truncate csv ≤306602 + sbatch] | `ferminet/jobs/silicon/t_relaxed_classical.sh` | `ferminet/jobs/silicon/t_relaxed_classical.out` | `muon_silicon_c_t_relaxed` | `ferminet/jobs/silicon/` |
| #12 Diamond, T-relaxed, PP quantum muon [CLOSED — STOPPED 2026-07-04 @478k −90.65979; muon HELD relaxed cage (muonium), superseded by T-SEEDED #18] | `ferminet/jobs/diamond_2x2/t_relaxed/pp.sh` | `ferminet/jobs/diamond_2x2/t_relaxed/pp.out` | `muon_d_qpp_t_rel` | `ferminet/jobs/diamond_2x2/t_relaxed/` |
| #13 Diamond, T-relaxed, classical muon (fixed-H, PP) [ACTIVE — job 5402669, FIRST launch 2026-06-27] | `ferminet/jobs/diamond_2x2/t_relaxed/classical.sh` | `ferminet/jobs/diamond_2x2/t_relaxed/classical.out` | `muon_d_cpp_t_rel` | `ferminet/jobs/diamond_2x2/t_relaxed/` |
| #14 Diamond, BC-seeded PP quantum muon (EXP-002) [ACTIVE — job 5370014, submitted 2026-06-24] | `ferminet/jobs/diamond_2x2/bc_relaxed/bc_seeded.sh` | `ferminet/jobs/diamond_2x2/bc_relaxed/bc_seeded.out` | `muon_d_qpp_bc_seeded` | `ferminet/jobs/diamond_2x2/bc_relaxed/` |
| #15 Silicon, BC-seeded PP quantum muon (EXP-002) [ACTIVE — job 5395609, SUBMITTED 2026-06-27, width 0.45] | `ferminet/jobs/silicon/bc_seeded.sh` | `ferminet/jobs/silicon/bc_seeded.out` | `muon_silicon_q_bc_seeded` | `ferminet/jobs/silicon/` |
| #16 Silicon, BC-relaxed classical muon (fixed-H @ BC, PP) [ACTIVE — config/script WRITTEN + LAUNCHED 2026-07-05, job 5501873, follow-up 5501876] | `ferminet/jobs/silicon/bc_relaxed_classical.sh` | `ferminet/jobs/silicon/bc_relaxed_classical.out` | `muon_silicon_c_bc_relaxed` | `ferminet/jobs/silicon/` |
| #17 Silicon, T-seeded PP quantum muon (EXP-003b) [ACTIVE — job 5419590, SUBMITTED 2026-06-29, seed (0.75a)³, width 0.5] | `ferminet/jobs/silicon/t_seeded.sh` | `ferminet/jobs/silicon/t_seeded.out` | `muon_silicon_q_t_seeded` | `ferminet/jobs/silicon/` |
| #18 Diamond, T-seeded PP quantum muon (EXP-003b, diamond analogue of #17) [ACTIVE — job 5491822 SUBMITTED 2026-07-04, follow-up 5491823; seed (0.7502845a)³, width 0.35] | `ferminet/jobs/diamond_2x2/t_relaxed/t_seeded.sh` | `ferminet/jobs/diamond_2x2/t_relaxed/t_seeded.out` | `muon_d_qpp_t_seeded` | `ferminet/jobs/diamond_2x2/t_relaxed/` |
| #19 Diamond, BC-relaxed WIDE-muon-proposal PP quantum muon (EXP-004) [ACTIVE — job 5514175 SUBMITTED 2026-07-06; UNSEEDED, muon_move_width=0.3, burn_in=2000] | `ferminet/jobs/diamond_2x2/bc_relaxed/wide_burnin.sh` | `muon_wide_burnin.out` (in repo root, NOT the jobs dir) | `muon_d_qpp_bc_rel_widebi` | `ferminet_piku/` (repo root — where sbatch was run) |

**#18 Diamond T-seeded (added 2026-07-04, NOT STARTED):** diamond analogue of #17. Same
geometry as #12 `diamond/t_relaxed/pp.py` (T-relaxed diamond cage, +0.4% expanded, particles
`(33,32,1)` quantum muon doublet, PP on C, a=6.74) but the **muon walkers are seeded at the
relaxed T-site** via `cfg.mcmc.muon_init_coord=(0.7502845*a,)*3` (verified centroid of the 4
expanded C #4-#7, 2.9305 bohr equidistant), `width 0.35` (scaled from Si's 0.5 by the T-cage
size ratio diamond 2.93/Si 4.39). Config `configs/diamond/t_relaxed/t_seeded.py` (py_compile
clean), script `jobs/diamond_2x2/t_relaxed/t_seeded.sh`. save `/projects/u6em/parv/diamond/
unpaired/t_seeded/pp`, fresh net (restore==save). **LAUNCHED 2026-07-04: job 5491822 (PENDING/Priority),
follow-up 5491823 (afterany:5491822). Empty save dir created before submit → fresh net guaranteed.**
NOTE unlike silicon (#10 muon FLED the
contracted cage → motivated #17), the diamond #12 unseeded muon already HELD the expanded relaxed
cage as muonium ([[project-diamond-t-relaxed-muon-result]]); #18 seeds inside for a clean symmetric
comparison with #17 and to confirm the held state isn't a warm-start/init artifact. **Sanity check
owed once it RUNS:** fresh-net log line ("Training new model"), muon seeds at cubic-frac ≈0.7503
coordinated by the expanded C #4-#7, no early NaN.

**#19 Diamond BC-relaxed WIDE muon proposal (EXP-004, SUBMITTED 2026-07-06, job 5514175).**
Same geometry/setup as #5 `diamond/bc_relaxed/pp.py` (BC-relaxed cage, particles `(33,32,1)`
quantum muon doublet, PP on C, a=6.74) and muon left at the DEFAULT carbon-centred init
(UNSEEDED, `muon_init_coord=None`) — this is the complement to the BC-seeded #14. The only
changes vs #5: the muon starts with a WIDE proposal `cfg.mcmc.muon_move_width=0.3` (new
per-species knob; electrons stay at move_width=0.02, each species adapts its own width
after) and `cfg.mcmc.burn_in=2000`. Config `configs/diamond/bc_relaxed/wide_burnin.py`,
FRESH net (`restore_path==save_path==/projects/u6em/parv/diamond/unpaired/bc_relaxed/
pp_wide_burnin`, created empty before submit → "Training new model"). Launcher matches the
running bc_seeded convention (`cd ~/ferminet_piku`, parvfect miniforge; NOT the stale
ferminet_remote/pp.sh paths). Tests whether keeping the muon proposal wide through the
early selection window un-traps it off-T→BC without seeding — see EXP-004; **H0 (re-collapse
to off-T) is the likely prior**, run as a diagnostic. Code: `base_config.mcmc.muon_move_width`
(default None → byte-identical to before) + `train.py` per-species width-init branch.
[[project-muon-mcmc-width-diffusion]], `experiments/EXP-004_muon_width_burnin_diffusion.md`.

**Checks owed once #19 has RUN a little (first ~few k steps / first checkpoints):**
1. **Fresh net:** `.out` shows "No checkpoint found. Training new model." (NOT a warm-start
   restore — EXP-004 confound #1; re-verify after every afterany restart).
2. **Initial widths:** muon `mcmc_width` = 0.3, electrons = 0.02 (the patch worked); and the
   muon walkers are at the DEFAULT carbon-centred spread, NOT at BC (unseeded).
3. **No early NaN:** electrons at 0.02 should be calm; `reset_if_nan` is on — watch the first
   ~5k steps.
4. **H1-vs-H0 fast read** — `tools/muon_diffusion_check.py` on the save_path
   (`--a 6.74 --particles 33,32,1 --site offT=0.474,0.474,0.474 --site BC=0.125,0.125,0.125`):
   H1 = early spread stays large past step ~2000 and settles on BC (0.125); H0 = collapses to
   ~0.15 bohr by ~2000 steps as in the off-T `bc_relaxed/pp` run.
5. **Width sanity** — `tools/muon_width_check.py --last`: muon stayed wide early, electrons
   ~0.02→~0.08 (rules out any electron-instability effect).
6. **Site + energy** — mid-training inference + `muon_site_analysis.py`: peak at BC (0.125)
   vs off-T (0.474); track block-avg VMC energy vs the off-T run's −90.669 Ha (below = H1).

This 7th job (added 2026-06-15) uses the unpaired-up-spin setup
`cfg.system.particles = (33, 32)` (65 e⁻, mol.charge=1, PP on C, H at
0.75a,0.75a,0.75a), save path
`/projects/u6em/parv/diamond/unpaired/classical/T_site/pp`. Queued separately
(later) from the other 6, so its wall-clock will usually be out of sync with them.

This 8th job (added 2026-06-18) is a quantum-muon training in a *second*
BC-relaxed geometry: the C atoms were re-relaxed in DFT in the charged
(open-shell, unpaired-electron) state, giving a slightly more expanded bond
centre than `pp.py`/`pp_new.py`. Config `diamond/bc_relaxed/pp_relax_2.py`,
particles `(33,32,1)` quantum muon (doublet), PP on C, save path
`/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_relax_2`. Motivation: the
muon failed to localise at the BC site with the earlier relaxed coordinates.

This 9th job (added 2026-06-19) is the silicon analogue of #8: a quantum muon
in a *BC-relaxed* silicon geometry (DFT-relaxed in the charged/open-shell state,
Si #0 and #8 pushed ~0.77 bohr outward along [111], +34.8% Si-Si expansion).
Config `silicon/bc_relaxed.py` (renamed from `bc.py`), particles `(33,32,1)`
quantum muon (doublet, 65 e⁻, PP on Si, a=10.26 bohr), save path
`/projects/u6em/parv/silicon_unpaired/bc_relaxed`. Started fresh (no prior
checkpoints). Motivation: test whether the muon localises at the Si BC site.

This 14th job (added 2026-06-24) is EXP-002: same geometry/setup as #8
(`pp_relax_2`, BC-relaxed cage, particles `(33,32,1)` quantum muon doublet, PP
on C) but the **muon walkers are seeded at the BC site** via the new
`cfg.mcmc.muon_init_coord=(0.125*a,)*3`, `muon_init_width=0.3`. Config
`diamond/bc_relaxed/bc_seeded.py`, FRESH net (`restore_path==save_path==
/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_bc_seeded`, created empty so
`find_last_checkpoint`→None ⇒ "No checkpoint found. Training new model.").
**Pre-launch sanity check still owed once it RUNS:** confirm the early log shows
the fresh-net line (NOT a warm-start restore — EXP-002 risk #1) and that the muon
localises at BC (≈0.125 cubic-frac), not off-T. Compare its VMC energy vs the
off-T #8 value −90.669 Ha. Full protocol: `experiments/EXP-002_bc_seeded_muon_run.md`.
See [[project-bc-relaxed-pp-relax-2-muon-result]].

This 15th job (SUBMITTED 2026-06-27, job 5395609) is the silicon analogue of #14:
same geometry as #9 `silicon/bc_relaxed.py` (BC-relaxed Si cage, particles
`(33,32,1)` quantum muon doublet, PP on Si, a=10.26) but the **muon walkers are
seeded at the Si BC site** via `cfg.mcmc.muon_init_coord=(0.125*a,)*3`,
`muon_init_width=0.45` (widened from diamond's 0.3 by the BC→host half-bond ratio
Si 2.99/C 2.06 ≈1.45, committed 294d2ef). Config `silicon/bc_seeded.py`, FRESH net
(`restore_path==save_path==/projects/u6em/parv/silicon_unpaired/bc_seeded`, created
empty so `find_last_checkpoint`→None ⇒ "Training new model"). Launched after #14
verified to HOLD BC ([[project-bc-seeded-muon-result]]) and #9 shown to drift to a
T cage ([[project-silicon-bc-relaxed-muon-result]]). **Sanity check owed once it
RUNS:** early log shows fresh-net line (not warm restart) + muon seeds/holds BC
(≈0.125 cubic-frac, ~1.28 bohr, NOT the T cage) + no early NaN. KEY question: does
BC-seeding pin the silicon muon at BC as it did in diamond?

This 17th job (SUBMITTED 2026-06-29, job 5419590) is EXP-003b — the T-site analogue
of the BC-seeded runs (#14/#15). Same geometry as #10 `silicon/t_relaxed.py`
(T-relaxed Si cage, particles `(33,32,1)` quantum muon doublet, PP on Si, a=10.26)
but the **muon walkers are seeded at the RELAXED T-site** via
`cfg.mcmc.muon_init_coord=(0.75*a,)*3` (= centroid of the 4 contracted Si #4-#7,
verified dead-on, 4.394 bohr equidistant), `muon_init_width=0.5` (below the measured
~0.81 bohr equilibrium T spread). Config `silicon/t_seeded.py`, FRESH net
(`restore_path==save_path==/projects/u6em/parv/silicon_unpaired/t_seeded`, created
empty so `find_last_checkpoint`→None ⇒ "Training new model"). Motivation: #10 showed
the unseeded quantum muon AVOIDS the relaxed cage (drifts ~8.5 bohr to an unrelaxed
T-site); this tests whether, seeded inside, it HOLDS the relaxed site (metastable
relaxed-site state) or escapes again. **Sanity check owed once it RUNS:** early log
shows fresh-net line (not warm restart) + muon seeds at the relaxed T-site
(cubic-frac ≈0.75, coordinated by the 0.0487-bohr-moved Si) + no early NaN. KEY
question: does T-seeding pin the muon in the DFT-relaxed cage? See
`experiments/EXP-003_silicon_t_relaxed_bound_state.md`.

All jobs use 8 nodes x 4 GPUs. Each restarts from the latest checkpoint
automatically, so resubmission is just `sbatch <script>` — no extra flags
needed.

Related: [[bc-relaxed-pp-new-muon-result]] (separate, one-off inference run —
not part of this daily set).
