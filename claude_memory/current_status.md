---
name: current-status
description: "Snapshot of the most recent session's focus and open threads — overwrite/update this each session rather than appending. LATEST (2026-07-07): EXP-004 wide-burnin production run 5529246 launched correctly on 8 nodes, muon width 0.3 intervention CONFIRMED live in log."
metadata: 
  node_type: memory
  type: project
  originSessionId: cd7babd3-df02-4561-aef6-eb933702fa94
---

## As of 2026-07-07 (#3, branch `muon_width`) — daily monitoring: #18 hung→resubmitted; #13 soft blow-up→restarted from pre-spike ckpt; swapped Si bc_seeded inference → t_seeded inference (bound-state test)

**Daily training-monitoring check + 4 actions (all user-directed).** 7 training jobs were RUNNING (#13–#19)
+ 2 inference. Energies (block-avg, none converged): **★ #14 d bc_seeded @570k −90.69682 ± 0.00029** (now
just below classical fixed-T #7 −90.69624, ~28 mHa below #8 off-T — lowest quantum diamond); **★ #17 si
t_seeded @218k −62.89249 ± 0.00017** (lowest quantum Si, ~21 mHa below frozen #10); #15 si bc_seeded @190k
−62.89198; #16 si classical bc_rel @58k −62.87677 (climbing out of warm-up, follow-up 5519019 ✓).

Actions:
- **🔴 #18 diamond t_seeded was HUNG** (log+csv frozen @step 121210 / "total 234.7min" since 07-06 19:17,
  ~12h, SLURM still RUNNING — same signature as prior #13/#14 hangs; had no follow-up). **Cancelled 5519015,
  resubmitted plain → job 5529408** (restores latest ckpt ~120k). Sanity owed once it RUNS: "Restoring
  checkpoint …120000" (NOT fresh/Not 121210), energy resumes ~−90.55.
- **⚠️ #13 diamond classical t_rel had a SOFT BLOW-UP.** Clean through 428–430k (−90.6926), then 430–432k
  spiked (min −47.6, mean −90.05); recovered only to −90.658 by 460k, elevated noise. **Restarted from the
  last clean ckpt 428000** (like #11/#15): cancelled 5519016, moved 19 post-428k ckpts →
  `pp/post430k_spike_bak_20260707/` (full csv backed up there as `train_stats_full_to460k.csv`), truncated
  live csv to ≤428000 (last row −90.69606), **resubmitted → job 5529411**. Sanity owed: log "Restoring
  checkpoint …428000", energy resumes ~−90.696. (5529411/5529408 PENDING w/ reason **AssocGrpCPUMinutesLimit**
  = account CPU-minutes hold, will start when minutes/nodes free — normal, not an error.)
- **Swapped silicon inference: cancelled bc_seeded inference (5514972), launched t_seeded inference → job
  5529415** (`muon_silicon_q_t_seeded_inf`, PENDING). NEW files (geometry byte-identical to `t_seeded.py` =
  the t_relaxed Si cage, verified via diff): `configs/silicon/inference_t_seeded.py` +
  `jobs/silicon/inference_t_seeded.sh` (home env, 8 nodes, optimizer=none so NO KFAC → safe either env,
  positions=True, restore `/projects/u6em/parv/silicon_unpaired/t_seeded` latest ckpt 218000, save
  `…/t_seeded/inference` empty→falls back, `positions/` pre-created to dodge the mkdir crash). **Purpose
  (user): does the T-SEEDED silicon muon form a BOUND STATE (muonium)?** — #17 is the lowest-E Si run and
  the muon is seeded/held in the relaxed T-cage; test contact/net-spin. **OWED once it dumps positions_*.npy:**
  add a `silicon_t_seeded` case to `tools/muon_site_analysis.py` (SI geom a=10.26, same as `silicon_t_relaxed`
  case but positions dir → t_seeded/inference/positions) then `site` / `srpd` / `srpd_extended_radius`
  (g_up≫g_down at r→0 or localized net-spin excess = muonium; prior Si result = DIAMAGNETIC, so this tests
  whether seeding+holding the cage changes that).

Open (offered, not done): afterany follow-ups for #14/#15/#17 (only #16 has one) and #19/EXP-004 — hold
until the AssocGrpCPUMinutes hold clears (queuing more now won't help). EXP-004 (#19, 5529283) still RUNNING
& healthy (see #2 section) — its step-100 muon walkers still fully diffuse (RMS ~13 bohr), watching for
contraction at ckpt 2000.

## As of 2026-07-07 (#2, branch `muon_width`) — EXP-004 RESOLVED onto home env: kfac_jax fresh-net crash diagnosed + fixed; run 5529283 training with muon width 0.3

**EXP-004 is finally running correctly.** The v3 run under the HOME env first CRASHED (job 5529246, FAILED
6.5min): fresh-net multi-host KFAC hit `AssertionError: value.sharding is NamedSharding` in kfac_jax. Full
diagnosis + fix in [[env-kfac-jax-fresh-net-crash]]. Key findings this session:
- **Root cause:** the HOME env's `kfac_jax` was a different (newer/dev) build than the OLD env's, both tagged
  `0.0.8`. The newer build's `get_first(step_counter)` asserts NamedSharding and dies on fresh-net KFAC
  (warm restores hide it — that's why long-running prod jobs never showed it).
- **The OLD env is NOT a fallback:** it imports a STALE/copied `ferminet`, so a `wide_burnin.py` run there
  printed NO `[EXP-004]`/`Initial MCMC width` lines = it runs a train.py without the `muon_move_width` block.
  Only the HOME env imports our editable repo. ⇒ **all our fork runs must use the HOME env.**
- **Fix:** swapped HOME env `kfac_jax` + dist-info with the OLD env's working copy (originals backed up
  `*.bak_20260707` in HOME site-packages).
- **✅ VERIFIED on 8-node fresh-net:** job **5529283** (HOME env, `pp_wide_burnin_v3`) prints `[EXP-004] ...0.3`
  + `Initial MCMC width per species: [0.02 0.02 0.3]`, cleared the KFAC step, wrote `ckpt_000000` with
  **muon width 0.3 / electrons 0.02** (`tools/muon_width_check.py --particles 33,32,1`), now training past
  step 0 (step ~22, −35.7 climbing, 1.5s/step). `wide_burnin.sh` env line is back on HOME `parvfection`.

**>>> NEXT: <<<** queue an afterany follow-up for 5529283 (24h limit) so the run continues gap-free; then
let it train and run muon-position inference to see if off-T→BC un-trapping happens (prior H0-likely).

## As of 2026-07-07 (branch `muon_width`) — EXP-004 wide-burnin FINALLY launched CORRECTLY on 8 nodes (v3); width intervention CONFIRMED live in the log; verify cancelled (would OOM)

**EXP-004 (wide muon proposal to un-trap off-T→BC) — the real production run is now RUNNING with the intervention verified.** After the v1/dense/v2 runs silently ran the *old* code (muon width stayed 0.02 — stale-bytecode/timing, see [[project-muon-mcmc-width-diffusion]] and the 07-06 section below), this session got it right end-to-end:

- **Cancelled the single-node verify.** Was going to `sbatch wide_burnin_verify.sh` (job 5529245) but user flagged it would **OOM** — batch_size 4096 across only 4 GPUs on 1 node. Cancelled 5529245 and went straight to the 8-node production run instead. (verify config/script still exist but are unused — the intervention is now proven on the real run, so the verify is unnecessary.)
- **Added an unmissable stdout print in `train.py`** (in the `if cfg.mcmc.muon_move_width is not None:` block, right after the `sample_all` check, ~line 979): `print('[EXP-004] muon_move_width applied: ...', flush=True)`. `print`+flush (not logging) so it lands in srun stdout regardless of the absl handler. Complements the existing `logging.info('Initial MCMC width per species: ...')` guardrail.
- **Cleared stale `ferminet/__pycache__/train*.pyc` + the config pyc** (a stale pyc is exactly what masked the fix on v2), py_compiled clean.
- **Created the fresh empty save dir** `/projects/u6em/parv/diamond/unpaired/bc_relaxed/pp_wide_burnin_v3` (was MISSING → would not have been a warm restore, but created empty to be explicit; `restore_path == save_path` so it starts a brand-new net and the width init actually applies).
- **Launched 8-node run: `sbatch ferminet/jobs/diamond_2x2/bc_relaxed/wide_burnin.sh` → job 5529246 `muon_d_qpp_bc_rel_widebi_v3` (RUNNING, 8 nodes, 24h limit).** Home env (`/home/u6em/parvfection.u6em/miniforge3` + `ferminet-piku`).
- **✅ WIDTH INTERVENTION CONFIRMED in `muon_wide_burnin_v3.out`** (~105 s after start): `[EXP-004] muon_move_width applied: last-species init width set to 0.3 (electrons at cfg.mcmc.move_width=0.02)` and `Initial MCMC width per species: [0.02 0.02 0.3]`, then `Burning in MCMC chain for 2000 steps`. This is the thing that silently failed on v1/dense/v2 — it is finally right.

**Config (`configs/diamond/bc_relaxed/wide_burnin.py`):** `muon_move_width=0.3`, electrons 0.02, `burn_in=2000`, batch 4096, `save_freq=100` (checkpoint every 100 steps — dense; `save_tfreq=235`min never binds at ~0.6s/step), 900k iters, save `pp_wide_burnin_v3`.

**Env note:** current home is `parvfection.u6em`; the `_oldenv` scripts + several production `.sh` (e.g. `silicon/t_seeded.sh`) still point at the OLD home `/home/u6em/parvfect.u6em/miniforge3`. `wide_burnin.sh`/`wide_burnin_verify.sh` correctly use the current home.

**>>> NEXT ACTIONS for EXP-004: <<<** (a) queue an afterany follow-up so the 24h run continues gap-free; (b) once it trains out, run muon-position inference and check whether the off-T→BC un-trapping happened (prior on outcome stays **H0-likely** = off-T re-collapse; the run is a diagnostic). Optionally bump `save_freq` 100→2000 to avoid piling up hundreds of ckpt files (won't change results). Diamond bc_seeded inference (5527930) confirmed working this session at ckpt 562000 (1000 fresh positions).

## As of 2026-07-06 (#2, daily check, branch `af`) — #11 si-classical-t_rel BLEW UP & CLOSED; #18 resubmitted; queued 5 afterany follow-ups; #14 ~25.5 mHa below #8 off-T; #17 lowest quantum Si

**Daily monitoring.** 5 training jobs RUNNING & healthy (#13 5501850, #14 5501851, #15 5501852, #16 5501876,
#17 5501853; csv fresh 11–67 min), + #15 bc_seeded INFERENCE 5514972 RUNNING (name truncates to
`muon_silicon_q_bc_seeded` in squeue — NOT a training race; writes `inference_bc_seeded.out`), + EXP-004
verify 5518646 still PENDING (Priority; background monitor pid 88061 alive). No follow-ups were queued for
the 5 running jobs → **queued 5 afterany (user OK'd): 5519016→#13 (dep 5501850), 5519017→#14 (5501851),
5519018→#15 (5501852), 5519019→#16 (5501876), 5519020→#17 (5501853).**

**🔴 #11 (si classical t_relaxed) BLEW UP & CLOSED.** Ran clean to step 306602 (−62.92218, ewvar 5.9e-5),
then INSTANT divergence at 306603 (E→1.4e10, ewvar→1.8e19), stayed garbage to 330k where job 5501849
**FAILED (exit 1:0)** 03:28 UTC. Ckpts 308k–330k are NaN-corrupt; last clean = `qmcjax_ckpt_306000.npz`.
**User chose LEAVE STOPPED** (was plateau-approaching −62.924 @300k, ~11 mHa below frozen #2 — treat as tight
upper bound). To recover later: move 308k–330k ckpts aside, truncate csv to ≤306602, sbatch (restores 306000).

**🟡 #18 (diamond t_seeded) timed out cleanly** @112k (−90.587), both chained follow-ups exhausted →
**RESUBMITTED plain: job 5519015** (Priority; resumes from 112k warm-up climbing toward ~−90.6).

**Energies (clean block-avg, ΔE vs 07-05; none converged):**
- #13 d c t_rel @416k −90.69097 ± 0.00023 (tool says CONVERGED but −6.0 mHa run-over-run ⇒ plateau-approaching, NOT settled), Δ+78k
- #14 d bc_seed @526k **−90.6947** (500–520k block; tool trailing # is incomplete-final-block artifact, real traj monotonic descending), Δ−3.8 mHa — **★ ~25.5 mHa BELOW #8 off-T −90.669, lowest quantum diamond**
- #15 si bc_seed @167k −62.88753 ± 0.00032 (−2.1e-7, descending), Δ−10.5 mHa, normal Si range
- #16 si c bc_rel @34k −62.7566 warm-up (restored from ckpt 030510, climbing healthily; orig fresh-net line not re-confirmable from follow-up log, run clean)
- #17 si t_seed @194k **−62.88820 ± 0.00026** (−1.6e-7, descending), Δ−7.5 mHa — **★ lowest quantum Si, ~16.9 mHa BELOW frozen #10 −62.87133**
- #18 d t_seed @112k −90.53204 warm-up climbing (resubmitted)

Both comparison tables refreshed. EXP-004 (5518646) status unchanged — still PENDING, no launch decision owed
until it runs (see prior 07-06 EXP-004 section below for the >>> NEXT ACTION <<<).

## As of 2026-07-06 (LATEST, branch `af`) — EXP-004 deployment bug found & fixed; muon_move_width verified on CPU; single-node verify QUEUED, then launch 8-node v3

**EXP-004 (wide muon proposal to un-trap off-T→BC) — first 3 runs were INVALID.** All three wide-burnin
runs `pp_wide_burnin` / `pp_wide_burnin_dense` / `pp_wide_burnin_v2` have ckpt_000000 **muon width 0.02,
not 0.3** (`tools/muon_width_check.py --particles 33,32,1`) — the `muon_move_width=0.3` intervention never
ran. Cause was NOT `adapt_during_burnin` (no such flag; burn-in never adapts width) and NOT the config
(all print muon_move_width=0.3). It was **stale-code/timing**: the `muon_move_width` block in `train.py`
either post-dated the launch (v1 06:00, dense 07:33 both before the 07:50 train.py edit) or was masked by a
stale `__pycache__/train.cpython-312.pyc` on the compute node (v2 launched 07:50:59, 53s after the edit,
but its stdout has no "Initial MCMC width per species" guardrail line → ran old code). Git commit status is
irrelevant to SLURM (python imports train.py from disk). See [[project-muon-mcmc-width-diffusion]].

**Actions taken this session:**
- **Cancelled** the running-but-broken v2 job `5514933` (was muon=0.02).
- **Cleared** stale `ferminet/__pycache__/train*.pyc`.
- **CPU-verified** the fix on the REAL wide_burnin config (optimizer='none', `taskset -c 0-1` +
  `--xla_cpu_multi_thread_eigen=false` to dodge the login-node pthread/ulimit abort) →
  logs **`Initial MCMC width per species: [0.02 0.02 0.3]`** ✓ (electrons 0.02, muon 0.3).
- **Committed** the fix locally as `68bb375` on branch `af` (user will handle push/git themselves — do NOT push).
  git identity set locally: Parvfect <parvagrw02@gmail.com>.
- **Production config** `configs/diamond/bc_relaxed/wide_burnin.py` now points at fresh
  **`pp_wide_burnin_v3`**; 8-node job `jobs/diamond_2x2/bc_relaxed/wide_burnin.sh` renamed `_v3`.
- **Single-node verify** `configs/.../wide_burnin_verify.py` + `jobs/.../wide_burnin_verify.sh`
  (burn_in=50, 201 iters, throwaway `pp_wide_burnin_verify`) **SUBMITTED = job 5518646 (PENDING/Priority)**.
- A background monitor (`scratchpad/wait_verify.sh`) polls until 5518646 finishes, then checks the width
  log line + ckpt_000000 width.

**>>> NEXT ACTION when verify (5518646) finishes: <<<** confirm (a) its stdout has
`Initial MCMC width per species: [0.02 0.02 0.3]`, (b) `pp_wide_burnin_verify/qmcjax_ckpt_000000.npz` muon
width = 0.3. **If both pass → USER APPROVED launching the 8-node production run:
`sbatch ferminet/jobs/diamond_2x2/bc_relaxed/wide_burnin.sh` (into fresh pp_wide_burnin_v3).** Then delete
the throwaway `pp_wide_burnin_verify` dir. Prior on EXP-004 outcome stays **H0-likely** (off-T re-collapse)
— run is a diagnostic. Conda env python: `/home/u6em/parvfection.u6em/miniforge3/envs/ferminet-piku/bin/python`.

## As of 2026-07-05 (branch `af`) — daily check: 6 RUNNING & healthy; only #18 had a follow-up → queued 5 afterany; #14 slope now FLAT & ~22 mHa below #8 off-T; #17 firmly lowest quantum Si; #18 climbing out of warm-up

**6 active jobs (#11/#13/#14/#15/#17/#18) all RUNNING & healthy** (csv fresh ≤40 min at 04:21 UTC, NaN
≤7/last-2000 = warm-up level). Current running IDs: 5491721 #11, 5485272 #13, 5491723 #14, 5485273 #15,
5491724 #17, 5491822 #18. **Only #18 had a pending afterany follow-up (5491823)** → the other 5 were
running with no continuation, so **queued 5 afterany (user OK'd): 5501849→#11 (dep 5491721), 5501850→#13
(5485272), 5501851→#14 (5491723), 5501852→#15 (5485273), 5501853→#17 (5491724)** — all PENDING/Dependency,
every active job now gap-free.

**Energies (clean block-avg, ΔE vs 07-04 #2; NONE converged):**
- #11 si c t_rel @300k **−62.92419 ± 0.00015** (−6.9e-8/step, still >2σ = plateau-approaching), Δ+19.6k / −1.4 mHa, ~11 mHa below frozen #2
- #13 d c t_rel @338k −90.68500 ± 0.00042 (−5.8e-8/step, slope stat. FLAT/noise-limited), Δ+37.8k / −2.4 mHa
- #14 d bc_seed @450k **−90.69088 ± 0.00029** (−8.1e-8/step, **slope now STATISTICALLY FLAT** = plateauing/noise-limited), Δ+36k / −3.9 mHa — **★ now ~21.9 mHa BELOW #8 off-T −90.669, lowest quantum diamond run, closing on classical fixed-T −90.696**
- #15 si bc_seed @126k −62.87699 ± 0.00053 (−3.8e-7/step, descending), Δ+20k / −7.7 mHa — normal Si range
- #17 si t_seed @154k **−62.88074 ± 0.00033** (−2.3e-7/step, descending), Δ+20k / −5.6 mHa — **★ firmly lowest quantum Si, ~9.4 mHa BELOW frozen #10 −62.87133**
- #18 d t_seed @42k warm-up climbing (−90.36 inst, ewmean −90.32, tool plateau garbage) — heading to ~−90.6

Both comparison tables refreshed. Verdicts: **★ #14 bc_seeded slope now flat & ~22 mHa below #8 off-T**
(BC-vs-off-T verdict firmly BC); **★ #17 t_seeded firmly lowest quantum Si**; **✅ #18 climbing cleanly
out of warm-up**. Watch next check: whether #14/#13 hold their flat plateaus and #18 reaches ~−90.6.

**POST-CHECK ACTION (2026-07-05, user request): ADDED + LAUNCHED #16 silicon BC-relaxed CLASSICAL**
(the long-planned silicon analogue of diamond #4). Wrote `configs/silicon/bc_relaxed_classical.py`
+ `jobs/silicon/bc_relaxed_classical.sh` (py_compile clean; mirrors `t_relaxed_classical.py` but with
the bc_relaxed 16-Si geometry + a FIXED 'H' at the BC midpoint (0.125a,0.125a,0.125a) = exact midpoint
of Si #0/#8, 2.994 bohr equidistant; particles (33,32) 65 e⁻, mol.charge=1, save
`silicon_unpaired/classical/bc_relaxed`). Save dir created empty before submit → fresh net.
**Job 5501873 (PENDING), follow-up 5501876 (afterany).** Now 7 active jobs. **Sanity check owed once it
RUNS:** tail `bc_relaxed_classical.out` for fresh-net line ("No checkpoint found. Training new model."),
no early NaN blow-up. This is the fixed-BC upper bound for the Si BC-vs-T energy question (Si muon is
diamagnetic, so a fixed-BC classical run bounds the BC contact density — pairs with #15 bc_seeded quantum).

## As of 2026-07-04 (#2, branch `af`) — daily check: 6 RUNNING & healthy w/ follow-ups already PENDING → NO requeue; #18 fresh-net sanity PASSES; #14 ~18 mHa below #8 off-T; #17 now below frozen #10

**Nothing to submit.** All 6 active jobs RUNNING (5485268 #11, 5485263 #13, 5485270 #14, 5485264 #15,
5485271 #17, 5491822 #18) and **each ALREADY had exactly one afterany follow-up PENDING** (5491721→#11,
5485272→#13, 5491723→#14, 5485273→#15, 5491724→#17, 5491823→#18) ⇒ every about-to-expire job covered
gap-free, no requeue this check. All healthy: csv fresh within ~20 min, NaN ≤5/last-2000 (warm-up level).
(Active set is now #11,#13,#14,#15,#17,#18 — #12 STOPPED earlier today.)

**✅ #18 diamond T-seeded fresh-net sanity check PASSES** (owed from earlier today): log shows "No
checkpoint found. Training new model." (fresh net, NOT warm restore), energy climbing normally out of
warm-up (−55.5@750 → −87.9@5750, heading to ~−90.6), 1.52s/step, step 0→6722, NaN confined to early
warm-up. Muon seed coord NOT printed to log → the cubic-frac ≈0.7503 seed check still owed at inference.

**Energies (clean block-avg, ΔE vs earlier 07-04 check; NONE converged):**
- #11 si c t_rel @280.4k **−62.92281 ± 0.00016** (−6.8e-8/step), Δ+12.4k / −0.5 mHa — plateau-approaching, NOT settled (slope >2σ), ~9.4 mHa below frozen #2
- #13 d c t_rel @300.2k −90.68258 ± 0.00022 (−6.4e-8/step, slope now statistically FLAT/noise-limited), Δ+22.2k / −3.2 mHa
- #14 d bc_seed @414k **−90.68701 ± 0.00025** (−1.1e-7/step, descending), Δ+20k / −2.0 mHa — **★ now ~18 mHa BELOW #8 off-T −90.669**
- #15 si bc_seed @106k −62.86925 ± 0.00060 (−4.5e-7/step), Δ+12k / −6.5 mHa — node healthy, normal Si range
- #17 si t_seed @134k **−62.87512 ± 0.00049** (−3.5e-7/step), Δ+12k / −4.6 mHa — **★ now edged BELOW frozen #10 −62.87133, firmly lowest quantum Si**
- #18 d t_seed @6k fresh-net warm-up (−88 climbing, tool garbage) — new run

Both comparison tables refreshed. Verdicts: **★ #14 bc_seeded now ~18 mHa below #8 off-T, still
descending** (BC-vs-off-T verdict keeps strengthening); **★ #17 t_seeded now below frozen #10 = lowest
quantum Si**; **✅ #15 resubmit holding** (+12k steps, healthy node); **✅ #18 launched clean as fresh net**.
Watch next check: #18 climbing out of warm-up + its inference-time seed check.

## As of 2026-07-04 (branch `af`) — daily check: 6 RUNNING & healthy; #13/#15 resumed cleanly from 07-03 hang/crawl; queued 4 afterany follow-ups; #14 ~16 mHa below #8 off-T; #17 now lowest quantum Si

**All 6 active jobs RUNNING & healthy** (5485268 #11, 5485269 #12, 5485263 #13, 5485270 #14, 5485264 #15,
5485271 #17; csv fresh 3–57 min, NaN ≤7/last-2000 = warm-up). **07-03's two fixes WORKED:** #13 (5485263)
resumed cleanly from its hang — now @278k, 1.48s/step, well past the 263k hang point; **#15 (5485264)
node recovered — now 2.79s/step (was 30.1s/step crawl), @94k, +8.4k steps since resubmit.** #13/#15
already had pending follow-ups (5485272/5485273); **queued 4 more afterany for the jobs that lacked them:
5491721→#11 (dep 5485268), 5491722→#12 (5485269), 5491723→#14 (5485270), 5491724→#17 (5485271)** — every
active job now has exactly one gap-free continuation.

**Energies (clean block-avg, ΔE vs 07-03):**
- #11 si c t_rel @268k **−62.92232 ± 0.00013** (−1.5e-8/step) — **CONVERGED per tool**, ~9.0 mHa below frozen #2; Δ+22k / −1.4 mHa
- #12 d q t_rel @478k −90.65979 ± 0.00023 (−6.2e-8/step, slope now statistically FLAT/noise-limited), Δ+38k / −2.2 mHa
- #13 d c t_rel @278k −90.67941 ± 0.00030 (−1.8e-7/step, descending; resumed from hang), Δ+14.6k / −2.4 mHa
- #14 d bc_seed @394k **−90.68506 ± 0.00030** (−1.2e-7/step, descending), Δ+38k / −6.1 mHa — **★ now ~16 mHa BELOW #8 off-T −90.669**
- #15 si bc_seed @94k −62.86272 ± 0.00075 (−6.4e-7/step), Δ+8.4k / −5.5 mHa — node recovered, normal Si range
- #17 si t_seed @122k **−62.87053 ± 0.00056** (−4.1e-7/step), Δ+22k / −10.8 mHa — **★ now the LOWEST-energy quantum Si run, ≈ frozen #10 −62.87133**

Both comparison tables refreshed. Verdicts: **★ #14 diamond bc_seeded firmly & increasingly BELOW #8
off-T** (BC lower-E than off-T trap); **★ #17 t_seeded now lowest quantum Si**; **★ #11 CONVERGED**;
**✅ #15 resubmit fixed the crawl**. Watch next check: whether #14 keeps descending.

**POST-CHECK ACTIONS (2026-07-04, user request):**
1. **STOPPED #12 diamond t_relaxed quantum** — cancelled 5485269 + its follow-up 5491722. Final
   −90.65979 ± 0.00023 @478k (was noise-limited, slope flat −6.2e-8). Its question is answered: the
   unseeded muon HELD the expanded relaxed T-cage as muonium ([[project-diamond-t-relaxed-muon-result]]).
   Restart if ever needed: `sbatch pp.sh` from `jobs/diamond_2x2/t_relaxed/`. Now 5 active jobs.
2. **ADDED + LAUNCHED #18 diamond T-seeded** — diamond analogue of #17. `configs/diamond/t_relaxed/
   t_seeded.py` + `jobs/diamond_2x2/t_relaxed/t_seeded.sh` (py_compile clean). Seeds the muon at the
   relaxed T-site (0.7502845a)³ (verified centroid of expanded C #4-#7, 2.9305 bohr equidistant), width
   0.35, fresh net, save `diamond/unpaired/t_seeded/pp` (created empty before submit). **Job 5491822
   (PENDING/Priority), follow-up 5491823.** Back to 6 active jobs. **SANITY CHECK OWED once it RUNS:**
   tail `jobs/diamond_2x2/t_relaxed/t_seeded.out` for "No checkpoint found. Training new model." (fresh
   net, NOT warm restore), muon seeds at cubic-frac ≈0.7503 coordinated by expanded C #4-#7, no early NaN.
   See [[training-monitoring]] #18.
3. **#11 "CONVERGED" is a LOCAL/window artifact — NOT truly converged.** energy_convergence.py's default
   window=20 says CONVERGED, but widening to window=50/100 flips to NOT CONVERGED (slope −5.9e-8 to
   −8.0e-8/step, >2σ from flat). 20k-chunk trajectory is still monotonically descending & decelerating:
   220-240k −62.92067 → 240-260k −62.92186 → 260-280k −62.92239 (~0.5-1.2 mHa lower each 20k block).
   And run-over-run it has dropped ~1.4 mHa/daily-check. **Verdict: plateau-APPROACHING / noise-limited,
   still inching down; treat as a tight upper bound, not settled.** Report this every time #11 is cited.

## As of 2026-07-03 (branch `af`) — daily check: #13 HUNG + #15 CRAWLING (both cancel+resubmit); queued 6 afterany follow-ups; #14 holds ~10 mHa below #8 off-T; #17 caught up to Si range

**Two problems, both fixed (user OK'd).** All 6 showed RUNNING but: **#13 (d classical t_rel, 5463405)
was HUNG** — log+csv frozen @263419 since 01:22 UTC (~10h), `total: 234.7min` at last line, holding 8
nodes (same as the 06-29 #14 hang). **#15 (si bc_seeded, 5463407) was CRAWLING at 30.1s/step (~20×
the normal 1.5s)** — alive/writing but eta ~283 days, a degraded node in its allocation (root cause of
the "abnormally slow step advance" flagged the prior 2 checks; NOT a hang). **Cancelled both, resubmitted
plain (resume from latest ckpt): #13→5485263, #15→5485264** (both PENDING for nodes). Other 4 (#11/#12/#14/#17)
healthy: fresh csv, 1.5–2.8s/step, NaN ≤6/last-2000. Queue had NO follow-ups → **queued 6 afterany: 5485268→#11
(dep 5463403), 5485269→#12 (5463404), 5485270→#14 (5463406), 5485271→#17 (5463408), 5485272→#13 (5485263),
5485273→#15 (5485264).**

**Energies (clean block-avg, ΔE vs 07-02 #2 check):**
- #11 si c t_rel @246k **−62.92093 ± 0.00013** (−2.9e-8/step) — now **CONVERGED per tool (slope flat)**, ~7.6 mHa below frozen #2; Δ+26k / −1.9 mHa
- #12 d q t_rel @440k −90.65759 ± 0.00023 (−8.2e-8/step), Δ+50k / −3.9 mHa
- #13 d c t_rel @263.4k −90.67705 ± 0.00027 (frozen at hang; was −1.3e-7/step), Δ+27.4k / −4.6 mHa
- #14 d bc_seed @356k **−90.67900 ± 0.00041** (−2.4e-7/step), Δ+50k / +0.7 mHa (within SEM; slope still descending) — **★ stays ~10 mHa BELOW #8 off-T −90.669**
- #15 si bc_seed @85.6k −62.85719 ± 0.00093 (−7.7e-7/step), Δ+2.6k / −1.1 mHa — slow advance was the 30s/step node (resubmitted)
- #17 si t_seed @100k **−62.85974 ± 0.00083** (−6.2e-7/step), Δ+26k / −24.2 mHa — **★ fresh-net has CAUGHT UP to normal Si quantum range (~−62.86), now ≈ matching #15**

Both comparison tables refreshed. Verdicts intact: **★ #14 diamond bc_seeded firmly BELOW #8 off-T**;
**★ #17 t_seeded caught up to Si range**; **★ #11 plateaued**. Watch next check: whether resubmitted #15
lands on healthy nodes (~1.5s/step) and #13 resumes cleanly from ckpt.

## As of 2026-07-02 (#2, branch `af`) — daily check: 6 jobs RUNNING with follow-ups ALREADY queued → NO action needed; all healthy

**Nothing to submit.** All 6 active jobs RUNNING (5457805 #11, 5457806 #12, 5457808 #13, 5457814 #14,
5457815 #15, 5457817 #17; 16–17.5h in, 6.5–8h left) and **each ALREADY had exactly one
correctly-chained afterany follow-up PENDING** (verified deps: 5463403→#11, 5463404→#12, 5463405→#13,
5463406→#14, 5463407→#15, 5463408→#17) ⇒ every "about-to-expire" job is covered gap-free, no requeue
needed this check. All healthy: csv fresh 07-02 13:2x–13:5x, NaN ≤6/last-2000 (warm-up level).

**Energies (clean block-avg, NONE converged; ΔE vs 07-02 #1 check):**
- #11 si c t_rel @220k −62.91902 ± 0.00017 (−9.6e-8/step), Δ+14k / ΔE −0.0012; ~5.6 mHa below frozen #2
- #12 d q t_rel @390k −90.65373 ± 0.00022 (−1.0e-7/step), Δ+24k / ΔE −0.0023
- #13 d c t_rel @236k −90.67246 ± 0.00030 (−1.8e-7/step), Δ+26k / ΔE −0.0061
- #14 d bc_seed @306k **−90.67965 ± 0.00025** (−1.5e-7/step), Δ+26k / ΔE −0.0041 — **★ now ~10.6 mHa BELOW #8 off-T −90.669**
- #15 si bc_seed @83k −62.85610 ± 0.00101 (−8.1e-7/step), Δ+1k / ΔE −0.0010 — ⚠️ **step barely moved (82k→83k over 2 checks) — abnormally slow, WATCH**
- #17 si t_seed @74k −62.83555 ± 0.00214 (−1.5e-6/step), Δ+14k / ΔE −0.0218 — fresh-net climbing

Both comparison tables refreshed. Verdicts intact: **★ #14 diamond bc_seeded now firmly BELOW #8
off-T** (BC lower-E than off-T trap), **★ #15 Si bc_seeded recovery holds** (normal Si range, but
watch its slow step advance).

## As of 2026-07-02 (branch `af`) — daily check: 6 jobs RESUBMITTED & RUNNING on re-owned `parv`; queued 6 afterany follow-ups; all healthy

**All 6 active jobs are RUNNING again** (5457805 #11, 5457806 #12, 5457808 #13, 5457814 #14,
5457815 #15, 5457817 #17; 5–7h in) after the 07-01 node-abort deaths — resubmitted (plain sbatch,
resumed from latest ckpt). Queue had NO pending afterany follow-ups → **queued one per job (user
OK'd):** 5463403→#11, 5463404→#12, 5463405→#13, 5463406→#14, 5463407→#15, 5463408→#17 (all
PENDING/Dependency). All healthy: fresh csv writes (07-02 02:3x–03:2x), NaN ≤4/last-2000 rows (warm-up level).

**PATH MIGRATION = DONE.** The parv→parvfection account migration is COMPLETE: resolved by
RE-OWNING `/projects/u6em/parv` to the current account (`ls` shows `parvfection.u6em brics.u6em`),
NOT by copying to a new `/projects/u6em/parvfection` base (which does NOT exist). All 6 configs still
(correctly) point to `parv` and it's writable → live data. `PATH_MIGRATION.md` is now OBSOLETE (can
be deleted). No path/config changes outstanding.

**NEXT STEPS (open):** Fix the conda env in this home directory — `conda`/`conda activate
ferminet-piku` do NOT work in this session's shell (`conda: command not found`, `python: command
not found`). Workaround used this session: call the env python by absolute path
`/home/u6em/parvfection.u6em/miniforge3/envs/ferminet-piku/bin/python` (miniforge3 is at
`/home/u6em/parvfection.u6em/miniforge3`). Need to wire conda init into the shell profile so
`conda activate ferminet-piku` / bare `python` work again. See [[feedback-python-env]].

**Energies (clean block-avg, NONE converged; ΔE vs 07-01 check):**
- #11 si c t_rel @206k −62.91778 ± 0.00016 (−8.7e-8/step), Δ+6k / ΔE −0.0007; ~4 mHa below frozen #2
- #12 d q t_rel @366k −90.65141 ± 0.00024 (−1.0e-7/step), Δ+14k / ΔE −0.0014
- #13 d c t_rel @210k −90.66638 ± 0.00046 (−3.7e-7/step), Δ+12k / ΔE −0.0044
- #14 d bc_seed @280k **−90.67555 ± 0.00039** (−2.4e-7/step), Δ+10k / ΔE −0.0022 — **★ now ~6.5 mHa BELOW #8 off-T −90.669**
- #15 si bc_seed @82k −62.85512 ± 0.00108 (−8.4e-7/step), Δ~0 / ΔE −0.0022 — recovered/holding, but **step barely moved from 07-01 82k (slow restart, watch)**
- #17 si t_seed @60k −62.81381 ± 0.00252 (−2.0e-6/step), Δ+4k / ΔE −0.0135 — fresh-net climbing

Tables refreshed in both comparison memories. Both verdicts intact: **★ #14 diamond bc_seeded stays
BELOW #8 off-T** (BC lower-E than off-T trap), **★ #15 Si bc_seeded recovery holds** (normal Si range).

## As of 2026-07-01 (LATEST, branch `af`) — daily check: QUEUE EMPTY, all 6 active jobs DEAD (node aborts); #14 CROSSED #8 off-T; #15 fully recovered; did NOT resubmit (user said don't)

**Nothing running.** `squeue` empty. All six active jobs are dead — the last follow-ups
**FAILED with exit `6:0` (task/node abort), NOT clean TIMEOUT**, between ~08:23–09:20 UTC
2026-07-01 (5431922 #11 08:44, 5431923 #12 08:34, 5431924 #13 08:29, 5431925/5440929 #14
08:23–08:25, 5426316 #15 09:01, 5426218/5440930 #17 09:18–09:20). Several died in the same
~1h window ⇒ looks like a cluster/node event, not per-run bugs. Checkpoints + train_stats
all INTACT (energies below read cleanly) → a plain `sbatch` from each submit dir will resume
from latest ckpt. **User said DON'T submit new jobs → I did NOT resubmit anything.** Two
inference jobs (5451338 si bc_seeded_inf, 5451339 si c_t_relaxed_inf) were CANCELLED-by-admin
06:56 (account migration housekeeping, per 723892f "before Isambard account migration").

**Energies (clean block-avg via energy_convergence.py, NONE converged):**
- #11 si c t_rel @200k −62.91709 ± 0.00016 (−8.0e-8/step), ~4 mHa below frozen #2 −62.91337
- #12 d q t_rel @352k −90.65002 ± 0.00032 (−1.4e-7/step, descending)
- #13 d c t_rel @198k −90.66204 ± 0.00045 (−3.2e-7/step, descending)
- #14 d bc_seed @269.6k **−90.67340 ± 0.00037** (−1.8e-7/step) **★ CROSSED #8 off-T (−90.669) — now ~4.5 mHa BELOW it**
- #15 si bc_seed @82k **−62.85292 ± 0.00128** (−9.5e-7/step) **★ fully RECOVERED, back in normal Si quantum range (~−62.86)**
- #17 si t_seed @55.8k −62.80028 ± 0.00379 (−2.8e-6/step, fresh-net climbing fast)

**★ #14 diamond bc_seeded has now CROSSED #8 off-T** (−90.67340 vs −90.669, ~4.5 mHa BELOW,
still descending): the BC-holding quantum muon is now LOWER in energy than the off-T trap —
the BC-vs-off-T energy verdict is landing in BC's favour, consistent with the classical PES.
**★ #15 Si bc_seeded fully recovered** (−62.837→−62.853), post-blowup regression gone.
Tables refreshed in both comparison memories. **Repo memory backup `claude_memory/` synced +
pushed this session** (was stale: missing project_silicon_bc_seeded_muon_result.md + old
current_status/energy tables).

## As of 2026-06-30 (branch `af`) — daily check #3: 6 RUNNING healthy; queued #14/#17 follow-ups; #14 ~1.8 mHa from #8 off-T; #15 recovery still holding

**Daily check.** 6 active jobs RUNNING, all healthy (≤7 NaN/last-2000-rows = warm-up only).
#11/#12/#13 had pending afterany follow-ups (5431922/23/24); #15→5426316 already had one;
**#14 (5431925) and #17 (5426218) were running on their former follow-ups with NO continuation
→ queued 5440929 (#14, dep afterany:5431925) and 5440930 (#17, dep afterany:5426218).**

**Energies (block-avg, none converged):** #11 si c t_rel @179.7k −62.91549 ±0.00023 (−1.2e-7,
now ~2 mHa BELOW frozen #2 −62.91337); #12 d q t_rel @316k −90.64729 ±0.00041 (−2.4e-7,
descending); #13 d c t_rel @161k −90.64973 ±0.00048 (−6.0e-7, descending); #14 d bc_seed @232k
−90.66724 ±0.00044 (−2.0e-7), **now only ~1.8 mHa from #8 off-T −90.669**; #15 si bc_seed @60k
−62.83710 ±0.00029 (−2.3e-6); #17 si t_seed @34k −62.75416 ±0.00037 (−7.7e-6, fresh climbing).

**★ #14 diamond bc_seeded about to cross #8 off-T (−90.669)** — closing on a BC-vs-off-T energy
verdict. **★ #15 Si bc_seeded recovery STILL HOLDING** (−62.828→−62.837, descending cleanly, no
re-blow-up). Tables refreshed in both comparison memories.

## As of 2026-06-30 (branch `af`) — daily check #2: 6 RUNNING healthy w/ follow-ups; #15 recovery holding; classical-BC inference DONE @338k

**Daily check.** 6 active jobs RUNNING, each ALREADY had a pending afterany follow-up
(5431922 #11, 5431923 #12, 5431924 #13, 5431925 #14, 5426316 #15, 5426218 #17) →
NO requeue needed this check. All healthy, ≤5 NaN/last-2000-rows (warm-up only).

**Energies (block-avg, none converged):** #11 si c t_rel @172k −62.91564 ±0.00024
(−1.8e-7, now ~2 mHa BELOW frozen #2 −62.91337); #12 d q t_rel @302k −90.64346 ±0.00042
(−1.4e-6, still descending); #13 d c t_rel @146k −90.64169 ±0.00050 (−1.2e-6, warm-up
done); #14 d bc_seed @218k −90.66541 ±0.00045 (−3.5e-7, now only ~3.6 mHa from #8 off-T
−90.669); #15 si bc_seed @54k −62.82833 ±0.00032 (−3.2e-6); #17 si t_seed @26k −62.70285
±0.00042 (−1.6e-5, fresh-net climbing). Tables refreshed in both comparison memories.

**★ #15 Si bc_seeded recovery HOLDING** — continued cleanly post-truncate/restart
(−62.747@42k → −62.828@54k), descending healthily, did NOT re-blow-up. ~35 mHa from
proper Si GSE. **★ #14 diamond bc_seeded ~3.6 mHa from #8 off-T plateau** and still
descending — closing on a BC-vs-off-T energy verdict.

**Classical fixed-BC inference 5431988 COMPLETED @09:56.** Confirmed log loaded
`qmcjax_ckpt_338000` (the converged #4 net, NOT 222k/321999) + dumped all 1000 fresh
positions. **OWED:** re-run `python tools/srpd_extended_radius.py diamond_classical_bc
4.5 90 2` on the 338k net to tighten the +0.53 net-spin excess (positions path in
muon_site_analysis `diamond_classical_bc` case now → the fresh 338k inference dir).

## As of 2026-06-30 (EARLIER, branch `af`) — daily check #1: 6 RUNNING healthy, #15 RECOVERED post-restart; diamond bc_seeded re-analysed at 176k

**Daily check.** 6 active training jobs RUNNING, no sustained NaN (≤5 NaN/last-2000-rows
each, warm-up only). Jobs #11/#12/#13/#14 were running on their old follow-ups (parents
timed out) with NO continuation → **queued 4 new afterany follow-ups: 5431922 (#11),
5431923 (#12), 5431924 (#13), 5431925 (#14).** #15→5426316, #17→5426218 already had theirs.

**★ #15 Si bc_seeded RECOVERED** — yesterday's truncate+restart from ckpt 034000 (job
5426315) WORKED: now @42k −62.74712 ± 0.00783 descending healthily (−5.9e-6/step), did
NOT re-blow-up past the old 35–40k failure point. Back on the pre-blowup trajectory.

**Energies (block-avg, none converged):** #11 si c t_rel @160k −62.91155 ±0.00024
(−1.5e-7, ~2 mHa from #2); #12 d q t_rel @284k −90.64201 ±0.00034 (−1.4e-7); #13 d c
t_rel @128k −90.61454 ±0.00305 (slope flat, noise-limited); #14 d bc_seed @200k
−90.65489 ±0.00094 (slope flat-ish, ~14 mHa from #8 off-T); #15 si bc_seed @42k
−62.74712 (recovered, see above); #17 si t_seed @16k −62.379 (fresh-net warm-up,
climbing). Tables refreshed in [[project-diamond-pp-energy-comparison]] /
[[project-silicon-pp-energy-comparison]].

**Diamond bc_seeded re-analysed at well-trained 176k net** (inference 5426303 dumped
all 1000 positions): muon HOLDS BC, now **0.037 bohr from BC** (circ mean, was 0.30),
RMS 0.61 bohr. Memory [[project-bc-seeded-muon-result]] updated with the 176k re-run
section.

**★ SPIN-DENSITY CORRECTION (2026-06-30): BC is NOT diamagnetic — it's anisotropic
muonium.** The contact-only SRPD (rmax=1) looked diamagnetic (g↑≈g↓ inside 1 bohr =
screening cloud). But `tools/srpd_extended_radius.py` (extended to handle fixed-origin/
classical cases) shows a LOCALIZED net-spin excess of **~+0.47 e⁻ @ ~2.7 bohr** for the
quantum bc_seeded net, and **~+0.53 @ ~2.6 bohr** for the CLASSICAL fixed-BC run (#4,
muon clamped, no ZPM) — both 5–7× above the uniform-band line ⇒ bond-centred (anisotropic)
muonium, an electronic property of BC (not a ZPM artifact). OVERTURNS the old "BC
diamagnetic" verdict in [[project-bc-seeded-muon-result]] + [[project-diamond-classical-bc-muon-result]]
(both updated). Diamond carries muonium at BOTH T (1s-contact) AND BC (bond-orbital);
difference is shape not presence. Run: `python tools/srpd_extended_radius.py bc_seeded
6.0 120 2` / `... diamond_classical_bc 4.5 90 2`.

**Classical fixed-BC inference RE-RUN at converged 338k → job 5431988
`bc_diamond_pp_T_inf` (SUBMITTED).** Reason: the classical positions we had were from
ckpt **222000** (log `Loading .../qmcjax_ckpt_222000.npz`), not the converged 338000 net
(#4 plateau −90.73010). Recovered the inference config+script from git stash cb697d2
(`configs/diamond/bc_relaxed/inference_pp_T.py` + `jobs/.../pp_T_inf.sh` — both had been
deleted from the tree; restore_path = training dir → latest ckpt 338000, save .../classical/
bc/inference, positions=True, srpd off). **Old 222k positions PRESERVED** in
`/projects/u6em/parv/diamond/unpaired/classical/bc/inference/prior_222k_jun15/` (1000
positions + the stale ckpt_000000/ckpt_321999 + train_stats). Cleared the inference root
so find_last_checkpoint→None falls back to 338000. **OWED once 5431988 runs:** confirm
log loads ckpt_338000 (NOT 222000/321999), dumps fresh positions, then re-run
`srpd_extended_radius.py diamond_classical_bc` on the converged net to tighten the +0.53.
NOTE the muon_site_analysis `diamond_classical_bc` case positions path now points at the
FRESH (338k) dir; old 222k are under prior_222k_jun15/.

## As of 2026-06-29 (LATEST, branch `af`) — daily check #3: #14 was HUNG (cancelled), #17 follow-up queued; diamond t_relaxed inference DONE

**Daily check #3.** 6 active training jobs + diamond t_relaxed position inference
(5420036) RUNNING. **#14 diamond bc_seeded (5404793) was HUNG** — train_stats/log
frozen at step 177729 since 04:43 UTC (~11.7h) while SLURM showed it RUNNING and
holding 8 nodes. **Cancelled 5404793** (user OK'd); its afterany follow-up
**5415386 will resume from ckpt 177729**. **#17 si t_seeded (5419590) had no
follow-up → queued 5426218** (dep afterany:5419590). Other active jobs all live
(fresh train_stats writes).

**Energies (block-avg, all NOT converged):** #11 si c t_rel @148k −62.90944
±0.00028 (−1.9e-7); #12 d q t_rel @260k −90.63805 ±0.00035 (−2.2e-7); #13 d c
t_rel @102k −90.58665 ±0.00262 (warm-up tail); #14 d bc_seed FROZEN @177.7k
−90.64741 (hung); #15 si bc_seed @68k −62.12808 ±0.00323 (still ~740 mHa high,
sporadic NaN-resets @69901, regression NOT recovered); #17 si t_seed fresh net
@~3.7k (−61.6 climbing, "No checkpoint found" ✓, 12 early NaN-resets steps
673–3583 = normal warm-up). Muon seed loc not printed to log — confirm at inference.

**Diamond t_relaxed quantum muon inference (5420036) ANALYSED** — see
[[project-diamond-t-relaxed-muon-result]]: muon HOLDS the expanded relaxed T-cage
(centered, +0.017 bohr expansion, no seeding) AND is MUONIUM (g_up≈6:1 contact,
localized_excess plateaus ~+0.4). Opposite of silicon (fled contracted cage,
diamagnetic). OWED next: contact spin density ρ_s(0)=ρ↑(0)−ρ↓(0) extraction (user
asked; note positions .npy shape is (4,128,198) → reshape(-1,66,3), muon=last).

**TWO follow-up actions done (2026-06-29 LATEST):**
1. **Diamond bc_seeded inference RE-RUN at ~176k net → job 5426303 (PENDING).**
   The old positions were from the Jun-27 inference (56k net); training is now
   ~176k. Moved the stale inference outputs (ckpt_000000 + 1000 old positions +
   train_stats) to `.../pp_bc_seeded/inference/prior_jun27_56k/` so
   find_last_checkpoint(save_path)→None falls back to restore_path (training dir,
   latest ckpt 176000). Fresh empty positions/ dir. Config/job unchanged
   (`inference_bc_seeded.py` / `bc_seeded_inf.sh`). **OWED:** confirm it dumps
   positions, then re-analyse muon site/SRPD on the well-trained net (compare to
   the 56k result: muon HOLDS BC, diamagnetic — [[project-bc-seeded-muon-result]]).
2. **#15 Si bc_seeded RESTARTED from last good ckpt 034000 → job 5426315 (Priority)
   + follow-up 5426316.** Reason: #15 descended healthily to −62.758 @33k then
   BLEW UP at 35–40k and stuck flat ~−62.11 for 35k steps (NOT recovering, ~760
   mHa above proper Si GSE). Cancelled the stuck run 5415387 + its follow-up
   5420128. Moved the 25 post-blowup ckpts (035520…070000) to
   `.../silicon_unpaired/bc_seeded/post35k_regression_bak/`, backed up full
   train_stats there (`train_stats_full_to70k.csv`) and truncated the live one to
   step≤34000 (last row 34000 = −62.797). Restart restores ckpt_034000 (seeding
   bypassed on restore). **OWED sanity check once 5426315 RUNS:** log shows
   "Restoring checkpoint …034000" (NOT "Training new model", NOT 070000), energy
   resumes near −62.76, and watch whether it re-blows-up at ~37k or trains through.

Also: #14's hung-job fix WORKED — follow-up 5415386 is RUNNING (resumed from
ckpt 176000).

## As of 2026-06-29 (LATE, branch `af`) — daily check #2: 6 RUNNING + #17 PENDING; #15 follow-up queued; #15 regression CONFIRMED but recovering slowly

Queue: 6 RUNNING training jobs (#11/#12/#13/#14 each w/ PENDING afterany follow-up
5415383/84/85/86), **#15 RUNNING on 5415387 with NO follow-up** (its parent 5402724
timed out so the follow-up took over), **#17 `muon_silicon_q_t_seeded` 5419590 PENDING/
Priority (first launch, not started)**, + inference 5420036 `muon_d_qpp_t_rel_inf`
PENDING. **Queued #15 follow-up: 5420128 (dep afterany:5415387).** User chose to LEAVE
#15 running and watch (not kill/restart).

**⚠️ #15 Si bc_seeded regression CONFIRMED, not recovered:** stuck ~740 mHa ABOVE the
proper Si GSE — energy_convergence @62k = −62.12045 ± 0.00473 vs all other Si runs
~−62.86–62.91. This-job span 61050 (−62.133) → 63138 (−62.122), basically flat at
−62.12. Trailing-window slope now mildly NEGATIVE (−3.1e-6/step, inching back down)
but from a badly destabilized state, nowhere near healthy. NOT normal warm-up; diamond
analogue #14 is fine so it's silicon-bc_seeded-specific. Watch next check.

**Energies refreshed** ([[project-diamond-pp-energy-comparison]],
[[project-silicon-pp-energy-comparison]]): #14 bc_seeded −90.64741 @177k (−4.5e-7,
~22 mHa from #8 off-T), #12 d t_rel −90.63569 @248k (−2.1e-7), #13 d c t_rel
−90.55766 @90k (warm-up −2.8e-6), #11 si c t_rel −62.90756 @140k (~6 mHa from #2),
#15 si bc_seeded −62.12045 @62k (see flag), #17 si t_seeded no train_stats yet
(PENDING). None converged. No sustained NaN. #10 si q t_rel CLOSED (frozen
−62.87133); superseded by #17 t_seeded.

## As of 2026-06-29 (branch `af`) — Diamond #12 T-relaxed muon-site inference launched

Verified from configs: diamond T-cage **EXPANDS +0.41%** (2.9185→2.9305 bohr, C #4-7
move outward 0.011-0.015 bohr) vs silicon CONTRACTS −1.10% (4.4427→4.3940, Si inward
0.0487). Physics: expand/contract tracks muonium(neutral Mu⁰, steric push-out, tight
diamond cage) vs diamagnetic(screened H⁺, attractive pull-in, open Si cage). Prediction:
diamond quantum muon should HOLD the relaxed (expanded) T-site WITHOUT seeding (relaxation
works WITH the muon), unlike silicon. **Launched inference to test: job 5420036
`muon_d_qpp_t_rel_inf` (PENDING)** off #12 ckpt (step 246000). New files (all verified,
geometry byte-identical to training, fresh inference dir w/ only positions/ subdir, no
stale ckpt → falls back to restore): `configs/diamond/t_relaxed/inference_pp.py`,
`jobs/diamond_2x2/t_relaxed/inference_pp.sh`. Added `diamond_t_relaxed` case to
`tools/muon_site_analysis.py` (C_T_RELAXED geom, a=6.74, bc_pair (0,8)). **Next:** once
5420036 dumps positions_*.npy → `python tools/muon_site_analysis.py diamond_t_relaxed
{site|spread|srpd}` + the relaxed-vs-unrelaxed-cage min-image check (does it sit at the
expanded relaxed cage, C #4-7 the movers, vs a different T-site like silicon did?).

## As of 2026-06-29 (branch `af`) — EXP-003b: #10 CLOSED, T-SEEDED run #17 launched

After the EXP-003 site analysis (muon avoids the relaxed cage), **stopped #10 Si
T-relaxed quantum** (cancelled training 5402683 + follow-up 5415382 + redundant
inference 5415683); kept classical #11 (5402686) as the clean relaxation-energy
comparator. **Launched #17 `muon_silicon_q_t_seeded` (job 5419590, PENDING)** —
EXP-003b, the T-site analogue of bc_seeded: seeds the muon at the relaxed T-site
(0.75a)³ (verified centroid of the 4 contracted Si, 4.394 bohr equidistant, offset
1.7e-5 bohr), width 0.5, fresh net, save `silicon_unpaired/t_seeded`. New files:
`configs/silicon/t_seeded.py` + `jobs/silicon/t_seeded.sh` (py_compile clean).
Tests whether the muon HOLDS the relaxed cage when seeded inside, vs escaping to an
unrelaxed T-site as #10 did. Added as #17 in [[training-monitoring]]. **Sanity check
owed once it RUNS:** fresh-net log line, muon at cubic-frac ≈0.75 coordinated by the
moved Si, no early NaN. (Energy puzzle resolved: t_relaxed's apparent 10 mHa edge
over #1 was convergence-inflated; classical #11-vs-#2 matched-step shows the true
relaxation energy ≈5–6 mHa. No paradox — relaxing a defect lattice lowers E by design.)

## As of 2026-06-29 (branch `af`) — EXP-003 ANALYSED: silicon T-relaxed = DIAMAGNETIC (H0 confirmed)

**EXP-003 DONE.** Job 5415683 wrote all 1000 positions (512k samples). Analysed:
muon at **T-site** (cubic-frac 0.75, 4 Si ~4.2–4.6 bohr), RMS 0.81 bohr anisotropic
(like other q-Si). Contact SRPD ≤1 bohr **spin-unpolarised** (g_up≈g_down; at r≈0.08
up 0.142 < down 0.169). Extended-radius net-spin: `localized_excess` ≈0/slightly
NEGATIVE within lattice scale (−0.002→−0.024 to 2.7 bohr), drifts to +0.15 only by
5.7 bohr (above lattice scale, tracks uniform band) — **no localized bound electron.**
**VERDICT: H0 confirmed — Si stays DIAMAGNETIC. ★ BUT MAJOR SITE CAVEAT:** the quantum
muon does NOT localise in the RELAXED cage. Relaxed Si = {4,5,6,7} (moved 0.0487 bohr
inward, cage centre cubic-frac (0.75,0.75,0.75)=7.695 bohr). Muon's 4 coordinating Si
= {15,11,14,10}, ALL ≈UNMOVED (0.0002 bohr) → an UNRELAXED T-site. Muon centre 8.47
bohr (min-image, FCC vecs [[a,a,0],[0,a,a],[a,0,a]]) from relaxed site; 0% of samples
within 3.5 bohr, closest 6.99 bohr → P(at relaxed site)≈0. WHY: DFT relaxation
CONTRACTED the cage (−1.1%, for a classical proton); quantum muon w/ ZPM prefers a
roomier unrelaxed T-site (classical-H ≠ quantum-muon site). CONSEQUENCE: diamagnetic
verdict valid, but the INTENDED "does tighter relaxed cage bind?" test was NOT
performed — muon never enters the cage. To test properly: SEED muon in relaxed cage
(EXP-002 bc_seeded style). Caveat 2: #10 not converged (−62.871 @130k) → re-run inf
at later ckpt; cross-check classical #11 fixed-H SRPD. Doc updated w/ "Site caveat" §.

## As of 2026-06-29 (branch `af`) — EXP-003 launched: silicon T-relaxed bound-state inference

NEW experiment **EXP-003** (`experiments/EXP-003_silicon_t_relaxed_bound_state.md`):
does the silicon **T-relaxed** quantum muon (#10) form a bound state (muonium), or
stay DIAMAGNETIC like ideal/BC silicon? Position-inference **job 5415683
`muon_silicon_q_t_relaxed_inf` (PENDING/Priority at launch)**. Created (all
verified): `configs/silicon/inference_t_relaxed.py` (geometry byte-identical to
training `t_relaxed.py`, (33,32,1), optimizer none, fake_energy, positions=True,
restore `/projects/u6em/parv/silicon_unpaired/t_relaxed` latest ckpt 130000, save
`…/t_relaxed/inference` empty→falls back, no stale ckpt), `jobs/silicon/
inference_t_relaxed.sh` (8×4). Pre-created `…/inference/positions/` (avoids EXP-002
mkdir crash). Added `silicon_t_relaxed` case to `tools/muon_site_analysis.py`
(SI_T_RELAXED geom, a=10.26, bc_pair (0,8)). **Next:** confirm 5415683 dumps
positions_*.npy (early log 65-e⁻ restore, no NaN), then `python tools/
muon_site_analysis.py silicon_t_relaxed {srpd|site|spread}` — srpd is the
bound-state check (g_up≫g_down at r→0 = muonium). CAVEAT: #10 not converged
(−62.871 @130k, still descending) → site/SRPD read on partial net.

## As of 2026-06-28 (branch `af`) — daily check: 6 RUNNING healthy, #12/#14 follow-ups queued

Queue was clean (no out-of-queue active jobs). **6 RUNNING, all healthy, no
sustained NaN:** #10 si q t_rel (5402651 @102.6k −62.850 v0.0001), #11 si c t_rel
(5402661 @103.5k −62.888 v0.0001), #12 d q t_rel (5402704 @180.6k −90.638 v0.0003),
#13 d c t_rel (5402669 @21.1k −90.28 v0.0046 fresh warm-up), #14 d bc_seeded
(5402711 @124.2k −90.586 v0.0005), #15 si bc_seeded (5395609 @26.2k −62.74 v0.0002
fresh warm-up).

**Follow-up gap fixed:** #12 (5402704) and #14 (5402711) were running on what had
been their own afterany follow-ups (parents 5391699/5391700 timed out), so had no
chain left. Submitted **5404792** (→#12, dep afterany:5402704) and **5404793**
(→#14, dep afterany:5402711), both PENDING/Dependency. The other four (#10/#11/#13/#15)
already had follow-ups (5402683/5402686/5402701/5402724). Every active job now
continues gap-free.

**Energies refreshed** ([[project-diamond-pp-energy-comparison]],
[[project-silicon-pp-energy-comparison]]): #12 t_relaxed −90.61085 (+60 mHa since
06-27), #14 bc_seeded −90.60321 (+123 mHa, now ~66 mHa from #8 off-T −90.669),
#11 si c t_rel −62.89578, #10 si q t_rel −62.85998 (now caught up to #1/#9 quantum
Si). #13/#15 still fresh-net warm-up (tool garbage; live inst −90.28/−62.74).
None converged.

## As of 2026-06-26 (eve, branch `af`) — #7 diamond classical T CLOSED + SRPD/positions inference

**#7 diamond classical T-site (job 5380708) CLOSED as CONVERGED** (E −90.69624 ±
0.00015 @472k, slope flat) — cancelled. Set up + launched its **SRPD+positions
inference: job 5391680** (`inf_muon_diamond_tsite`, config
`configs/diamond/classical_muon/t_site_inference_pp.py`, script
`jobs/diamond_2x2/classical_muon/inference_pp.sh`). Config was already correct
(65 e⁻ (33,32), restore_path `.../classical/T_site/pp`, save_path `.../inference`,
SRPD on with fixed origin at T-site [0.75a]³) — I ADDED `cfg.observables.positions
= True` (committed ba03a98, pushed). **Critical fix:** the inference dir held a
STALE `qmcjax_ckpt_000000.npz` from the Jun-21 run (a non-converged ~256k net);
find_last_checkpoint(save_path) would have restored THAT instead of falling back
to restore_path. Moved all Jun-21 outputs (ckpt + srpd_1..17 + train_stats) to
sibling `.../classical/T_site/pp/inference_prior_jun21_256k/`, so inference now
correctly falls back to the 472k training ckpt. positions/ subdir pre-created (and
train.py auto-creates it now too, commit 5442c07). **Next:** check 5391680 dumps
srpd_*.txt + positions/positions_*.npy, then analyse (muon_site_analysis.py
`diamond_classical_bc` is the BC analogue; this is the T-site classical SRPD).

## As of 2026-06-26 (eve, branch `af`) — diamond bc_seeded inference CRASHED, fixed & resubmitted

The diamond bc_seeded (EXP-002) muon-position inference job **5380879
`muon_d_qpp_bc_seeded_inf` FAILED in 45 s** (16:43): `FileNotFoundError` on
`.../pp_bc_seeded/inference/positions/positions_0.npy` — the `positions/` subdir
did not exist and the position writer does NOT mkdir it. ZERO positions dumped →
muon site still UNKNOWN. Fix: `mkdir -p .../pp_bc_seeded/inference/positions`,
then resubmitted as **job 5391609**. Inference restores the latest training ckpt
at launch (training was at step ~56000 when resubmitted; net NOT converged, so the
site check is on a partially-trained net). The KEY EXP-002 question: does the
BC-seeded quantum muon HOLD the BC site (classical DFT says BC is true min, 70 mHa
below off-T) or drift back to the off-T trap? **Next:** confirm 5391609 dumps
positions, then `python tools/muon_site_analysis.py` (add a `bc_seeded` case —
carbons = C_BC_RELAXED_2 geometry, bc_pair (0,1), a=6.74; positions dir
`.../pp_bc_seeded/inference/positions`). See [[project-bc-relaxed-pp-relax-2-muon-result]].

## As of 2026-06-26 (branch `af`) — silicon bc_relaxed position-inference queued

Daily check + new inference. Live queue: **2 RUNNING** (#12 `muon_d_qpp_t_rel`
5370051 @93.8k −90.53 var0.0007, now past warm-up; #14 `muon_d_qpp_bc_seeded`
5370014 @37.6k −90.36 var0.0011 — **EXP-002 fresh-net check PASSES**: log shows
"No checkpoint found. Training new model.", no NaN, still climbing). **PENDING
(resubmitted/timed-out):** #7 5380708 (@467k), #9 5380709 (@127k), #10 5380707
(@61k), #11 5380706 (@61k), plus diamond bc_seeded inference 5380879.

**NEW: silicon bc_relaxed muon-POSITION inference SUBMITTED — job 5385972
`muon_silicon_q_bc_relaxed_inf` (PENDING).** Wrote fresh files (the existing
`configs/silicon/inference.py` is UNUSABLE — wrong *unrelaxed* ideal geometry,
64 e⁻ `(32,32,1)`, stale `/projects/u6em/parv/silicon/` path):
- `configs/silicon/inference_bc_relaxed.py` — bc_relaxed geometry copied VERBATIM
  from training `bc_relaxed.py` (verified byte-identical, `(33,32,1)` 65 e⁻, NET 0),
  `optimizer="none"`, `mcmc.fake_energy=True`, `observables.positions=True`.
- `jobs/silicon/inference_bc_relaxed.sh` (8×4 GPU, output `inference_bc_relaxed.out`).
- restore_path `/projects/u6em/parv/silicon_unpaired/bc_relaxed` (latest ckpt 126000
  at submit); save_path `…/bc_relaxed/inference` (created EMPTY → find_last_checkpoint
  falls back to restore_path; positions → `…/inference/positions/positions_{t}.npy`;
  training train_stats/ckpts untouched). Mirrors the diamond `inference_bc_seeded.py`
  pattern. CAVEAT: #9 still descending (not converged) → site check is on the
  current net, muon site could shift. **Next:** check early log (65-e⁻ restore, no
  NaN) + analyse positions once it runs (tools/muon_site_analysis.py).

## >>> NEXT ACTION (top of queue) <<<

**EXP-002 code is IMPLEMENTED, committed & pushed (branch `af`, commit `f14e142`,
2026-06-24).** All 4 changes done + verified (py_compile clean; CPU numerical test
confirms muon seeded at (0.843,0.843,0.843) std≈0.3, electrons untouched, last
particle = muon species, default no-seed path unchanged):
1. `init.py init_electrons` — `muon_init_coord`/`muon_init_width` kwargs override
   ONLY the last particle (muon); `None` default preserves every existing run.
2. `init.py init_mcmc_data` — passes `cfg.mcmc.muon_init_coord/width` through.
3. `base_config.py` mcmc — `muon_init_coord=None`, `muon_init_width=0.5` defaults.
4. `configs/diamond/bc_relaxed/bc_seeded.py` + `jobs/.../bc_seeded.sh` — pp_relax_2
   geometry, muon seeded at BC (0.84282 bohr/comp, width 0.3), fresh net
   (`restore_path==save_path==.../pp_bc_seeded`, empty ⇒ "Training new model"),
   `load_data` left default True.

**LAUNCHED 2026-06-24: job 5370014 `muon_d_qpp_bc_seeded` (PENDING, 8 nodes).**
muon_init_width kept at 0.3 (decided: init width only sets the START cloud, does
NOT confine the muon — MCMC+ψ govern later motion; 0.3 keeps the seed cloud well
inside the BC basin, ~1.2 bohr clear of the C1 watershed at 2.06 bohr, and away
from the carbon cusp). save dir `/projects/.../bc_relaxed/pp_bc_seeded` created
EMPTY → fresh net guaranteed. Added to [[training-monitoring]] as job #14 (ACTIVE).

**STILL OWED once it RUNS (the pre-launch sanity check, now done in-job):** tail
`jobs/diamond_2x2/bc_relaxed/bc_seeded.out` and confirm (a) "No checkpoint found.
Training new model." — NOT a warm-start restore (EXP-002 risk #1); (b) the muon
localises at BC (≈0.125 cubic-frac) not off-T; (c) no early NaN blow-up (risk #4).
Then track VMC energy vs the off-T #8 value −90.669 Ha. load_data left True (risk
#2). Full protocol + risks in `experiments/EXP-002_bc_seeded_muon_run.md`. See
[[project-bc-relaxed-pp-relax-2-muon-result]]. (Optional parallel: multi-seed
pp_relax_2 reruns = EXP-003 to separate SSB vs representability bias.)

## As of 2026-06-24 (eve, branch `af`) — #8 stopped, bc_seeded launched, silicon bc_seeded staged

- **#8 bc_rel_2 STOPPED & marked COMPLETED** (cancelled pending continuation
  5365663). Final E −90.66904 ± 0.00029 @280196 — lowest quantum diamond run but
  muon localised OFF-T not BC, so its question is answered; BC now tested by #14.
  Restart if needed: `sbatch pp_relaxed_2.sh` from `jobs/diamond_2x2/bc_relaxed/`.
- **#14 diamond bc_seeded LAUNCHED** (job 5370014, PENDING). See NEXT ACTION.
- **#15 silicon bc_seeded STAGED, NOT STARTED** — `silicon/bc_seeded.py` +
  `jobs/silicon/bc_seeded.sh` (muon at BC 0.125*a, width 0.3, fresh net, save
  `silicon_unpaired/bc_seeded`), committed `2d30101`. Launch ONLY after #14 is
  verified to seed/hold BC. Added to [[training-monitoring]] + [[job-save-paths]].
- **Only 2 jobs actually RUNNING right now:** 5358557 #12 diamond t_relaxed q
  (8h, WARM-UP ~18.6k, E −89.x garbage) and 5344741 #9 silicon bc_relaxed q
  (19.7h, −62.857 @116k still descending). PENDING: 5365660 #10, 5365661 #11,
  5365662 #7-cont, 5370014 #14, 5365664 #9-followup(dep).
- Energy tables refreshed: [[project-diamond-pp-energy-comparison]],
  [[project-silicon-pp-energy-comparison]].

## As of 2026-06-24 (branch `af`) — pp_relax_2 muon-site inference ANALYSED ✓

Inference job 5360284 (`muon_d_qpp_bc_rel_2_inf`) RUNNING & already dumped all
1000 positions files (512k samples). Verified: training `train_stats.csv`
(14 MB, 280198 rows, last step 280196 E=−90.65) is INTACT — inference writes its
own throwaway 55-byte train_stats into the separate `inference/` subdir, no
overwrite. **Result: muon → T-site (0.32 bohr from ideal T, 4.06 from BC),
DIAMAGNETIC (no muonium).** Full details + new analysis case in
[[project-bc-relaxed-pp-relax-2-muon-result]].

## As of 2026-06-24 (branch `af`) — daily check: 3 RUNNING, silicon T-relaxed pair recovered

Daily check. #7/#8/#9 (jobs 5344739/40/41) RUNNING healthy, no NaNs:
| # | Config | Job | Step | E_h | var |
|---|---|---|---|---|---|
| #7 diamond classical T-site | 5344739 | 389715 | −90.68 | 0.0002 |
| #8 diamond bc_rel_2 q | 5344740 | 258694 | −90.68 | 0.0003 |
| #9 silicon bc_relaxed q | 5344741 | 97524 | −62.86 | 0.0001 |

**Silicon T-relaxed pair (#10/#11) had FAILED:** originals 5334500/5334501 TIMEOUT
21:19, auto-resubmit 5356791 then ABORTED 23:35 (node task aborts). Were OUT of
queue with no follow-up. Ckpts intact ~step 30486 (q) / 30760 (classical).
**Resubmitted** plain: 5365660 (#10 q), 5365661 (#11 classical) — restart from ckpt.

**Queued afterany follow-ups** (user OK'd all three): 5365662→#7 (dep 5344739),
5365663→#8 (dep 5344740), 5365664→#9 (dep 5344741).

**#12 diamond T-relaxed q (5358557) STILL PENDING** — never started since 2026-06-23
submit (FIFO/backfill, waiting on nodes). **#13 diamond T-relaxed classical still
NOT submitted** (user declined launching it today). New inference job 5360284
(`muon_d_qpp_bc_rel_2_inf`) sits PENDING — not part of training set.

**Diamond GSE refreshed** ([[project-diamond-pp-energy-comparison]]): #7 classical T
= −90.69101 ± 0.00029 (still −1.3e-7/step), #8 bc_rel_2 = −90.66535 ± 0.00037
(still −2.6e-7/step). KEY: #8 has now edged BELOW converged #6 unrelaxed (−90.66312)
and is still dropping → #8 (new BC) is now the lowest quantum run, as matched-step
predicted.

## As of 2026-06-23 (branch `af`) — daily check: 5 active jobs RUNNING

All 5 active training jobs RUNNING, no NaNs. Silicon T-relaxed pair (#10/#11,
jobs 5334500/5334501) started cleanly and is ~5h40m in. The 3 older active jobs
each already have an afterany follow-up PENDING (Dependency): 5344739→#7,
5344740→#8, 5344741→#9 (queued in a prior session).

| Config | Job | Step | E_h | var |
|---|---|---|---|---|
| #7 diamond classical T-site | 5318071 | 343455 | −90.676 | 0.0002 |
| #8 diamond bc_rel_2 q | 5318070 | 216040 | −90.648 | 0.0002 |
| #9 silicon bc_relaxed q | 5318069 | 85979 | −62.837 | 0.0002 |
| #10 silicon t_relaxed q | 5334500 | 7252 | −62.32 | 0.0004 |
| #11 silicon t_relaxed classical | 5334501 | 7292 | −62.53 | 0.0002 |

**LOG-PATH FIX:** #7's live StdOut is `classical_muon/t_site.out` (step 343k),
NOT `t_site_pp.out` (stale, cancelled Jun-19 at step 180k). Memory table fixed.

Refreshed diamond GSE ([[project-diamond-pp-energy-comparison]]): #7 classical T
= −90.68720 ± 0.00020 (still dropping −9e-8/step), #8 bc_rel_2 = −90.65399 ±
0.00050 (still dropping −3.5e-7/step). Both NOT converged.

**Diamond T-relaxed pair (#12/#13) still NOT STARTED** — silicon pair is now well
underway, so these can be launched. See [[project-t-relaxed-runs]].

## As of 2026-06-22 (branch `af`) — T-relaxed configs added; silicon pair submitted

Git pull added 4 T-relaxed configs+scripts (commit `2a3d5a8`). Verified all four
(syntax + charge/geometry sanity: 65 e⁻ doublet, NET 0, classical fixed-H at
quantum T-site, Si nn≈4.394 / C nn≈2.930 bohr). Created the 4 empty save dirs
(no collisions; lowercase `t_relaxed`, distinct from empty leftover capital
`T_relaxed` dirs). **Submitted the two SILICON jobs** (PENDING at submit):
- **5334500** `muon_silicon_q_t_relaxed` (quantum), save `silicon_unpaired/t_relaxed`
- **5334501** `muon_silicon_c_t_relaxed` (classical), save `silicon_unpaired/classical/t_relaxed`
Both added to the [[training-monitoring]] active daily set (#10/#11).

**Diamond T-relaxed pair (#12 quantum, #13 classical) = RUN NOT STARTED** — configs
verified, scripts ready (`jobs/diamond_2x2/t_relaxed/{pp,classical}.sh`), save dirs
created. Start only after the silicon pair is underway. See [[project-t-relaxed-runs]].

Queue at submit time also had running: 5318069 (si bc_relaxed q), 5318070
(diamond bc_rel_2 q), 5318071 (diamond tsite classical), + 2 inference jobs
(5317932 diamond, 5318061 diamond tsite).

**Next:** confirm silicon 5334500/5334501 start RUNNING (no NaN early); when well
underway, launch diamond #12/#13.

## As of 2026-06-21 (branch `af`) — diamond unrelaxed pp SRPD inference jobs queued (quantum + classical)

Queued **two** diamond unrelaxed-pp SRPD inference jobs, both mirroring the
silicon fix pattern (correct e⁻ count + restore from trained ckpt + fresh empty
save dir so `find_last_checkpoint(save_path)`→None falls back to `restore_path`).
SRPD writes `srpd_<N>.txt` (every save_freq) + `srpd_final.txt` into `save_path`.

1. **Quantum muon** — job **5317932** (`muon_2x2_diamond_inference`,
   `jobs/diamond_2x2/muon_inference.sh` → `configs/diamond/test_run_inference.py`).
   Training run **plateaued & cancelled** (latest ckpt 522000). Fixes: particles
   `(32,32,1)`→`(33,32,1)` (66 particles, verified vs ckpt positions 198/3);
   restore_path→`/projects/u6em/parv/diamond/unpaired/unrelaxed/pp`; save_path→
   `.../unrelaxed/pp/inference`. That inference dir had a STALE Jun-15 run — I
   **deleted its `qmcjax_ckpt_*.npz`** (else it'd restore ckpt_099999 not 522000);
   left old `srpd_*.txt` (will be overwritten, user OK'd) + `positions/` + train_stats.
   SRPD uses default `use_fixed_origin=False` (moving muon).

2. **Classical T-site muon** — job **5318061** (`inf_muon_diamond_tsite`,
   `jobs/diamond_2x2/classical_muon/inference_pp.sh` →
   `configs/diamond/classical_muon/t_site_inference_pp.py`). Training (#7,
   job 5310346) **STILL RUNNING** (ckpt ~256000) — inference restores whatever's
   latest at launch. Fixes: particles `(32,32)`→`(33,32)` (65 e⁻, verified 195/3);
   restore_path→`/projects/u6em/parv/diamond/unpaired/classical/T_site/pp`;
   save_path→`.../T_site/pp/inference` (created empty via mkdir). SRPD
   `use_fixed_origin=True`, origin_coord = H T-site [0.75a]³.

**Next:** check both jobs' `srpd_*.txt` output once they run.

## As of 2026-06-20 (later, branch `af`) — daily check + afterany follow-ups queued

Daily check: 4 active jobs all RUNNING, no NaNs:
- #6 diamond unrelaxed pp q — step 477k, E−90.64, var 0.0002 (plateau-ish). job 5301535
- #7 diamond classical T-site pp — step 180k, E−90.66, var 0.0008. job 5298307 (~2:49 left)
- #8 diamond bc_relax_2 q (NEW BC) — step 82.9k, E−90.53, var 0.0011, still climbing. job 5301537
- #9 silicon bc_relaxed q — step 9.2k, E−62.43, var 0.0008, early warm-up. job 5298184

**Queued afterany follow-ups** (one per active job, PENDING/Dependency):
5310345→#6 (dep 5301535), 5310346→#7 (dep 5298307), 5310347→#8 (dep 5301537),
5310348→#9 (dep 5298184). Each continues gap-free when its parent ends.

**Queuing note (this cluster):** `scontrol show config` → PriorityWeightAge=0
(and FairShare/QOS/JobSize all 0, no ACCRUE_ALWAYS) → effectively FIFO/job-id +
backfill. So afterany gives NO priority/age benefit and does NOT reserve a node
slot while held; its only value is operational (hands-off gap-free continuation,
no checkpoint race, and immediate backfill into the parent's just-freed 8 nodes).

**Diamond BC compare (new vs old):** #8 `bc_relaxed/pp_relax_2` (charged-state
re-relaxed, more-expanded BC; save `.../bc_relaxed/pp_relax_2`) vs #5
`bc_relaxed/pp` (old BC, CLOSED, plateaued ewmean −90.599 @336k). Using ewmean:
at matched step ~82k, #8 = −90.522 vs #5 = −90.455 → #8 ~67 mHa LOWER and still
descending. Still 254k steps from #5's plateau — premature to compare final
plateaus or re-run muon-site analysis. (Raw `energy` col is noisy; use col3 ewmean.)

**Silicon SRPD inference** still PENDING/Priority — job number is now **5308478**
(`inf_t_site.sh`, name `inf_muon_silicon_tsite`); not yet started. Config fixes
from earlier today already applied (particles (33,32), restore_path
`/projects/u6em/parv/silicon_unpaired/classic`, fresh save
`.../classic/inference_srpd`). **Next:** check its output / srpd_*.txt once it runs.

## As of 2026-06-20 (earlier, branch `af`) — silicon classical T-site finished + SRPD inference

Daily check: 4 active jobs healthy (#6 diamond unrelaxed pp ~step 458k E−90.65,
#7 diamond classical T-site pp ~step 180k E−90.66, #8 diamond bc_relax_2 ~step
63k E−90.42 still climbing, #9 silicon bc_relaxed pp PENDING/warming up). No NaNs.

**Silicon classical T-site (#2) FINISHED** — cancelled job 5301536 (~step
223695, E−62.92). Submitted SRPD inference job (`inf_t_site.sh`).
Had to FIX `configs/silicon/t_site_inference.py` first: it had `particles=(32,32)`
(64 e⁻) but the trained net is 65 e⁻ → changed to `(33,32)`; and `save_path`
pointed at an unrelated Jun-10 `silicon/T_classical` run (would restore the wrong
checkpoint) → now `restore_path=/projects/u6em/parv/silicon_unpaired/classic`
(latest ckpt 223695) + fresh `save_path=.../classic/inference_srpd`. SRPD was
already enabled (fixed origin at T-site [0.75a]³). NOTE the prior Jun-15
`.../classic/inference/` dir has NO srpd_*.txt — SRPD never actually ran before
for this run.

## As of 2026-06-16 (branch `af`) — daily training requeue

Ran the daily [[training-monitoring]] check. 6 of the 7 production training
jobs had hit their 1-day time limit overnight (TIMEOUT between 08:52 and
09:21); the 7th (diamond classical T-site PP, `t_site_pp.sh`, job 5251125)
was still RUNNING with ~1h left.

Latest step / energy / variance at requeue time:

| Config | Step | E_h | var | Action |
|---|---|---|---|---|
| Silicon q unrelaxed (`silicon/muon.sh`) | 126637 | -62.86 | **nan** | requeued 5266687 |
| Silicon classical t-site (`silicon/t_site.sh`) | 126766 | -62.90 | 0.0001 | requeued 5266688 |
| Diamond bc_relaxed nopp (`bc_relaxed/nopp.sh`) | 730139 | -609.26 | 0.0014 | requeued 5266689 |
| Diamond bc_relaxed classical pp_T (`bc_relaxed/pp_classic.sh`) | 237521 | -90.74 | 0.0002 | requeued 5266690 |
| Diamond bc_relaxed pp q (`bc_relaxed/pp.sh`) | 280433 | -90.60 | 0.0004 | requeued 5266691 |
| Diamond unrelaxed pp (`diamond_2x2/muon.sh`) | 280481 | -90.65 | 0.0002 | requeued 5266692 |
| Diamond classical t-site pp (`classical_muon/t_site_pp.sh`) | 54848 | -90.44 | 0.0008 | RUNNING 5251125; follow-up queued afterany → 5266693 |

The 6 dead jobs submitted as plain `sbatch` (PENDING/Priority). The running
7th's follow-up (5266693) queued with `--dependency=afterany:5251125`
(PENDING/Dependency) so it starts gap-free without racing the live job's
checkpoint/log.

**Still open:** silicon q-unrelaxed variance is *still* `nan` (energy fine at
-62.86) — the known [[potential-bugs]] isnan-vs-isfinite issue, not yet fixed.

**Next session:** repeat the daily check; expect these 6 to have timed out
again ~24h later and 5266693 to have started once 5251125 ended.
