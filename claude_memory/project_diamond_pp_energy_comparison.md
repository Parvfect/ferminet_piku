---
name: project-diamond-pp-energy-comparison
description: "Diamond PP-run GSE comparison table + how to refresh it; use whenever the user asks to compare diamond energies"
metadata:
  node_type: memory
  type: project
  originSessionId: diamond-energy-compare
---

## Diamond PP runs — ground-state energy comparison

When the user asks for "diamond energy comparison" / "GSE of diamond PP runs",
produce a table of plateau energies from `tools/energy_convergence.py`
([[tool-energy-convergence]]) over the [[job-save-paths]] dirs. Values below
are plateau (trailing-window mean) ± 1 SEM.

★ marks ACTIVE (still-training) runs; unmarked rows are frozen/closed. Always
keep this ★-on-active convention ([[feedback-mark-active-runs-with-star]]).

| Run | Muon | Geometry | save_path tail | Step | Plateau E (E_h) | Status (as of 2026-06-27) |
|---|---|---|---|---|---|---|
| #4 classical BC | fixed @ BC | bc_relaxed | `classical/bc` | 338k (frozen) | **−90.73010 ± 0.00029** | CONVERGED (frozen) |
| #7 classical T-site ★ | fixed @ T | unrelaxed | `classical/T_site/pp` | 472k | **−90.69624 ± 0.00015** | **CONVERGED** (slope flat −2.6e-8/step), still RUNNING |
| #8 bc_rel_2 NEW | quantum | bc_relaxed (charged-state, more-expanded BC) | `bc_relaxed/pp_relax_2` | 280k (frozen) | **−90.66904 ± 0.00029** | **STOPPED/COMPLETED 2026-06-24 @280196** (muon localised OFF-T not BC; was still −1.5e-7/step) |
| #6 unrelaxed | quantum | unrelaxed | `unrelaxed/pp` | 522k (frozen) | **−90.66312 ± 0.00017** | CONVERGED (frozen) |
| #5 bc OLD | quantum | bc_relaxed (original) | `bc_relaxed/pp` | 336k (frozen) | **−90.59794 ± 0.00151** | plateaued/noise-limited, CLOSED (frozen) |
| #12 t_relaxed | quantum | t_relaxed | `t_relaxed/pp` | 478k (frozen) | **−90.65979 ± 0.00023** | **STOPPED 2026-07-04** (noise-limited, slope flat −6.2e-8); muon HELD expanded relaxed cage as muonium → question answered, superseded by T-seeded #18 |
| #14 bc_seeded ★ | quantum | bc_relaxed (BC-seeded muon, EXP-002) | `bc_relaxed/pp_bc_seeded` | 450k | **−90.69088 ± 0.00029** | RUNNING (5491723), slope now statistically FLAT (−8.1e-8/step) — plateauing/noise-limited; **~21.9 mHa BELOW #8 off-T −90.669**, lowest quantum diamond run; muon HOLDS BC ([[project-bc-seeded-muon-result]]) |
| #13 classical t_rel ★ | fixed @ T | t_relaxed | `classical/t_relaxed/pp` | 338k | **−90.68500 ± 0.00042** | RUNNING (5485272), slope statistically FLAT/noise-limited (−5.8e-8/step) |
| #18 t_seeded ★ | quantum | t_relaxed (T-seeded, EXP-003b) | `t_seeded/pp` | 42k | warm-up (climbing, −90.36 inst) | RUNNING (5491822), FRESH net; climbing cleanly out of warm-up |

UPDATE 2026-07-08: **★ #14 bc_seeded @580k = −90.69597 ± 0.00019 E_h — now CONVERGED** (slope flat
−3.5e-8/step, drift < tol). Firmly lowest quantum diamond, **~26.9 mHa BELOW #8 off-T −90.66904**, tied
with classical fixed-T #7 −90.69624 (Δ≈0.3 mHa < SEM). This is the seed-INDEPENDENT variational proof that
BC (not off-T) is the muon site — retires the "off-T is variationally lower" objection that held at the
un-converged 176k read. See [[project-bc-seeded-muon-result]] + repo `localization_optimization_interplay.md`.

UPDATE 2026-07-07: daily check. **★ #14 bc_seeded @570k = −90.69682 ± 0.00029 — now edged just BELOW
classical fixed-T #7 −90.69624 (~0.6 mHa, within 2 SEM) and ~28 mHa below #8 off-T; firmly lowest quantum
diamond, still descending.** ⚠️ **#13 classical t_rel had a SOFT BLOW-UP** @430–432k (spiked to −90.05, min
−47.6), recovered only to −90.658 @460k → **RESTARTED from last clean ckpt 428000** (−90.696; cancelled
5519016, post-428k ckpts moved to `pp/post430k_spike_bak_20260707/`, csv truncated ≤428000, resubmitted
5529411). 🔴 **#18 t_seeded HUNG** @121210 ~12h → cancelled 5519015, resubmitted 5529408 (restores ~120k).
**#19 EXP-004 wide-burnin RUNNING** (5529283, home env, @warm-up, muon width 0.3 confirmed — see
[[env-kfac-jax-fresh-net-crash]]). 5529408/5529411 PENDING (AssocGrpCPUMinutesLimit hold). None converged.

UPDATE 2026-07-06 (#2): daily check. 5 diamond/si jobs RUNNING; queued 5 afterany follow-ups (5519016→#13,
5519017→#14). **★ #14 bc_seeded @526k = −90.6947 (500–520k block; tool trailing # is an incomplete-final-block
artifact — real trajectory monotonic descending: 480-500k −90.69343 → 500-520k −90.69465) — now ~25.5 mHa
BELOW #8 off-T −90.669, firmly the lowest quantum diamond run.** #13 classical t_rel @416k = −90.69097 ±
0.00023 — tool says CONVERGED (window) but dropped ~6 mHa run-over-run ⇒ plateau-APPROACHING, NOT settled.
**#18 t_seeded TIMED OUT clean @112k (−90.532 warm-up), RESUBMITTED plain 5519015.** None converged.

UPDATE 2026-07-05: daily check — all 6 active jobs RUNNING & healthy (csv fresh ≤40 min, NaN
≤7/last-2000). Only #18 had a pending follow-up → queued 5 afterany (5501849→#11, 5501850→#13,
5501851→#14, 5501852→#15, 5501853→#17). Diamond active rows: **★ #14 bc_seeded @450k = −90.69088 ±
0.00029 — slope now STATISTICALLY FLAT (−8.1e-8 < 2×4.7e-8), plateauing/noise-limited; now ~21.9 mHa
BELOW #8 off-T −90.669 and the lowest quantum diamond run, closing on classical fixed-T −90.696
(Δ−3.9 mHa this check).** #13 classical t_rel @338k = −90.68500 ± 0.00042 (−5.8e-8/step, noise-limited).
**★ #18 diamond T-seeded @42k climbing cleanly out of warm-up** (−90.36 inst, ewmean −90.32, NaN
warm-up only). None converged.

UPDATE 2026-07-04 (#2): daily check — all 6 active jobs RUNNING & healthy (csv fresh ~20 min, NaN
≤5/last-2000), each ALREADY had one afterany follow-up PENDING ⇒ NO requeue. Diamond active rows:
**★ #14 bc_seeded @414k = −90.68701 ± 0.00025 (−1.1e-7/step) — now ~18 mHa BELOW #8 off-T −90.669,
BC-vs-off-T verdict keeps strengthening (Δ−2.0 mHa this check).** #13 classical t_rel @300k =
−90.68258 ± 0.00022 (−6.4e-8/step, slope now statistically FLAT/noise-limited). **#18 diamond T-seeded
NEW: fresh-net sanity PASSED** ("Training new model", −88 climbing @6k, 1.52s/step, no sustained NaN;
seed cubic-frac check owed at inference). None converged.

UPDATE 2026-07-04: daily check. All 6 active jobs RUNNING & healthy (csv fresh, NaN ≤4/last-2000).
#13 (5485263) & #15 resumed cleanly from 07-03's hang/crawl (#13 now @278k, 1.48s/step, past its
263k hang point). Queued 4 afterany follow-ups for the jobs that lacked them (5491721→#11, 5491722→#12,
5491723→#14, 5491724→#17); #13/#15 already had 5485272/5485273 pending. Diamond active rows: **★ #14
bc_seeded @394k = −90.68506 ± 0.00030 (−1.2e-7/step) — now ~16 mHa BELOW #8 off-T −90.669, BC-vs-off-T
verdict continues to strengthen (Δ−6.1 mHa this check, still descending).** #12 t_relaxed @478k =
−90.65979 ± 0.00023 (−6.2e-8/step, slope now statistically flat/noise-limited). #13 classical t_rel
@278k = −90.67941 ± 0.00030 (−1.8e-7/step, descending). None converged (by tol).

UPDATE 2026-07-03: daily check. **#13 (5463405) HUNG** (frozen @263419 since 01:22 UTC, ~10h,
holding 8 nodes) → cancelled + resubmitted 5485263 (resumes from ckpt). #12/#14 healthy (fresh csv,
1.5s/step). Queued afterany follow-ups (5485269→#12, 5485270→#14, 5485272→#13). Diamond active rows:
**★ #14 bc_seeded @356k = −90.67900 ± 0.00041 (−2.4e-7/step) — stays ~10 mHa BELOW #8 off-T −90.669**
(trailing mean +0.7 mHa this check = within SEM, slope still descending). #12 t_relaxed @440k =
−90.65759 ± 0.00023 (−8.2e-8/step). #13 classical t_rel @263.4k = −90.67705 ± 0.00027 (value frozen at
hang, still valid block-avg). None converged.

UPDATE 2026-07-02 (#2): daily check — all 6 active jobs RUNNING (5457805/06/08/14/15/17, 6.5–8h
left) and each ALREADY had one correctly-chained afterany follow-up PENDING (5463403→#11,
5463404→#12, 5463405→#13, 5463406→#14, 5463407→#15, 5463408→#17) ⇒ NO requeue needed. All healthy
(csv fresh 13:2x–13:5x, NaN ≤6/last-2000 = warm-up). Diamond active rows: **★ #14 bc_seeded @306k =
−90.67965 ± 0.00025 (−1.5e-7/step) — now ~10.6 mHa BELOW #8 off-T −90.669, BC-vs-off-T verdict
strengthening further.** #12 t_relaxed @390k = −90.65373 ± 0.00022 (−1.0e-7/step). #13 classical
t_rel @236k = −90.67246 ± 0.00030 (−1.8e-7/step). None converged; all still descending.

UPDATE 2026-07-02: ALL 6 active jobs RESUBMITTED & RUNNING again on `parv` (5457805/06/08/14/15/17;
queue had NO pending follow-ups). Path note: account migration was resolved by RE-OWNING
`/projects/u6em/parv` to the current account (`parvfection.u6em`, group brics.u6em) — NOT by
copying to a new `/projects/u6em/parvfection` base, so `PATH_MIGRATION.md` is now obsolete and all
configs still (correctly) point to `parv`. Diamond active rows: **★ #14 bc_seeded @280k = −90.67555
± 0.00039 (−2.4e-7/step) — now ~6.5 mHa BELOW #8 off-T −90.669, BC-vs-off-T verdict strengthening.**
#12 t_relaxed @366k = −90.65141 ± 0.00024 (−1.0e-7/step, descending). #13 classical t_rel @210k =
−90.66638 ± 0.00046 (−3.7e-7/step, descending). None converged; NaN ≤4/last-2000 (warm-up level).

UPDATE 2026-07-01: ALL diamond active jobs DEAD (queue empty; follow-ups FAILED exit 6:0 node
abort ~08:23–08:34, not clean TIMEOUT; ckpts intact; NOT resubmitted per user). **★ #14
bc_seeded @269.6k = −90.67340 ± 0.00037 (−1.8e-7/step) has CROSSED #8 off-T −90.669 — now
~4.5 mHa BELOW it, so the BC-holding muon is lower-E than the off-T trap (verdict favours BC).**
#12 t_relaxed @352k = −90.65002 ± 0.00032 (−1.4e-7/step, descending). #13 classical t_rel
@198k = −90.66204 ± 0.00045 (−3.2e-7/step, descending). None converged.

UPDATE 2026-06-30 (#3): #14 bc_seeded @232k = −90.66724 ± 0.00044 (−2.0e-7/step, still
descending), now only ~1.8 mHa from #8 off-T −90.669 — about to cross it. #12 t_relaxed @316k
= −90.64729 ± 0.00041 (−2.4e-7/step, descending). #13 classical t_rel @161k = −90.64973 ±
0.00048 (−6.0e-7/step, descending). None converged.

UPDATE 2026-06-30 (#2): #14 bc_seeded @218k = −90.66541 ± 0.00045 (−3.5e-7/step,
still descending), now only ~3.6 mHa from #8 off-T −90.669 — closing on a BC-vs-off-T
verdict. #12 t_relaxed @302k = −90.64346 ± 0.00042 (−1.4e-6/step, descending). #13
classical t_rel @146k = −90.64169 ± 0.00050 (−1.2e-6/step, warm-up done, descending).
None converged.

UPDATE 2026-06-30: #14 bc_seeded @200k = −90.65489 ± 0.00094 (−1.7e-7/step, slope
flat-ish/noise-limited), ~14 mHa from #8 off-T. #12 t_relaxed @284k = −90.64201 ±
0.00034 (−1.4e-7/step, still descending). #13 classical t_rel @128k = −90.61454 ±
0.00305 (slope flat, noise-limited). None converged.

UPDATE 2026-06-29 (late): #14 bc_seeded @177k = −90.64741 ± 0.00067 (−4.5e-7/step),
~22 mHa from #8 off-T. #12 t_relaxed @248k = −90.63569 ± 0.00033 (−2.1e-7/step).
#13 classical t_rel @90k = −90.55766 ± 0.00377 (−2.8e-6/step, still warm-up). None
converged.

UPDATE 2026-06-29: refreshed active rows. #14 bc_seeded @176k = −90.64681 ±
0.00070 (−4.7e-7/step), +44 mHa since 06-28, now only ~22 mHa above #8 off-T
−90.669 and still descending — still too early for a BC-vs-off-T energy verdict.
#12 t_relaxed @232k = −90.63158 ± 0.00038 (−2.6e-7/step), +21 mHa since 06-28.
#13 classical t_rel @74k = −90.50608 ± 0.00507 (−3.8e-6/step) — out of the
worst warm-up spikes, tool gives a clean (if loose) number now. None converged.

UPDATE 2026-06-28: refreshed active rows. #12 t_relaxed q @181k = −90.61085 ±
0.00082 (−5.9e-7/step), up ~60 mHa from 06-27's −90.551 @130k. #14 bc_seeded
@124k = −90.60321 ± 0.00189 (−1.4e-6/step), climbed ~123 mHa from 06-27's −90.481
@74k → now within ~66 mHa of #8's off-T plateau −90.669, still descending; far
too early for a BC-vs-off-T energy verdict. #13 diamond classical t_relaxed (NEW,
fresh net launched 06-27) @21k still in warm-up — tool gives garbage (−89.54±0.23
from early spikes); live inst −90.28. Neither active quantum run converged.

PRIOR 2026-06-27: #7 classical-T stays CONVERGED at −90.69624 ± 0.00015 @472k
(slope flat). #12 t_relaxed q now at 130k — early VMC spikes are GONE, so
energy_convergence.py gives a clean −90.55055 ± 0.00493 directly (no robust hack
needed), still descending −3.5e-6/step, NOT settled. #14 bc_seeded (EXP-002) at
74k = −90.48064 ± 0.00524, still descending fast (−4.0e-6/step) → must climb
~190 mHa more to rival #8's −90.669; far too early for a BC-vs-off-T energy
verdict. Its position-inference (job 5391700/5391609) answers the BC-site
question, not the energy yet.

PRIOR (2026-06-26): #7 converged @472k; #12 @112k still had VMC spikes (tool
garbage); #14 @56k −90.429 descending.

EARLIER (2026-06-24 eve): #8 bc_rel_2 STOPPED & marked completed at final
−90.66904 ± 0.00029 @280196 — clearly the lowest quantum run (~5.9 mHa below
converged #6 unrelaxed −90.66312), BUT its muon sat at OFF-T, not the BC site it
was meant to test, so #14 bc_seeded (EXP-002) now carries the BC question.

base path for all: `/projects/u6em/parv/diamond/unpaired/`

### How to refresh
- **Don't re-run the 3 frozen/converged rows** (#4, #6, #5) — their training is
  stopped/closed, values above are final. Just reuse them.
- **Re-run only the ACTIVE runs** #7 and #8 (both still descending), e.g.
  `python tools/energy_convergence.py /projects/u6em/parv/diamond/unpaired/classical/T_site/pp`
  (ferminet-piku env, [[feedback-python-env]]). Their numbers are upper bounds
  until they plateau.

### Interpretation (state every time)
1. **Classical (#4, #7) vs quantum (#5,#6,#8) are NOT directly comparable** —
   classical fixes the muon as a point charge → no muon zero-point energy → sits
   artificially lower. Different quantity, not a "better" energy.
2. **Among quantum runs:** by *final plateau* #6 unrelaxed (−90.663) < #8 new BC
   (−90.654, still falling) < #5 old BC (−90.598). BUT this final-plateau ordering
   of #6 vs #8 is PREMATURE — it compares converged #6 (522k) to half-trained #8
   (216k). On a **matched-step** basis #8 is consistently ~10–28 mHa LOWER than #6
   and still dropping faster (see [[project-diamond-unrelaxed-vs-bcrel2-matched-step]]).
   Do NOT cite #6<#8 as settled until #8 reaches ~500k steps. (#5 old BC being
   highest is robust.) [[research-goal-muon-site]], [[project-bc-relaxed-pp-muon-result]].
3. **#8 vs #5 (the BC-relaxation comparison):** #8 (charged-state re-relaxed,
   more-expanded BC) is already ~33 mHa LOWER than #5's final plateau at only
   half the steps and still descending → the new BC geometry is genuinely more
   stable than the original BC-relaxed coords.
