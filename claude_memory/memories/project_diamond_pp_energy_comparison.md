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
| #12 t_relaxed ★ | quantum | t_relaxed | `t_relaxed/pp` | 352k | **−90.65002 ± 0.00032** | DEAD 2026-07-01 (node abort), NOT CONVERGED still descending (−1.4e-7/step) — not resubmitted per user |
| #14 bc_seeded ★ | quantum | bc_relaxed (BC-seeded muon, EXP-002) | `bc_relaxed/pp_bc_seeded` | 269.6k | **−90.67340 ± 0.00037** | DEAD 2026-07-01 (node abort), NOT CONVERGED still descending (−1.8e-7/step); **CROSSED #8 off-T −90.669, now ~4.5 mHa BELOW**; muon HOLDS BC ([[project-bc-seeded-muon-result]]) |
| #13 classical t_rel ★ | fixed @ T | t_relaxed | `classical/t_relaxed/pp` | 198k | **−90.66204 ± 0.00045** | DEAD 2026-07-01 (node abort), NOT CONVERGED still descending (−3.2e-7/step) |

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
