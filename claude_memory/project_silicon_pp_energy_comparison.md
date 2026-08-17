---
name: project-silicon-pp-energy-comparison
description: "Silicon PP-run GSE comparison table + how to refresh it; use whenever the user asks to compare silicon energies"
metadata:
  node_type: memory
  type: project
  originSessionId: silicon-energy-compare
---

## Silicon PP runs — ground-state energy comparison

When the user asks for "silicon energy comparison" / "GSE of silicon PP runs",
produce a table of plateau energies from `tools/energy_convergence.py`
([[tool-energy-convergence]]) over the [[job-save-paths]] dirs. Values below
are plateau (trailing-window mean) ± 1 SEM unless noted.

★ marks ACTIVE (still-training) runs; unmarked rows are frozen/closed/not-started.
Always keep this ★-on-active convention ([[feedback-mark-active-runs-with-star]]).

| Run | Muon | Geometry | save_path tail | Step | Plateau E (E_h) | Status (as of 2026-06-27) |
|---|---|---|---|---|---|---|
| #2 classical T-site | fixed @ T | unrelaxed | `classic` | 224k (frozen) | **−62.91337 ± 0.00015** | FINISHED/cancelled @223695 (frozen; slope still slightly down at end) |
| #11 t_relaxed | fixed @ T | t_relaxed | `classical/t_relaxed` | 306k (blew up) | **−62.924** (last good @306k) | **BLEW UP & CLOSED 2026-07-06** — instant divergence @step 306603 (E→1.4e10), job 5501849 FAILED; last clean ckpt 306000; user LEFT STOPPED (~11 mHa below frozen #2, tight upper bound) |
| #9 bc_relaxed | quantum | bc_relaxed (charged-state, Si pushed ~0.77 bohr along [111]) | `bc_relaxed` | 138k (frozen) | **−62.86382 ± 0.00012** (slope ~flat −4e-7) | **STOPPED 2026-06-27 @138000 (frozen)** — matched #1 unrelaxed (~0.2 mHa); muon drifted to T cage so closed |
| #1 unrelaxed | quantum | unrelaxed | `unrelaxed` | 180k (frozen) | **−62.86151 ± 0.00047** | CLOSED/cancelled @180000 (frozen; was still drifting down) |
| #10 t_relaxed | quantum | t_relaxed | `t_relaxed` | 138k (frozen) | **−62.87133 ± 0.00048** @130k | **STOPPED 2026-06-29** — muon avoided relaxed cage (EXP-003); apparent ~10 mHa edge over #1 was convergence-inflated (matched-step classical #11-vs-#2 ⇒ true relaxation E ≈5–6 mHa). Superseded by T-seeded #17 |
| #17 t_seeded ★ | quantum | t_relaxed (T-seeded) | `t_seeded` | 154k | **−62.88074 ± 0.00033** | RUNNING (5491724), still descending (−2.3e-7/step) — EXP-003b; **LOWEST quantum Si run, now ~9.4 mHa BELOW frozen #10 −62.87133** |
| #15 bc_seeded ★ | quantum | bc_relaxed (BC-seeded muon, EXP-002) | `bc_seeded` | 126k | **−62.87699 ± 0.00053** | RUNNING (5485273), node healthy; descending (−3.8e-7/step), normal Si range |
| #16 classical BC ★ | fixed @ BC | bc_relaxed | `classical/bc_relaxed` | 0 (fresh) | warm-up | LAUNCHED 2026-07-05 (job 5501873, follow-up 5501876), fresh net; silicon analogue of diamond #4 (fixed-H @ BC midpoint (0.125a)³, 2.994 bohr from flanking Si #0/#8) |

UPDATE 2026-07-08: refreshed active rows. **★ #17 t_seeded @224k = −62.89124 ± 0.00051 (noise-limited,
slope flat +1.0e-7/step) — lowest quantum Si, ~17.3 mHa BELOW frozen unseeded #10 t_relaxed −62.87392.**
★ #15 bc_seeded @196k = −62.89170 ± 0.00022 (still descending −1.2e-7/step), ~tied with #17. KEY READ:
seeding the muon INTO the contracted relaxed T cage (#17) is ~17 mHa LOWER than the unseeded run (#10)
that FLED it → the flight was a basin-selection TRAP, not a ZPM preference for the roomier cage (ZPE
penalty for the tighter well is real but sub-dominant). Parallels diamond BC-seeded. Feeds repo
`localization_optimization_interplay.md` + [[project-bc-seeded-muon-result]]. None converged.

UPDATE 2026-07-07: daily check. **★ #17 t_seeded @218k = −62.89249 ± 0.00017 (Δ−4.3 mHa) — lowest quantum
Si, now ~21 mHa BELOW frozen #10 −62.87133, still descending.** #15 bc_seeded @190k = −62.89198 ± 0.00017
(Δ−4.5 mHa, descending, normal Si range). #16 classical bc_rel @58k = −62.87677 ± 0.00020 (climbing out of
warm-up; 1:57 left, follow-up 5519019 ✓). All healthy (csv fresh / .out advancing). **NEW: launched #17
t_seeded muon-POSITION inference (job 5529415, `muon_silicon_q_t_seeded_inf`) to test for a BOUND STATE
(muonium)** — cancelled the redundant bc_seeded inference (5514972) to free nodes. Files
`configs/silicon/inference_t_seeded.py` + `jobs/silicon/inference_t_seeded.sh` (restore t_seeded ckpt 218000,
positions=True). OWED: `silicon_t_seeded` case in muon_site_analysis.py + srpd/net-spin (prior Si = diamagnetic).
None converged.

UPDATE 2026-07-06 (#2): daily check. **🔴 #11 classical t_rel BLEW UP & CLOSED** — clean to step 306602
(−62.92218), instant divergence @306603 (E→1.4e10, ewvar→1.8e19), job 5501849 FAILED (exit 1:0). Ckpts
308k–330k NaN-corrupt; last clean = ckpt_306000. User chose LEAVE STOPPED (was ~−62.924 @300k, ~11 mHa
below frozen #2 — tight upper bound; recover later via truncate+move-bad-ckpts+restart-from-306000).
**★ #17 t_seeded @194k = −62.88820 ± 0.00026 (−1.6e-7/step) — lowest quantum Si, now ~16.9 mHa BELOW frozen
#10 −62.87133 (Δ−7.5 mHa).** #15 bc_seeded @167k = −62.88753 ± 0.00032 (−2.1e-7/step), descending, normal
Si range (Δ−10.5 mHa). #16 classical bc_rel @34k warm-up (restored ckpt 030510, climbing healthily).
Queued afterany follow-ups (5519018→#15, 5519019→#16, 5519020→#17). None converged.

UPDATE 2026-07-05: daily check — all active jobs RUNNING & healthy (csv fresh ≤40 min, NaN ≤6/last-2000).
Only #18 had a follow-up → queued 5 afterany (incl. 5501849→#11, 5501852→#15, 5501853→#17). #11 classical
t_rel @300k = −62.92419 ± 0.00015 (−6.9e-8/step, still >2σ from flat = plateau-approaching, ~11 mHa below
frozen #2). **★ #17 t_seeded @154k = −62.88074 ± 0.00033 (−2.3e-7/step) — firmly lowest quantum Si, now
~9.4 mHa BELOW frozen #10 −62.87133 (Δ−5.6 mHa this check).** #15 bc_seeded @126k = −62.87699 ± 0.00053
(−3.8e-7/step), descending, normal Si range (Δ−7.7 mHa). None converged.

UPDATE 2026-07-04 (#2): daily check — all active jobs RUNNING & healthy, each ALREADY had one afterany
follow-up PENDING ⇒ NO requeue. #11 classical t_rel @280k = −62.92281 ± 0.00016 (−6.8e-8/step), still
NOT settled (slope >2σ), ~9.4 mHa below frozen #2. **★ #17 t_seeded @134k = −62.87512 ± 0.00049
(−3.5e-7/step) — now edged BELOW frozen #10 −62.87133, firmly the lowest quantum Si run (Δ−4.6 mHa this
check).** ✅ #15 bc_seeded @106k = −62.86925 ± 0.00060 (−4.5e-7/step), node healthy, +12k steps, normal
Si range. None converged.

UPDATE 2026-07-04: daily check. All active jobs RUNNING & healthy. **✅ #15 bc_seeded RESUBMIT WORKED —
now 2.79s/step (was 30.1s/step crawl on the degraded node), @94k = −62.86272 ± 0.00075 (−6.4e-7/step),
+8.4k steps since resubmit, in normal Si range.** #11 classical t_rel @268k = −62.92232 ± 0.00013
(−1.5e-8/step) — still CONVERGED per tool, ~9.0 mHa below frozen #2. **★ #17 t_seeded @122k =
−62.87053 ± 0.00056 (−4.1e-7/step) — now the LOWEST-energy quantum Si run, ≈ matching frozen #10
−62.87133; Δ−10.8 mHa this check, still descending.** Queued afterany follow-ups (5491721→#11,
5491724→#17; #15 already had 5485273). None converged except #11.

UPDATE 2026-07-03: daily check. **#15 (5463407) CRAWLING at 30.1s/step (~20× slow, degraded node,
eta ~283 days)** — root cause of the prior 2 checks' "abnormally slow step advance" (alive/writing, NOT
a hang) → cancelled + resubmitted 5485264 (resumes from ckpt, hopefully healthy nodes). #11/#17 healthy
(1.5–2.8s/step). Queued afterany follow-ups (5485268→#11, 5485271→#17, 5485273→#15). #11 classical t_rel
@246k = −62.92093 ± 0.00013 (−2.9e-8/step) — **now CONVERGED per tool**, ~7.6 mHa below frozen #2. **★ #17
t_seeded @100k = −62.85974 ± 0.00083 (−6.2e-7/step) — CAUGHT UP to normal Si quantum range, ≈ matching #15.**
#15 bc_seeded @85.6k = −62.85719 ± 0.00093 (−7.7e-7/step), in Si range. Watch: does resubmitted #15 land
on healthy nodes (~1.5s/step)?

UPDATE 2026-07-02 (#2): daily check — all active jobs RUNNING with correctly-chained afterany
follow-ups already PENDING (no requeue needed). #11 classical t_rel @220k = −62.91902 ± 0.00017
(−9.6e-8/step), ~5.6 mHa below frozen #2 −62.91337. **✅ #15 bc_seeded @83k = −62.85610 ± 0.00101
(−8.1e-7/step)** — holding in normal Si range, recovery intact; ⚠️ step only advanced 82k→83k
across the last two checks (~2k steps in ~24h) — abnormally slow, WATCH next check. #17 t_seeded
@74k = −62.83555 ± 0.00214 (−1.5e-6/step, fresh-net climbing, +22 mHa this check). None converged;
NaN ≤3/last-2000 (warm-up level).

UPDATE 2026-07-02: ALL active jobs RESUBMITTED & RUNNING again on `parv` (queue had NO pending
follow-ups). #11 classical t_rel @206k = −62.91778 ± 0.00016 (−8.7e-8/step), ~4 mHa below frozen #2
−62.91337. **✅ #15 bc_seeded @82k = −62.85512 ± 0.00108 (−8.4e-7/step)** — holding in normal Si
quantum range (~−62.86), recovery intact; NOTE step barely advanced from 07-01's 82k (slow restart
— watch progress next check). #17 t_seeded @60k = −62.81381 ± 0.00252 (−2.0e-6/step, fresh-net
climbing). None converged; NaN ≤4/last-2000 (warm-up level). (Path: `parv` re-owned to current
account, PATH_MIGRATION.md obsolete — see [[project-diamond-pp-energy-comparison]] 07-02 note.)

UPDATE 2026-07-01: ALL silicon active jobs DEAD (queue empty; follow-ups FAILED exit 6:0 node
abort ~08:44/09:01/09:20, not clean TIMEOUT; ckpts intact; NOT resubmitted per user). #11
classical t_rel @200k = −62.91709 ± 0.00016 (−8.0e-8/step), ~4 mHa below frozen #2 −62.91337.
**✅ #15 bc_seeded FULLY RECOVERED** — @82k = −62.85292 ± 0.00128 (−9.5e-7/step), back in the
normal Si quantum range (~−62.86), post-blowup regression gone. #17 t_seeded @55.8k =
−62.80028 ± 0.00379 (−2.8e-6/step, fresh-net climbing fast). None converged.

UPDATE 2026-06-30 (#3): #11 classical t_rel @179.7k = −62.91549 ± 0.00023 (−1.2e-7/step),
~2 mHa BELOW frozen #2 −62.91337. **✅ #15 bc_seeded recovery STILL HOLDING** — @60k =
−62.83710 ± 0.00029 descending cleanly (−2.3e-6/step), no re-blow-up; ~30 mHa from proper Si
GSE. #17 t_seeded @34k = −62.75416 ± 0.00037 (−7.7e-6/step, fresh-net climbing fast). None
converged.

UPDATE 2026-06-30 (#2): #11 classical t_rel @172k = −62.91564 ± 0.00024 (−1.8e-7/step),
now ~2 mHa BELOW frozen #2 −62.91337 (both noise-level). **✅ #15 bc_seeded recovery
HOLDING** — @54k = −62.82833 ± 0.00032 descending healthily (−3.2e-6/step), did NOT
re-blow-up; now ~35 mHa from proper Si GSE. #17 t_seeded @26k = −62.70285 ± 0.00042
(−1.6e-5/step, fresh-net warm-up, climbing fast). None converged.

UPDATE 2026-06-30: #11 classical t_rel @160k = −62.91155 ± 0.00024 (−1.5e-7/step),
~2 mHa from frozen #2. **✅ #15 bc_seeded RECOVERED** — truncate+restart from ckpt
034000 (job 5426315) worked: @42k = −62.74712 ± 0.00783 descending healthily
(−5.9e-6/step), did NOT re-blow-up past the old 35–40k failure point. #17 t_seeded
@16k = −62.379 ± 0.089 (fresh-net warm-up, climbing). None converged.

UPDATE 2026-06-29 (late): #11 classical t_rel @140k = −62.90756 ± 0.00035
(−2.4e-7/step), ~6 mHa from frozen #2. **⚠️ #15 bc_seeded @62k = −62.12045 ±
0.00473 — regression CONFIRMED, NOT recovered:** stuck ~740 mHa ABOVE proper Si GSE
(all other Si runs ~−62.86–62.91), this-job span flat at −62.12 (61050→63138).
Trailing slope now mildly NEGATIVE (−3.1e-6/step) i.e. inching down from a badly
destabilized state, nowhere near healthy. User chose LEAVE-running-and-watch (not
kill). #17 t_seeded launched (5419590 PENDING, no train_stats yet). #10 q t_relaxed
CLOSED/frozen −62.87133. None of the active runs converged.

UPDATE 2026-06-29: refreshed active rows. #10 q t_relaxed @130k = −62.87133 ±
0.00048 (−3.4e-7/step), now ~10 mHa BELOW frozen #1 unrelaxed (−62.86151) and #9
bc_relaxed (−62.86382) — the quantum Si geometries are NO LONGER tied; t_relaxed
is winning. #11 classical t_rel @132k = −62.90594 ± 0.00032 (−2.1e-7/step), ~8 mHa
from frozen #2 (−62.91337). **⚠️ #15 bc_seeded @54k = −62.19466 ± 0.04245 DRIFTING
UP** (slope +2.1e-5/step, block|dE| mean 0.039): this-job span 30441 (−62.751) →
54904 (−62.155), ΔE +0.60 WORSE — fresh-net warm-up but getting worse not settling;
if it keeps climbing the BC-seeded Si net may be destabilizing. Watch next check.

UPDATE 2026-06-28: refreshed active rows. #10 t_relaxed q @103k = −62.85998 ±
0.00089 (−6.6e-7/step) — now caught up to frozen #1 unrelaxed (−62.86151) and #9
bc_relaxed (−62.86382); the three quantum Si geometries are converging to ~the
same energy. #11 classical t_rel @104k = −62.89578 ± 0.00061 (−4.4e-7/step), still
~17 mHa above frozen #2 (−62.91337). #15 bc_seeded (job 5395609) now RUNNING and
producing data @26k, but fresh-net warm-up → tool gives spiky −62.57±0.03; live
inst −62.74. Neither active run converged.

PRIOR 2026-06-27 (eve): #9 bc_relaxed STOPPED/frozen at −62.86382 ± 0.00012
@138000 (slope ~flat) — statistically tied with converged #1 unrelaxed
(−62.86151, ~0.2 mHa) → the two quantum geometries are energetically
indistinguishable in silicon. Closed because position inference showed the muon
drifted to a T cage, not BC ([[project-silicon-bc-relaxed-muon-result]]). #10/#11
t_relaxed advanced to ~81-82k, still descending, numbers essentially unchanged:
#10 q −62.84158 ± 0.00158 (−1.2e-6/step), #11 classical −62.88349 ± 0.00109
(−8.2e-7/step) — neither converged. #15 silicon bc_seeded (job 5395609, width
0.45) is still PENDING in queue — no train_stats.csv yet. NOTE: the tool's
early-spike problem is gone for #9/#10/#11 now — energy_convergence.py gives clean
numbers directly.

base path for all: `/projects/u6em/parv/silicon_unpaired/`

### How to refresh
- **Don't re-run the 2 frozen rows** (#2, #1) — training stopped/cancelled,
  values above are final. Reuse them.
- **Re-run only #9** (active, fresh). IMPORTANT: `energy_convergence.py` gives
  GARBAGE for #9 (e.g. −79010 ± 78948 E_h) because its early `train_stats.csv`
  has huge VMC outlier spikes (block max ~1.6e6 E_h) that blow up the block
  average. Use a robust recent estimate instead — median of the clean last-10k
  steps after rejecting |E+62.8|>5 outliers (spikes are all early; last 10k are
  0% spikes). Cross-check against the live `.out` log. #9 is far from
  converged → treat as a loose upper bound.

### Interpretation (state every time)
1. **Classical (#2) vs quantum (#1,#9) are NOT directly comparable** — classical
   fixes the muon → no muon zero-point energy → sits lower (−62.913). Different
   quantity, not "better".
2. **#1 carries the NaN-`ewvar` bug** ([[potential-bugs]]) but its energy column
   is fine, so its energy estimate is valid.
3. **Too early for a geometry conclusion:** among quantum runs #1 unrelaxed
   (−62.862 @180k) is below #9 bc_relaxed (−62.804 @61k), but #9 is far from
   converged. Revisit once #9 plateaus.
4. **Physics context:** silicon muon is DIAMAGNETIC (no muonium); classical SRPD
   matches quantum — key contrast with diamond
   ([[project-silicon-quantum-muon-result]]).

Sibling: [[project-diamond-pp-energy-comparison]] (same format for diamond).
