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
| #11 t_relaxed ★ | fixed @ T | t_relaxed | `classical/t_relaxed` | 200k | **−62.91709 ± 0.00016** | DEAD 2026-07-01 (node abort), NOT CONVERGED (−8.0e-8/step), ~4 mHa BELOW frozen #2; not resubmitted per user |
| #9 bc_relaxed | quantum | bc_relaxed (charged-state, Si pushed ~0.77 bohr along [111]) | `bc_relaxed` | 138k (frozen) | **−62.86382 ± 0.00012** (slope ~flat −4e-7) | **STOPPED 2026-06-27 @138000 (frozen)** — matched #1 unrelaxed (~0.2 mHa); muon drifted to T cage so closed |
| #1 unrelaxed | quantum | unrelaxed | `unrelaxed` | 180k (frozen) | **−62.86151 ± 0.00047** | CLOSED/cancelled @180000 (frozen; was still drifting down) |
| #10 t_relaxed | quantum | t_relaxed | `t_relaxed` | 138k (frozen) | **−62.87133 ± 0.00048** @130k | **STOPPED 2026-06-29** — muon avoided relaxed cage (EXP-003); apparent ~10 mHa edge over #1 was convergence-inflated (matched-step classical #11-vs-#2 ⇒ true relaxation E ≈5–6 mHa). Superseded by T-seeded #17 |
| #17 t_seeded ★ | quantum | t_relaxed (T-seeded) | `t_seeded` | 55.8k | **−62.80028 ± 0.00379** | DEAD 2026-07-01 (node abort), fresh-net still climbing (−2.8e-6/step) — EXP-003b, muon seeded at relaxed T-site (0.75a)³ width 0.5 |
| #15 bc_seeded ★ | quantum | bc_relaxed (BC-seeded muon, EXP-002) | `bc_seeded` | 82k | **−62.85292 ± 0.00128** | DEAD 2026-07-01 (node abort); ✅ FULLY RECOVERED, back in normal Si quantum range (~−62.86), regression gone |
| #16 classical BC | fixed @ BC | bc_relaxed | `classical/bc_relaxed` (planned) | — | — | NOT STARTED — config/script not yet written (silicon analogue of diamond #4) |

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
