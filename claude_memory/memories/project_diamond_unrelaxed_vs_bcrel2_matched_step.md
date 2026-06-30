---
name: project-diamond-unrelaxed-vs-bcrel2-matched-step
description: "Matched-step energy-curve comparison of diamond #6 unrelaxed q vs #8 bc_rel_2 q — at equal steps #8 is LOWER, so the final-plateau verdict that #6 is preferred is premature"
metadata:
  node_type: memory
  type: project
  originSessionId: matched-step-compare
---

## Diamond quantum muon: #6 unrelaxed vs #8 bc_rel_2 — matched-step comparison (2026-06-23)

Compared matched-step energies for the two quantum-muon runs, to control for the
fact that #6 has more training steps than #8. save_paths:
`diamond/unpaired/unrelaxed/pp` (#6, converged 522k) and
`diamond/unpaired/bc_relaxed/pp_relax_2` (#8, still running). NOTE: use the
smoothed **ewmean** column (col 3) — the raw `energy` col (col 2) has positive
outliers that wrongly inflate block-average means (e.g. raw plateau avg for #6
came out −90.56 vs true −90.663).

Refreshed 2026-06-23 with ewmean (±500-step avg), #8 now at step 228k (job
5344740 RUNNING):

| step | #6 unrelaxed ewmean | #8 bc_rel_2 ewmean | gap (8−6) mHa |
|---|---|---|---|
| 60k | −90.4355 | −90.4547 | −19 |
| 100k | −90.5534 | −90.5584 | −5 |
| 140k | −90.6017 | −90.6163 | −15 |
| 180k | −90.6211 | −90.6444 | −23 |
| 215k | −90.6318 | −90.6584 | −27 |
| 228k | −90.6352 | −90.6604 | −25 |

#6 converged plateau (515–522k) ewmean = **−90.6632**. So #8 at only 228k has
already essentially matched #6's fully-converged energy, and is still dropping.

**Key finding: at EVERY matched step #8 (bc_rel_2) sits ~10–28 mHa LOWER than #6
(unrelaxed).** #8 is on the lower/steeper trajectory throughout.

But #6 trained far longer and kept descending past #8's current point:
#6 step 215k=−90.632 → 260k=−90.642 → 320k=−90.651 → 400k=−90.656 →
460k=−90.660 → 510k=−90.663 (final plateau −90.66312±0.00017). So #6 gained
~31 mHa over steps 215k→510k.

**Caveat this OVERRIDES in [[project-diamond-pp-energy-comparison]]:** the
final-plateau table compares #6's converged −90.663 against #8's *current*
−90.654 and concludes the quantum muon "energetically prefers the unrelaxed/T
config." That verdict is PREMATURE — it compares converged #6 to a half-trained
#8. On a matched-step basis #8 is consistently lower and still dropping at
−3.5e-7/step (steeper than #6 was at the same step). #8 must be trained to a
comparable step count (≥~500k) before the unrelaxed-vs-bc_rel_2 energetic
ordering can be called. Do NOT cite #6<#8 as settled until #8 plateaus.
See [[research-goal-muon-site]].
